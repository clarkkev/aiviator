import einops
import flax
import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn

import configure
import featurization
import game


@flax.struct.dataclass
class ModelOutput:
  logits: jnp.ndarray
  reroll_logits: jnp.ndarray
  anticipation_logits: jnp.ndarray
  value: jnp.ndarray
  anticipation_reroll: jnp.ndarray

  @jax.jit
  def get_top_action(self):
    action_id = jnp.argmax(self.logits, -1)
    rerolls = jnp.argmax(self.reroll_logits, -1)
    anticipation_id = jnp.argmax(self.anticipation_logits, -1)
    return self.make_action(action_id, rerolls, anticipation_id)

  @jax.jit
  def sample_action(self, key: jax.random.PRNGKey):
    reroll_key, action_key = random.split(key)
    action_id = random.categorical(action_key, self.logits)
    rerolls = random.categorical(reroll_key, self.reroll_logits)
    anticipation_id = random.categorical(reroll_key, self.anticipation_logits)
    return self.make_action(action_id, rerolls, anticipation_id)

  @jax.jit
  def make_action(self, action_id, rerolls, anticipation_id):
    rerolls = jnp.where((action_id == self.logits.shape[-1] - 1)[:, None], rerolls, -1)
    anticipation_rerolls = jax.nn.one_hot(
      anticipation_id, 4, dtype=jnp.int32) * (anticipation_id < 4)[:, None]
    rerolls = jnp.where(self.anticipation_reroll[:, None], anticipation_rerolls, rerolls)
    return game.Action(action_id=action_id, rerolls=rerolls)

  @jax.jit
  def log_prob(self, action: game.Action):
    def log_probs(logits, action_id=None, mask=None):
      if mask is None:
        mask = jax.nn.one_hot(action_id, logits.shape[-1])
      return (jax.nn.log_softmax(logits) * mask).sum(-1)

    action_log_probs = log_probs(self.logits, action.action_id)
    reroll_log_probs = log_probs(self.reroll_logits, action.rerolls).sum(-1)
    reroll_log_probs *= action.rerolls.sum(-1) >= 0
    anticipation_log_probs = log_probs(self.anticipation_logits, mask=jnp.concatenate([
      action.rerolls > 0, (action.rerolls.sum(-1) == 0)[:, None]], -1))

    return jnp.where(self.anticipation_reroll, anticipation_log_probs,
                     action_log_probs + reroll_log_probs)


class Model(nn.Module):
  config: configure.Config

  @nn.compact
  def __call__(self, state: game.State):
    features = jax.vmap(featurization.featurize, in_axes=(None, 0))(self.config, state)

    x = jnp.concat([
      nn.Dense(self.config.hidden_dim)(features.board_features)[:, None],
      nn.Dense(self.config.hidden_dim)(features.dice_features)], axis=1)
    x = Transformer(self.config)(x)
    x = nn.RMSNorm()(x)

    state_repr, dice_reprs = x[:, 0], x[:, 1:]
    logits = nn.Dense(game.N_SLOTS)(dice_reprs)
    logits = einops.rearrange(logits, 'b nc s -> b (nc s)')

    dice_reprs = einops.rearrange(dice_reprs, 'b (n c) h -> b n c h', n=4)
    dice_reprs = dice_reprs[:, :, 2]
    reroll_logits = nn.Dense(5)(dice_reprs) - 1000 * (1 - features.reroll_mask)
    anticipation_logits = jnp.concatenate([
      nn.Dense(1)(dice_reprs).squeeze(-1) - 1000 * (1 - features.anticipation_mask),
      self.param('skip_logit', lambda _: jnp.zeros(())) * jnp.ones((dice_reprs.shape[0], 1))], -1)

    value_repr = state_repr
    if self.config.use_ally_dice_for_value:
      value_repr = state_repr + nn.Dense(self.config.hidden_dim)(features.ally_dice).sum(-2)
      value_repr += MLP(self.config)(nn.RMSNorm()(value_repr))
    value = nn.Dense(1)(value_repr).squeeze(-1)
    value *= self.param("value_scale", lambda _: jnp.array(0.01, dtype=jnp.float32))

    do_reroll = nn.Dense(1)(state_repr)
    logits = jnp.concatenate([logits, do_reroll], axis=-1)
    logits -= 1000 * (1 - features.action_mask)

    return ModelOutput(logits=logits, reroll_logits=reroll_logits,
                       anticipation_logits=anticipation_logits, value=value,
                       anticipation_reroll=state.anticipation_reroll)


class Transformer(nn.Module):
  config: configure.Config

  @nn.compact
  def __call__(self, x):
    for _ in range(self.config.num_layers):
      x = TransformerBlock(self.config)(x)
    return x


class TransformerBlock(nn.Module):
  config: configure.Config

  @nn.compact
  def __call__(self, x):
    x += MHA(self.config)(nn.RMSNorm()(x))
    x += MLP(self.config)(nn.RMSNorm()(x))
    return x


class MLP(nn.Module):
  config: configure.Config

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(self.config.mlp_dim)(x)
    x = jax.nn.gelu(x)
    x = nn.Dense(self.config.hidden_dim)(x)
    return x


class MHA(nn.Module):
  config: configure.Config

  @nn.compact
  def __call__(self, x):
    k, q, v = jnp.split(nn.Dense(3 * self.config.hidden_dim)(x), 3, axis=-1)
    k, q, v = map(lambda x: einops.rearrange(x, 'b l (h d) -> b h l d', h=self.config.num_heads),
                  (k, q, v))
    scores = jnp.einsum('b h q d, b h k d -> b h q k', q, k)
    attn = jax.nn.softmax(scores / jnp.sqrt(q.shape[-1]))
    o = jnp.einsum('b h q k, b h k d -> b h q d', attn, v)
    o = einops.rearrange(o, 'b h l d -> b l (h d)')
    return nn.Dense(self.config.hidden_dim)(o)
