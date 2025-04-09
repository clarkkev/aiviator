import functools
import os
import time

import einops
import flax
import jax
import jax.numpy as jnp
import jax.random as random
import optax
import orbax.checkpoint as ocp
from flax.training import train_state

import airports
import configure
import game
import global_stopwatch
import modeling
import utils

MAX_TURNS = 2 + 7 * (8 + 2 + 2)
#AIRPORT = airports.KEF
AIRPORT = airports.random_state


def make_score_fn():
  def get_raw_score(state: game.State):
    is_landing = (state.altitude == 0) & (state.track_left == 1)
    axis_done = (state.pilot_axis > 0) & (state.copilot_axis > 0)
    engine_done = (state.pilot_engine > 0) & (state.copilot_engine > 0)
    fuel_done = jnp.where(state.fuel_rule, state.is_filled[game.FUEL_INDEX],
                          engine_done | ~state.leak_rule)

    return (-4 * state.altitude +
            -4 * state.track_left +
            -2 * state.num_planes() +
            -0.5 * state.wind_speed() +
            jnp.sum(state.is_on) +
            0.1 * jnp.where(fuel_done, state.fuel, state.fuel - 6) +
            1.0 * (axis_done & (state.track_left > 0) &
                   ((state.min_tilt[0] != -2) | (state.max_tilt[0] != 2)) &
                   (state.tilt >= state.min_tilt[0]) & (state.tilt <= state.max_tilt[0])) +
            0.1 * is_landing * (state.pilot_engine > 0) * (6 - state.pilot_engine) +
            0.1 * is_landing * (state.copilot_engine > 0) * (6 - state.copilot_engine) +
            (is_landing & engine_done &
             (state.pilot_engine + state.copilot_engine <= state.brake_speed)) +
            (is_landing & axis_done & (state.tilt == 0)))

  r0 = get_raw_score(AIRPORT())
  rmax = get_raw_score(AIRPORT().replace(
    coffees=jnp.array(3, dtype=jnp.int32),
    rerolls=jnp.array(3, dtype=jnp.int32),
    altitude=jnp.array(0, dtype=jnp.int32),
    track_left=jnp.array(1, dtype=jnp.int32),
    is_on=jnp.array([slot.is_switch for slot in game.SLOTS]),
    approach_track=jnp.array([0, -1, -1, -1, -1, -1, -1]),
    brake_speed=jnp.array(2, dtype=jnp.int32),
    pilot_engine=jnp.array(1, dtype=jnp.int32),
    copilot_engine=jnp.array(1, dtype=jnp.int32),
    pilot_axis=jnp.array(1, dtype=jnp.int32),
    copilot_axis=jnp.array(1, dtype=jnp.int32),
    wind=jnp.array(9, dtype=jnp.int32),
  ))

  @jax.jit
  def score_state(state: game.State):
    return jnp.where(
      state.result == game.WIN, 1, 0.25 * (get_raw_score(state) - r0) / (rmax - r0))

  return score_state


@flax.struct.dataclass
class Examples:
  states: game.State
  actions: game.Action
  returns: jnp.ndarray
  advantages: jnp.ndarray
  log_probs: jnp.ndarray


def make_train_state(config: configure.Config, key: random.PRNGKey):
  state = AIRPORT()
  model = modeling.Model(config)
  params = model.init(key, jax.tree.map(lambda x: x[None], state))
  tx = optax.adam(optax.warmup_constant_schedule(
    init_value=0.0,
    peak_value=config.learning_rate,
    warmup_steps=100
  ))
  return train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=tx
  )


def make_checkpoint_manager(path):
  checkpointer = ocp.PyTreeCheckpointer()
  options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
  return ocp.CheckpointManager(os.path.abspath(path), checkpointer, options)


def load_checkpoint(train_state, path):
  checkpoint_manager = make_checkpoint_manager(path)
  device = jax.devices()[0]
  sharding = jax.sharding.SingleDeviceSharding(device)
  restore_args = jax.tree.map(
    lambda _: ocp.type_handlers.ArrayRestoreArgs(sharding=sharding), train_state)
  return checkpoint_manager.restore(checkpoint_manager.latest_step(), train_state,
                                    restore_kwargs={"restore_args": restore_args})


@functools.partial(jax.jit, static_argnums=(0,))
def train_step(config: configure.Config, train_state, examples: Examples):
  @functools.partial(jax.grad, has_aux=True)
  def grad_fn(params):
    outputs = train_state.apply_fn(params, examples.states)
    log_probs = outputs.log_prob(examples.actions)
    if config.ppo:
      ratio = jnp.exp(log_probs - examples.log_probs)
      clipped_ratio = jnp.clip(ratio, 1.0 - config.ppo_clip_ratio, 1.0 + config.ppo_clip_ratio)
      policy_loss = -jnp.mean(jnp.minimum(
        ratio * examples.advantages, clipped_ratio * examples.advantages))
    else:
      policy_loss = -jnp.mean(log_probs * examples.advantages)

    example_mask = (examples.states.result == game.SAFE)
    value_loss = jnp.mean(example_mask * jnp.square(examples.returns - outputs.value))
    entropy = -jnp.mean(example_mask * jnp.sum(
      jax.nn.softmax(outputs.logits) * jax.nn.log_softmax(outputs.logits), axis=-1))

    loss = policy_loss + config.value_weight * value_loss - config.entropy_weight * entropy
    return loss, {'loss': loss, 'policy_loss': policy_loss, 'value_loss': value_loss,
                  'entropy': entropy}

  grads, metrics = grad_fn(train_state.params)
  new_train_state = train_state.apply_gradients(grads=grads)

  return new_train_state, metrics


@functools.partial(jax.jit, static_argnums=3)
@functools.partial(jax.vmap, in_axes=(0, 0, None, None))
def get_returns_and_advantages(rewards, values, discount_factor, gae_lambda):
  def discounted_return(ret, reward):
    ret = reward + discount_factor * ret
    return ret, ret
  _, returns = jax.lax.scan(discounted_return, 0, rewards[::-1])
  returns = returns[::-1]

  if gae_lambda < 0:  # REINFORCE
    return returns, returns

  if gae_lambda == 1:  # REINFORCE with baseline
    return returns, returns - values

  def gae_advantage(advantage, delta):
      advantage = delta + discount_factor * gae_lambda * advantage
      return advantage, advantage

  deltas = rewards + discount_factor * jnp.append(values[1:], 0) - values
  _, advantages = jax.lax.scan(gae_advantage, 0.0, deltas[::-1])
  advantages = advantages[::-1]

  return returns, advantages


make_state = jax.jit(jax.vmap(AIRPORT))


@jax.jit
def trajectory_step(train_state, states, scores, key):
  outputs = train_state.apply_fn(train_state.params, states)
  if key is None:
    actions = outputs.get_top_action()
  else:
    actions = outputs.sample_action(key)
  log_probs = outputs.log_prob(actions)

  new_states = jax.vmap(game.do_action)(states, actions)
  new_scores = jax.vmap(make_score_fn())(new_states)

  rewards = new_scores - scores
  # if the action caused a loss, or the game was already over, zero-out the reward
  rewards *= (new_states.result > game.SAFE) | (states.result == game.SAFE)
  values = outputs.value * (states.result == game.SAFE)

  return new_states, new_scores, (states, actions, rewards, values, log_probs)


def collect_trajectories(
    config: configure.Config, key, train_state, collect_metrics=True):
  states = make_state(key=random.split(key, config.batch_size))
  scores = jnp.zeros(config.batch_size)

  history = []
  global_stopwatch.start("collect_trajectory")
  for _ in range(MAX_TURNS):  # replace with a scan?
    key, step_key = random.split(key)
    states, scores, result = trajectory_step(train_state, states, scores, step_key)
    history.append(result)
  global_stopwatch.stop()

  global_stopwatch.start("postprocess_trajectory")
  states, actions, rewards, values, log_probs = map(lambda x: stack_lists(x, 1), zip(*history))

  returns, advantages = get_returns_and_advantages(
    rewards, values,
    jnp.array(config.discount_factor, jnp.float32),
    config.gae_lambda)

  examples = Examples(
    states=states, actions=actions, returns=returns, advantages=advantages, log_probs=log_probs)
  _, shuffle_key = random.split(key)
  examples = make_minibatches(examples, shuffle_key, config.train_batch_size)
  global_stopwatch.stop()

  if not collect_metrics:
    return examples, None

  global_stopwatch.start("get_metrics")
  for j in range(rewards.shape[1]):
    result = states.result[0, j]
    if result != game.SAFE:
      j -= 1
      break
  final_state = jax.tree.map(lambda x: x[0, j], states)
  total_reward = float(rewards[0].sum())
  metrics = {
    "reward": 100 * total_reward,
    "altitude": float(final_state.altitude),
    "turns": float(8 * (7 - final_state.altitude) -
                  final_state.pilot_dice.num() - final_state.copilot_dice.num() + 1),
    "planes_left": float(jnp.maximum(0, final_state.approach_track).sum()),
    "landing_gear_on": float(final_state.min_speed - 5),
    "brakes_on": float((final_state.brake_speed - 1) if final_state.ice_rule else
                       (final_state.brake_speed / 2)),
    "flaps_on": float(final_state.max_speed - 8),
    "rerolls": float(final_state.rerolls),
    "result": int(result),
    "track_left": int(final_state.track_left),
    "win_percent": 100.0 if total_reward > 0.99 else 0.0
  }
  global_stopwatch.stop()

  return examples, metrics


@functools.partial(jax.jit, static_argnums=(1,))
def stack_lists(elems, axis=0):
  return jax.tree.map(lambda *args: jnp.stack(args, axis), *elems)


@functools.partial(jax.jit, static_argnums=(2,))
def make_minibatches(examples, key, batch_size):
  n_examples =  examples.returns.shape[0] * examples.returns.shape[1]
  n_batches = n_examples // batch_size
  reshaped = jax.tree.map(lambda x: einops.rearrange(x, 'n m ... -> (n m) ...'), examples)
  idx = random.permutation(key, n_examples)
  shuffled = jax.tree.map(lambda x: x[idx], reshaped)
  return [jax.tree.map(lambda leaves: leaves[i * batch_size:(i + 1) * batch_size], shuffled)
          for i in range(n_batches)]


class History:
  def __init__(self):
    self.keys = []
    self.history = []

  def append(self, metrics):
    self.keys = list(metrics.keys())
    self.history.append(list(metrics.values()))

  def get_avgs(self, last_n=0):
    avgs = jax.tree.map(lambda *args: sum(args) / len(args), *self.history[-last_n:])
    return {k: avg for k, avg in zip(self.keys, avgs)}

  def print_avgs(self, last_n=0):
    print(", ".join([f"{k}: {v:.2f}" for k, v in self.get_avgs(last_n).items()]))

  def write(self, path):
      utils.write_pickle((self.keys, self.history), path)


def main(**kwargs):
  config_kwargs = dict(write=True, overwrite=True)
  config_kwargs.update(**kwargs)
  config = configure.Config(**config_kwargs)

  key = random.PRNGKey(1)
  key, init_key = random.split(key)
  train_state = make_train_state(config, init_key)
  checkpoint_manager = make_checkpoint_manager(config.checkpoint_dir)
  if config.init_checkpoint:
    train_state = load_checkpoint(train_state, config.init_checkpoint)
  start_time = time.time()
  history = History()

  for step in range(config.num_steps):
    key, trajectory_key = random.split(key)
    examples, metrics = collect_trajectories(config, trajectory_key, train_state,
                                                 step % config.metrics_every == 0)
    global_stopwatch.start("train_step")
    for e in examples:
      train_state, train_metrics = train_step(config, train_state, e)
    global_stopwatch.stop()

    if metrics is not None:
      metrics.update(train_metrics)
      metrics["games_played"] = step * config.batch_size
      metrics["time"] = time.time() - start_time
      history.append(metrics)

    if step % config.metrics_every == 0:
      print(f"{config.experiment_name}: step {step}, ({metrics['games_played']} games), " +
            f"{metrics['time']:.2f}s elapsed")
      history.print_avgs(100)
    if step < 2:
      global_stopwatch.clear()
    elif step % 10 == 0:
      global_stopwatch.print_times()
    if step % 100 == 0:
      global_stopwatch.start("write_history_and_checkpoint")
      history.write(config.history_path)
      checkpoint_manager.save(step, train_state)
      global_stopwatch.stop()


if __name__ == "__main__":
  main(experiment_name="aiviator",
       num_steps=1000000,
       num_layers=8,
       batch_size=256,
       train_batch_size=MAX_TURNS * 256 // 32)
