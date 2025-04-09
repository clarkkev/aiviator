import collections
import functools

import jax
import jax.numpy as jnp

import configure
import game


@functools.partial(jax.jit, static_argnums=(0,))
def featurize_board(config: configure.Config, state: game.State):
  return jnp.concat(
    [jax.nn.one_hot(state.pilot_engine, 7),
     jax.nn.one_hot(state.copilot_engine, 7),
     jax.nn.one_hot(state.pilot_axis, 7),
     jax.nn.one_hot(state.copilot_axis, 7),
     jax.nn.one_hot(state.pilot_swap, 7),
     jax.nn.one_hot(state.copilot_swap, 7),
     jax.nn.one_hot(state.num_dice() - 1, 8),
     jax.nn.one_hot(state.altitude, 7),
     jax.nn.one_hot(state.rerolls, 4),
     jax.nn.one_hot(state.coffees, 4),
     jax.nn.one_hot(state.tilt + 2, 5),
     jax.nn.one_hot(state.history, 7).flatten() if config.use_history else jnp.array([0]),
     jnp.array([
       state.tilt,
       state.min_speed,
       state.max_speed,
       state.brake_speed,
       state.coffees,
       state.rerolls,
       state.altitude,
       state.is_pilot_turn,
       state.is_filled.sum(),
       state.altitude % 2 == 0,
       state.altitude == 0,
       state.mandatory_reroll,
       state.anticipation_reroll,
       state.wind_speed(),
       state.fuel / 20.0,
       state.num_dice(),
       state.mastery,
       state.control,
       state.anticipation,
       state.working_together,
       state.fuel_rule,
       state.leak_rule,
       state.wind_rule,
       state.ice_rule]),
     state.is_on,
     state.is_filled,
     state.approach_track,
     state.black_dice,
     state.min_tilt,
     state.max_tilt])


@functools.partial(jax.jit, static_argnums=(0,))
def featurize_die(config: configure.Config, state: game.State, die, place_mask):
  ally_engine = jnp.where(state.is_pilot_turn, state.copilot_engine, state.pilot_engine)
  ally_axis = jnp.where(state.is_pilot_turn, state.copilot_axis, state.pilot_axis)

  speed_result = state.copilot_engine + state.pilot_engine + die + state.wind_speed()
  steps_result = jnp.where(speed_result < state.min_speed, 0,
                           jnp.where(speed_result > state.max_speed, 2, 1))

  tilt_result = state.tilt + state.copilot_axis - state.pilot_axis + jnp.where(
    state.is_pilot_turn, -die, die)
  min_tilt, max_tilt = state.min_tilt[0], state.max_tilt[0]
  lower_bound = jnp.where(state.is_pilot_turn, min_tilt - tilt_result, tilt_result - max_tilt)
  upper_bound = jnp.where(state.is_pilot_turn, max_tilt - tilt_result, tilt_result - min_tilt)
  lower_bound = jnp.maximum(1, lower_bound)
  upper_bound = jnp.minimum(6, upper_bound)
  ally_placements = jnp.maximum(0, upper_bound - lower_bound + 1)

  is_landing = state.altitude == 0
  is_moving = ~is_landing & (ally_engine > 0)

  return jnp.concat([
    jax.nn.one_hot(die - 1, 6),
    is_moving * jax.nn.one_hot(steps_result, 3),
    (ally_axis > 0) * jax.nn.one_hot(jnp.clip(tilt_result + 3, 0, 6), 7),
    place_mask,
    (state.approach_track[die - 1] > 0)[None],
    (state.approach_track[die - 1])[None],
    ((ally_axis > 0) & ((tilt_result > max_tilt) | (tilt_result < min_tilt)))[None],
    ((ally_axis > 0) & ((tilt_result > 2) | (tilt_result < -2)))[None],
    (is_moving & (steps_result == 2) & (state.approach_track[1] != 0))[None],
    (is_landing & (ally_engine > 0) & (speed_result > state.brake_speed))[None],
    ((ally_axis > 0) & ((steps_result > 0) | is_landing) & (state.approach_track[0] != 0))[None],
    jnp.where(ally_axis == 0, ally_placements, -1)[None],
  ])


@functools.partial(jax.jit, static_argnums=(0,))
def featurize(config: configure.Config, state: game.State):
  dice = state.current_dice()
  value_masks = jax.vmap(game.can_place, in_axes=(None, 0))(state, 1 + jnp.arange(6))
  value_features = jax.vmap(featurize_die, in_axes=(None, None, 0, 0))(
    config, state, 1 + jnp.arange(6), value_masks)

  coffees = jnp.array([-2, -1, 0, 1, 2])
  coffee_features = jnp.stack([
    jnp.stack([
      jnp.concatenate([
        jax.nn.one_hot(abs(coffee), 3),
        jnp.array([abs(coffee) <= num_coffees])])
      for coffee in coffees])
    for num_coffees in [0, 1, 2, 3]
  ])

  coffee_dice = dice.values[:, None] + coffees
  features = jnp.concatenate([
    value_features[coffee_dice - 1],
    jnp.tile(jax.nn.one_hot(dice.values - 1, 6)[:, None], [1, 5, 1]),
    jnp.tile(jax.nn.one_hot(dice.counts - 1, 4)[:, None], [1, 5, 1]),
    jnp.tile(coffee_features[state.coffees][None], [4, 1, 1])
  ], axis=-1)

  valid = ((1 <= coffee_dice) & (coffee_dice <= 6) & (dice.counts[:, None] > 0))[..., None]
  features *= valid

  action_mask = value_masks[coffee_dice - 1]
  action_mask *= valid & jnp.broadcast_to((jnp.abs(coffees) <= state.coffees)[:, None],
                                          action_mask.shape)
  features = features.reshape(4 * 5, -1)
  action_mask = action_mask.reshape(-1)
  action_mask *= ~state.mandatory_reroll
  action_mask = jnp.append(action_mask, state.mandatory_reroll | (state.rerolls > 0))

  return Features(
    board_features=featurize_board(config, state),
    dice_features=features,
    ally_dice=jax.nn.one_hot(state.ally_dice().values_with_repeats(), 7),
    action_mask=action_mask,
    reroll_mask=dice.counts[:, None] >= jnp.arange(5),
    anticipation_mask=dice.counts > 0)


Features = collections.namedtuple(
  'Features', ['board_features', 'dice_features', 'ally_dice', 'action_mask', 'reroll_mask',
               'anticipation_mask'])
