import jax
import jax.numpy as jnp
import jax.random as random

import game


def make_airport(initial_track, **initial_state):
  def make_state(key=random.PRNGKey(0), n_planes=jnp.array(-1, dtype=jnp.int32), **updates):
    key, pilot_key, copilot_key, track_key, rules_key = random.split(key, 5)

    initial_planes = (initial_track * (initial_track > 0)).sum()
    n_planes = jnp.where(n_planes == -1, initial_planes, n_planes)
    max_planes = 10
    indices = jnp.repeat(jnp.arange(initial_track.size), initial_track,
                        total_repeat_length=max_planes)
    samples = jax.random.choice(track_key, indices, shape=(max_planes,),
                                p=(jnp.arange(max_planes) < initial_planes), replace=False)
    samples = (samples + 1) * (jnp.arange(max_planes) < n_planes)
    approach_track = jnp.bincount(samples, length=initial_track.size + 1)[1:]

    if all(k not in updates for k in ["mastery", "control", "anticipation", "working_together"]):
      rules = jax.random.permutation(rules_key, jnp.array([True, True, False, False]))
    else:
      rules = jnp.array([False, False, False, False])

    state = game.State(
      pilot_engine=jnp.array(0, dtype=jnp.int32),
      copilot_engine=jnp.array(0, dtype=jnp.int32),
      pilot_axis=jnp.array(0, dtype=jnp.int32),
      copilot_axis=jnp.array(0, dtype=jnp.int32),
      tilt=jnp.array(0, dtype=jnp.int32),
      min_speed=jnp.array(5, dtype=jnp.int32),
      max_speed=jnp.array(8, dtype=jnp.int32),
      brake_speed=jnp.array(0, dtype=jnp.int32),
      coffees=jnp.array(0, dtype=jnp.int32),
      rerolls=jnp.array(1, dtype=jnp.int32),
      fuel=jnp.array(20, dtype=jnp.int32),
      wind=jnp.array(0, dtype=jnp.int32),
      altitude=jnp.array(6, dtype=jnp.int32),
      is_pilot_turn=jnp.array(True, dtype=jnp.bool_),
      track_left=jnp.array(approach_track.size, dtype=jnp.int32),
      is_filled=jnp.zeros(game.N_SLOTS, dtype=jnp.bool_),
      is_on=jnp.zeros(game.N_SLOTS, dtype=jnp.bool_),
      approach_track=approach_track,
      black_dice=jnp.array([0] * len(approach_track)),
      min_tilt=jnp.array([-2] * (len(approach_track) - 1) + [0]),
      max_tilt=jnp.array([2] * (len(approach_track) - 1) + [0]),
      pilot_dice=game.roll_dice(pilot_key),
      copilot_dice=game.roll_dice(copilot_key),
      key=key,
      result=game.SAFE,
      mandatory_reroll=False,
      anticipation_reroll=False,
      pilot_swap=jnp.array(0, dtype=jnp.int32),
      copilot_swap=jnp.array(0, dtype=jnp.int32),
      history=jnp.zeros(8, dtype=jnp.int32),
      fuel_rule=False,
      leak_rule=False,
      wind_rule=False,
      ice_rule=False,
      mastery=rules[0],
      control=rules[1],
      anticipation=rules[2],
      working_together=rules[3],
    ).replace(**initial_state).replace(**updates)

    def pad_track(track, v):
      return jnp.concat([track, jnp.array([v] * (8 - len(initial_track)))])
    state = state.replace(mandatory_reroll=state.anticipation,
                          anticipation_reroll=state.anticipation,
                          approach_track=pad_track(state.approach_track, -1),
                          min_tilt=pad_track(state.min_tilt, -2),
                          max_tilt=pad_track(state.max_tilt, 2),
                          black_dice=pad_track(state.black_dice, 0))

    return game.add_planes(state)
  return jax.jit(make_state)


YUL = make_airport(jnp.array([0, 0, 1, 2, 1, 3, 2]))
KEF = make_airport(
  initial_track=jnp.array([0, 0, 2, 1, 1, 0]),
  black_dice=jnp.array([2, 0, 1, 0, 0, 0]),
  min_tilt=jnp.array([-2, -2, -2, 0, -1, 0]),
  max_tilt=jnp.array([2, 0, 2, 2, 1, 0]),
  wind_rule=True,
  ice_rule=True,
)
KUL = make_airport(
  initial_track=jnp.array([0, 0, 1, 0, 1, 1, 1, 1]),
  black_dice=jnp.array([3, 0, 1, 0, 0, 1, 0, 0]),
  min_tilt=jnp.array([-2, 0, -2, -2, -1, 0, 1, 0]),
  max_tilt=jnp.array([2, 1, 2, -2, 0, 1, 2, 0]),
  fuel_rule=True,
)
PBH = make_airport(
  initial_track=jnp.array([1, 0, 1, 1, 1, 1]),
  black_dice=jnp.array([3, 0, 1, 1, 1, 0]),
  min_tilt=jnp.array([-2, -2, -2, -2, 2, 0]),
  max_tilt=jnp.array([2, 2, -1, -1, 2, 0]),
  fuel_rule=True,
)
PBH_RED = make_airport(
  initial_track=jnp.array([0, 1, 1, 1, 1, 1]),
  black_dice=jnp.array([3, 0, 0, 1, 0, 0]),
  min_tilt=jnp.array([-2, -2, -2, -2, 1, 0]),
  max_tilt=jnp.array([2, 2, -1, 0, 2, 0]),
  fuel_rule=True,
)
GIG = make_airport(
  initial_track=jnp.array([0, 1, 2, 2, 1, 1, 2]),
  black_dice=jnp.array([3, 0, 1, 0, 1, 0, 0]),
  leak_rule=True,
  wind_rule=True,
)
TGU = make_airport(
  initial_track=jnp.array([0, 1, 1, 1, 2]),
  black_dice=jnp.array([3, 0, 2, 0, 0]),
  min_tilt=jnp.array([-2, -2, -1, -2, 0]),
  max_tilt=jnp.array([2, -1, 0, -1, 0]),
  fuel_rule=True,
  wind_rule=True,
)
OSL = make_airport(
  initial_track=jnp.array([1, 0, 1, 0, 0, 1, 0, 0]),
  black_dice=jnp.array([3, 0, 1, 0, 1, 0, 1, 0]),
  leak_rule=True,
  ice_rule=True,
)


@jax.jit
def random_state(key=random.PRNGKey(0)):
  airport_key, choice_key = random.split(key)
  states = [airport(airport_key) for airport in [KEF, KUL, PBH, PBH_RED, GIG, TGU, OSL]]
  i = jax.random.choice(choice_key, len(states), p=jnp.array([0.2, 0.1, 0.2, 0.1, 0.1, 0.1, 0.2]))
  return jax.tree.map(lambda *args: jnp.stack(args)[i], *states)
