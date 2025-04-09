import collections
from dataclasses import dataclass

import flax
import jax
import jax.numpy as jnp
import jax.random as random

import utils

WIN = jnp.array(1, dtype=jnp.int32)
SAFE = jnp.array(0, dtype=jnp.int32)
COLLISION = jnp.array(-1, dtype=jnp.int32)
OVERSHOT = jnp.array(-2, dtype=jnp.int32)
SPIN = jnp.array(-3, dtype=jnp.int32)
CRASH_LANDING = jnp.array(-4, dtype=jnp.int32)
OUT_OF_FUEL = jnp.array(-5, dtype=jnp.int32)


@flax.struct.dataclass
class Dice:
  values: jnp.ndarray
  counts: jnp.ndarray

  @jax.jit
  def num(self):
    return self.counts.sum()

  @jax.jit
  def values_with_repeats(self):
    indices = jnp.repeat(jnp.arange(len(self.counts)), self.counts, total_repeat_length=4)
    return jnp.take(self.values, indices) * (jnp.arange(4) < self.num())

  def to_list(self):
    return list(sorted([int(d) for d in self.values_with_repeats() if d > 0]))


@jax.jit
def make_dice(rolls):
  values, counts = jnp.unique(rolls, return_counts=True, size=rolls.size, fill_value=0)
  return Dice(values=values, counts=counts * (values > 0))


@jax.jit
def roll_dice(key, num_dice=4):
  return make_dice(random.randint(key, (num_dice,), 1, 7))


@jax.jit
def add_die(dice, die):
  return make_dice(dice.values_with_repeats().at[-1].set(die))


@jax.jit
def remove_die(dice, die):
  return dice.replace(counts=dice.counts - (dice.values == die))


@jax.jit
def reroll_dice(dice, rerolls, key):
  new_counts = dice.counts - rerolls
  rolls = dice.replace(counts=new_counts).values_with_repeats()
  new_rolls = random.randint(key, (4,), 1, 7) * (jnp.arange(4) < dice.num())
  rolls = jnp.where(jnp.arange(4) < new_counts.sum(), rolls, new_rolls)
  return make_dice(rolls)


@flax.struct.dataclass
class Action:
  action_id: jnp.ndarray
  rerolls: jnp.ndarray


@flax.struct.dataclass
class State:
  pilot_engine: jnp.ndarray
  copilot_engine: jnp.ndarray
  pilot_axis: jnp.ndarray
  copilot_axis: jnp.ndarray
  tilt: jnp.ndarray
  min_speed: jnp.ndarray
  max_speed: jnp.ndarray
  brake_speed: jnp.ndarray
  coffees: jnp.ndarray
  rerolls: jnp.ndarray
  fuel: jnp.ndarray
  wind: jnp.ndarray
  altitude: jnp.ndarray
  is_pilot_turn: jnp.ndarray
  is_filled: jnp.ndarray
  is_on: jnp.ndarray
  track_left: jnp.ndarray

  approach_track: jnp.ndarray
  black_dice: jnp.ndarray
  min_tilt: jnp.ndarray
  max_tilt: jnp.ndarray

  history: jnp.ndarray
  result: jnp.ndarray

  key: random.PRNGKey
  pilot_dice: Dice
  copilot_dice: Dice
  mandatory_reroll: jnp.ndarray
  anticipation_reroll: jnp.ndarray

  pilot_swap: jnp.ndarray
  copilot_swap: jnp.ndarray

  fuel_rule: jnp.ndarray
  leak_rule: jnp.ndarray
  wind_rule: jnp.ndarray
  ice_rule: jnp.ndarray

  mastery: jnp.ndarray
  control: jnp.ndarray
  anticipation: jnp.ndarray
  working_together: jnp.ndarray

  @jax.jit
  def current_dice(self):
    return utils.where_pytree(self.is_pilot_turn, self.pilot_dice, self.copilot_dice)

  @jax.jit
  def ally_dice(self):
    return utils.where_pytree(self.is_pilot_turn, self.copilot_dice, self.pilot_dice)

  @jax.jit
  def num_dice(self):
    return self.current_dice().num()

  @jax.jit
  def num_planes(self):
    return (self.approach_track * (self.approach_track > 0)).sum()

  @jax.jit
  def wind_speed(self):
    return jnp.where(
      self.wind_rule,
      jnp.array([3, 3, 2, 2, 1, 0, -1, -2, -2, -3, -3])[jnp.abs(self.wind)],
      jnp.array(0, dtype=jnp.int32))

  @jax.jit
  def must_swap(self):
    return (self.pilot_swap > 0) ^ (self.copilot_swap > 0)

  @jax.jit
  def make_action(self, die, slot, coffees, rerolls):
    dice_values = self.current_dice().values
    die_index = jnp.argmax(dice_values == die)
    coffee_index = jnp.argmax(jnp.array([-2, -1, 0, 1, 2]) == coffees)
    action_id = die_index * (5 * N_SLOTS) + coffee_index * N_SLOTS + slot
    return Action(action_id=action_id, rerolls=rerolls)

  @jax.jit
  def unpack_action(self, action: Action):
    die = self.current_dice().values[action.action_id // (5 * N_SLOTS)]
    coffee = jnp.array([-2, -1, 0, 1, 2])[(action.action_id % (5 * N_SLOTS)) // N_SLOTS]
    slot = action.action_id % N_SLOTS
    return die, slot, coffee


@dataclass
class Slot(object):
  allowed_values: tuple = (1, 2, 3, 4, 5, 6)
  for_pilot: bool = False
  for_copilot: bool = False
  is_switch: bool = False
  has_prerequisite: bool = False


@dataclass
class Engines(Slot):
  def place(self, state: State, die):
    pilot_engine = jnp.where(state.is_pilot_turn, die, state.pilot_engine)
    copilot_engine = jnp.where(~state.is_pilot_turn, die, state.copilot_engine)
    engine_done = (pilot_engine > 0) & (copilot_engine > 0)
    advance_happens = (state.altitude > 0) & engine_done

    speed = pilot_engine + copilot_engine + state.wind_speed()
    steps = jnp.where(speed < state.min_speed, 0,
            jnp.where(speed > state.max_speed, 2, 1))
    advance_result = jnp.where(steps >= state.track_left, OVERSHOT, SAFE)
    advance_result = jnp.where(
      ((steps > 0) & (state.approach_track[0] > 0)) |
      ((steps > 1) & (state.approach_track[1] > 0)) |
      ((steps > 0) & (state.tilt < state.min_tilt[0])) |
      ((steps > 1) & (state.tilt < state.min_tilt[1])) |
      ((steps > 0) & (state.tilt > state.max_tilt[0])) |
      ((steps > 1) & (state.tilt > state.max_tilt[1])),
      COLLISION, advance_result
    )
    result = jnp.where(advance_happens, advance_result, state.result)

    rerolls = state.rerolls
    rerolls = jnp.where((pilot_engine == copilot_engine) & state.mastery,
                        jnp.minimum(rerolls + 1, 3), rerolls)

    fuel = jnp.where(engine_done & state.leak_rule,
                     state.fuel - jnp.abs(pilot_engine - copilot_engine) - 1, state.fuel)
    result = jnp.where(fuel < 0, OUT_OF_FUEL, result)

    def shift_track(track, fill_value):
      beyond_track = jnp.arange(track.shape[0]) >= (track.shape[0] - steps)
      shifted_track = jnp.where(beyond_track, fill_value, jnp.roll(track, -steps))
      return jnp.where(advance_happens, shifted_track, track)

    return state.replace(
      pilot_engine=pilot_engine,
      copilot_engine=copilot_engine,
      rerolls=rerolls,
      fuel=fuel,
      track_left=jnp.where(advance_happens, state.track_left - steps, state.track_left),
      approach_track=shift_track(state.approach_track, -1),
      black_dice=shift_track(state.black_dice, 0),
      min_tilt=shift_track(state.min_tilt, -2),
      max_tilt=shift_track(state.max_tilt, 2),
      result=result,
    )


@dataclass
class Axis(Slot):
  def place(self, state: State, die):
    pilot_axis = jnp.where(state.is_pilot_turn, die, state.pilot_axis)
    copilot_axis = jnp.where(~state.is_pilot_turn, die, state.copilot_axis)
    axis_done = (pilot_axis > 0) & (copilot_axis > 0)
    tilt = jnp.where(axis_done, state.tilt + copilot_axis - pilot_axis, state.tilt)
    coffees = state.coffees
    coffees = jnp.where(axis_done & (pilot_axis == copilot_axis) & state.control,
                        jnp.minimum(coffees + 1, 3), coffees)
    return state.replace(
      pilot_axis=pilot_axis,
      copilot_axis=copilot_axis,
      tilt=tilt,
      wind=jnp.where(axis_done, state.wind + tilt, state.wind),
      coffees=coffees,
      result=jnp.where((jnp.abs(tilt) >= 3) & axis_done, SPIN, state.result),
    )


@dataclass
class Radio(Slot):
  def place(self, state: State, die):
    return state.replace(approach_track=jnp.where(
        die > state.track_left,
        state.approach_track,
        state.approach_track.at[die - 1].set(jnp.maximum(0, state.approach_track[die - 1] - 1))
    ))


@dataclass
class Fuel(Slot):
  for_pilot: bool = True
  for_copilot: bool = True

  def place(self, state: State, die):
    return state.replace(fuel=state.fuel - die,
                         result=jnp.where(state.fuel - die < 0, OUT_OF_FUEL, state.result))


@dataclass
class Coffee(Slot):
  for_pilot: bool = True
  for_copilot: bool = True

  def place(self, state: State, die):
    return state.replace(coffees=jnp.minimum(state.coffees + 1, 3))


@dataclass
class LandingGear(Slot):
  for_pilot: bool = True
  is_switch: bool = True

  def place(self, state: State, die):
    return state.replace(min_speed=state.min_speed + 1)


@dataclass
class Brakes(Slot):
  for_pilot: bool = True
  is_switch: bool = True

  def place(self, state: State, die):
    return state.replace(brake_speed=state.brake_speed + 2)


@dataclass
class IceBrakes(Slot):
  is_switch: bool = True


@dataclass
class Flaps(Slot):
  for_copilot: bool = True
  is_switch: bool = True

  def place(self, state: State, die):
    return state.replace(max_speed=state.max_speed + 1)


@dataclass
class Swap(Slot):
  def place(self, state: State, die):
    pilot_swap = jnp.where(state.is_pilot_turn, die, state.pilot_swap)
    copilot_swap = jnp.where(~state.is_pilot_turn, die, state.copilot_swap)
    do_swap = (pilot_swap > 0) & (copilot_swap > 0)
    return state.replace(
      pilot_swap=pilot_swap,
      copilot_swap=copilot_swap,
      pilot_dice=utils.where_pytree(
        do_swap, add_die(state.pilot_dice, copilot_swap), state.pilot_dice),
      copilot_dice=utils.where_pytree(
        do_swap, add_die(state.copilot_dice, pilot_swap), state.copilot_dice),
    )

SLOTS = [
  Engines(for_pilot=True),
  Axis(for_pilot=True),
  Radio(for_pilot=True),

  LandingGear(allowed_values=(1, 2)),
  LandingGear(allowed_values=(3, 4)),
  LandingGear(allowed_values=(5, 6)),
  Brakes(allowed_values=(2,)),
  Brakes(allowed_values=(4,), has_prerequisite=True),
  Brakes(allowed_values=(6,), has_prerequisite=True),

  IceBrakes(allowed_values=(2,), for_pilot=True),
  IceBrakes(allowed_values=(3,), has_prerequisite=True, for_pilot=True),
  IceBrakes(allowed_values=(4,), has_prerequisite=True, for_pilot=True),
  IceBrakes(allowed_values=(5,), has_prerequisite=True, for_pilot=True),
  IceBrakes(allowed_values=(2,), for_pilot=True, for_copilot=True),
  IceBrakes(allowed_values=(3,), has_prerequisite=True, for_pilot=True, for_copilot=True),
  IceBrakes(allowed_values=(4,), has_prerequisite=True, for_pilot=True, for_copilot=True),
  IceBrakes(allowed_values=(5,), has_prerequisite=True, for_pilot=True, for_copilot=True),

  Engines(for_copilot=True),
  Axis(for_copilot=True),
  Radio(for_copilot=True),
  Radio(for_copilot=True),

  Flaps(allowed_values=(1, 2)),
  Flaps(allowed_values=(2, 3), has_prerequisite=True),
  Flaps(allowed_values=(4, 5), has_prerequisite=True),
  Flaps(allowed_values=(5, 6), has_prerequisite=True),

  Coffee(),
  Coffee(),
  Coffee(),

  Fuel(),

  Swap(for_pilot=True),  # for working together
  Swap(for_copilot=True),
]
N_SLOTS = len(SLOTS)
FUEL_INDEX = 0
for i, s in enumerate(SLOTS):
  if isinstance(s, Fuel):
    FUEL_INDEX = i


@jax.jit
def can_place(state: State, die):
  n_slots = len(SLOTS)
  for_pilot = jnp.array([s.for_pilot for s in SLOTS])
  for_copilot = jnp.array([s.for_copilot for s in SLOTS])
  correct_turn = jnp.where(state.is_pilot_turn, for_pilot, for_copilot)

  allowed_values = jnp.zeros((7, n_slots), dtype=jnp.bool_)
  has_prereq = jnp.zeros((n_slots,), dtype=jnp.bool_)
  is_required = jnp.zeros((n_slots,), dtype=jnp.bool_)
  rules_allow = jnp.ones((n_slots,), dtype=jnp.bool_)
  swap_mask = jnp.repeat(~state.must_swap(), n_slots)
  for i, slot in enumerate(SLOTS):
    for val in slot.allowed_values:
      allowed_values = allowed_values.at[val, i].set(True)
    if slot.has_prerequisite:
      has_prereq = has_prereq.at[i].set(True)
    if isinstance(slot, Engines) or isinstance(slot, Axis):
      is_required = is_required.at[i].set(True)
    elif isinstance(slot, Swap):
      swap_mask = swap_mask.at[i].set(True)
      rules_allow = rules_allow.at[i].set(
        state.working_together & ((state.ally_dice().num() > 0) | state.must_swap()))
    elif isinstance(slot, IceBrakes):
      rules_allow = rules_allow.at[i].set(state.ice_rule)
    elif isinstance(slot, Brakes):
      rules_allow = rules_allow.at[i].set(~state.ice_rule)
    elif isinstance(slot, Fuel):
      rules_allow = rules_allow.at[i].set(state.fuel_rule)

  correct_die = allowed_values[die]
  prereq_filled = jnp.roll(state.is_on, 1) | ~has_prereq
  unfilled_requirements = (correct_turn & is_required & ~state.is_filled).sum()
  required = jnp.where(
    state.num_dice() <= unfilled_requirements, is_required, jnp.ones(n_slots, dtype=jnp.bool_))
  for i, slot in enumerate(SLOTS):
    if isinstance(slot, Swap):
      required = required.at[i].set(True)

  return (correct_turn & ~state.is_filled & ~state.is_on & correct_die & prereq_filled & required &
          swap_mask & rules_allow)


@jax.jit
def add_planes(state: State):
  key, roll_key = random.split(state.key)
  positions = random.randint(roll_key, (3,), 1, 7)
  positions = jnp.minimum(positions, state.track_left)
  positions *= (jnp.arange(3) < state.black_dice[0])
  addition = jnp.bincount(jnp.array(positions), length=state.approach_track.shape[0] + 1)[1:]
  return state.replace(approach_track=state.approach_track + addition, key=key)


@jax.jit
def place_die(state: State, action: Action):
  die, slot, coffees = state.unpack_action(action)

  state=state.replace(
    coffees=state.coffees - jnp.abs(coffees),
    is_filled=state.is_filled.at[slot].set(True),
    history=jnp.roll(state.history, 1).at[0].set(die + coffees),
    pilot_dice=utils.where_pytree(state.is_pilot_turn,
                                  remove_die(state.pilot_dice, die), state.pilot_dice),
    copilot_dice=utils.where_pytree(~state.is_pilot_turn,
                                    remove_die(state.copilot_dice, die), state.copilot_dice),
  )
  die += coffees

  slot_type_indices = collections.defaultdict(list)
  for i, slot_obj in enumerate(SLOTS):
    slot_type_indices[type(slot_obj)].append(i)

  for slot_type, indices in slot_type_indices.items():
    is_active = jnp.isin(slot, jnp.array(indices))
    if slot_type == IceBrakes:
      for i, s in enumerate(SLOTS):
        if isinstance(s, IceBrakes) and s.for_copilot:
          first_copilot_index = i
          break
      companion_index = jnp.where(slot < first_copilot_index, slot + 4, slot - 4)
      state = utils.where_pytree(
        state.is_filled[companion_index] & is_active & state.ice_rule,
        state.replace(brake_speed=jnp.where(state.brake_speed == 0, 2, state.brake_speed + 1),
                      is_on=state.is_on.at[slot].set(True).at[companion_index].set(True)),
        state
      )
    else:
      slot_obj = slot_type()
      state = utils.where_pytree(is_active, slot_obj.place(state, die), state)
      if slot_obj.is_switch:
        state = state.replace(is_on=state.is_on.at[slot].set(state.is_on[slot] | is_active))

  return state.replace(is_pilot_turn=~state.is_pilot_turn)


@jax.jit
def do_reroll(state: State, rerolls: jnp.array):
  key, reroll_key = random.split(state.key)
  rerolled = reroll_dice(state.current_dice(), rerolls, reroll_key)
  return state.replace(
    rerolls=jnp.where(state.mandatory_reroll, state.rerolls, state.rerolls - 1),
    is_pilot_turn=~(state.is_pilot_turn ^ state.anticipation_reroll),
    key=key,
    pilot_dice=utils.where_pytree(state.is_pilot_turn, rerolled, state.pilot_dice),
    copilot_dice=utils.where_pytree(~state.is_pilot_turn, rerolled, state.copilot_dice),
    mandatory_reroll=~state.mandatory_reroll,
    anticipation_reroll=False
  )


@jax.jit
def end_turn(state: State):
  state = add_planes(state)

  landing_gear = jnp.array([1 if isinstance(s, LandingGear) else 0 for s in SLOTS])
  flaps = jnp.array([1 if isinstance(s, Flaps) else 0 for s in SLOTS])

  result = jnp.where(state.altitude == 0,
    jnp.where(
      (state.track_left == 1) &
      (state.approach_track[0] == 0) &
      (state.tilt == 0) &
      ((state.pilot_engine + state.copilot_engine + state.wind_speed()) <= state.brake_speed) &
      (state.brake_speed > 0) &
      (~state.ice_rule | (state.brake_speed == 5)) &
      (state.is_on @ landing_gear == landing_gear.sum()) &
      (state.is_on @ flaps == flaps.sum()),
      WIN, CRASH_LANDING
    ), SAFE)
  fuel = jnp.where(state.is_filled[FUEL_INDEX] | ~state.fuel_rule, state.fuel, state.fuel - 6)
  result = jnp.where(fuel < 0, OUT_OF_FUEL, result)

  key, pilot_key, copilot_key = random.split(state.key, 3)
  return state.replace(
    altitude=state.altitude - 1,
    is_pilot_turn=~state.is_pilot_turn,
    fuel=fuel,
    is_filled=jnp.zeros_like(state.is_filled),
    pilot_engine=jnp.array(0, dtype=jnp.int32),
    copilot_engine=jnp.array(0, dtype=jnp.int32),
    pilot_axis=jnp.array(0, dtype=jnp.int32),
    copilot_axis=jnp.array(0, dtype=jnp.int32),
    pilot_swap=jnp.array(0, dtype=jnp.int32),
    copilot_swap=jnp.array(0, dtype=jnp.int32),
    result=result,
    key=key,
    pilot_dice=roll_dice(pilot_key),
    copilot_dice=roll_dice(copilot_key),
    history=jnp.zeros(8, dtype=jnp.int32),
    mandatory_reroll=state.anticipation,
    anticipation_reroll=state.anticipation,
  )


@jax.jit
def do_action(state: State, action: Action):
  original = state
  state = place_die(state, action)
  turn_over = ((state.result == SAFE) &
               (state.pilot_dice.num() == 0) &
               (state.copilot_dice.num() == 0))
  state = utils.where_pytree(turn_over, end_turn(state), state)
  state = utils.where_pytree(action.rerolls.sum() >= 0, do_reroll(original, action.rerolls), state)
  state = state.replace(result=jnp.where(original.result == SAFE, state.result, original.result))
  return state
