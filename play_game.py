import collections
import functools
import itertools
import sys
import time

import jax
import jax.numpy as jnp
import jax.random as random

import airports
import configure
import featurization
import game
import learning
import utils


def blue(text):
  return "\033[94m" + str(text) + "\033[0m"


def red(text):
  return "\033[91m" + str(text) + "\033[0m"


def green(text):
  return "\033[92m" + str(text) + "\033[0m"


def orange(text):
  return "\033[38;5;214m" + str(text) + "\033[0m"


def find_common_length(str1, str2, max_len=None):
  common_len = 0
  for a, b in zip(str1, str2):
    if a == b and (max_len is None or common_len < max_len):
      common_len += 1
    else:
      break
  return common_len


def highlight_diff(old, new):
  prefix_len = find_common_length(old, new)
  max_suffix = min(len(old), len(new)) - prefix_len
  suffix_len = find_common_length(old[::-1], new[::-1], max_len=max_suffix)

  result = []
  if prefix_len > 0:
    result.append(new[:prefix_len])

  middle_start = prefix_len
  middle_end = len(new) - suffix_len
  if middle_start < middle_end:
    result.append(green(new[middle_start:middle_end]))

  if suffix_len > 0:
    result.append(new[-suffix_len:])

  result = "".join(result)
  if middle_start >= middle_end and result != old:
    result = green(result)

  return result


def altitude_str(state):
  return "Altitude:", f"{state.altitude}"


def resources_str(state):
  return "Resources:", f"{'☕' * state.coffees}{'🎲'* state.rerolls}"


def markers_str(state):
  return "Speed:", f"{blue(state.min_speed)}-{orange(state.max_speed)}"


def tilt_str(state):
  return "Tilt:", f"{state.tilt}"


def landing_speed_str(state):
  return "Brakes:", f"{red(state.brake_speed)}"


def fuel_str(state):
  return "Fuel:", f"{state.fuel}"


def wind_str(state):
  return "Wind:", f"{state.wind_speed()}"


def ice_brakes_str(state):
  if state.brake_speed == 5:
    return "Ice Brakes:", "Done!"
  for i, s in enumerate(game.SLOTS):
    if isinstance(s, game.IceBrakes) and not state.is_on[i]:
      return ("Ice Brakes:", f"{s.allowed_values[0]}: " +
              f"{blue('✔️' if state.is_filled[i] else '✖️')} " +
              f"{'✔️' if state.is_filled[i + 4] else '✖️'}")


def switch_str(state, slot_type):
  return "".join("🟢" if state.is_on[i] else "🔴" for i, s in enumerate(game.SLOTS)
                 if isinstance(s, slot_type) and (slot_type != game.IceBrakes or s.for_copilot))


def gear_str(state):
  return "Gear:", switch_str(state, game.LandingGear)


def brakes_str(state):
  return "Brakes:", (switch_str(state, game.IceBrakes) if state.ice_rule else
                     switch_str(state, game.Brakes))


def flaps_str(state):
  return "Flaps:", switch_str(state, game.Flaps)


def track_str(track):
  return "[" + " ".join(f"{int(x): d}" for x in track) + " ]"


def approach_track_str(state):
  return "Approach Track:", track_str(state.approach_track[:state.track_left])


def print_state(state, hide_pilot=False, hide_copilot=False):
  print(60 * "=")
  print(*altitude_str(state), end=", ")
  print(*resources_str(state))
  print(*markers_str(state), end=", ")
  print(*landing_speed_str(state), end=", ")
  print(*tilt_str(state), end=", ")
  if state.fuel_rule or state.leak_rule:
    print(*fuel_str(state), end=", " if state.wind_rule else "\n")
  if state.wind_rule:
    print(*wind_str(state))
  print("Engines:", f"{blue(state.pilot_engine)} {orange(state.copilot_engine)}", end=", ")
  print("Axis:", f"{blue(state.pilot_axis)} {orange(state.copilot_axis)}", end="")
  if state.working_together:
    print(", ", end="")
    print("Swap:", f"{blue(state.pilot_swap)} {orange(state.copilot_swap)}", end="")
  if state.ice_rule:
    print(", ", end="")
    print(*ice_brakes_str(state), end="")
  print()
  print(*gear_str(state), end=", ")
  print(*brakes_str(state), end=", ")
  print(*flaps_str(state))
  print(*approach_track_str(state))
  print("Black Die:     ", track_str(state.black_dice[:state.track_left]))
  if state.track_left > 1 and (state.min_tilt[:state.track_left - 1].max() != -2 or
                               state.max_tilt[:state.track_left - 1].min() != 2):
    print("Min Tilt:      ", track_str(state.min_tilt[:state.track_left]))
    print("Max Tilt:      ", track_str(state.max_tilt[:state.track_left]))

  pilot_indicator = ">" if state.is_pilot_turn else ""
  copilot_indicator = ">" if not state.is_pilot_turn else ""
  pilot_dice_display = str(["?" if hide_pilot else d
                            for d in state.pilot_dice.to_list()]).replace("'", "")
  copilot_dice_display = str(["?" if hide_copilot else d
                              for d in state.copilot_dice.to_list()]).replace("'", "")
  print("Die:", f"{blue(pilot_indicator + pilot_dice_display)} " +
               f"{orange(copilot_indicator + copilot_dice_display)}")

  if state.mandatory_reroll:
    print("You must reroll" + (" one die (anticipation)" if state.anticipation_reroll else ""))


def print_diff(state, prev_state):
  if prev_state is not None:
    fns = [altitude_str, resources_str, markers_str, landing_speed_str, tilt_str, fuel_str,
           wind_str, gear_str, brakes_str, ice_brakes_str, flaps_str, approach_track_str]
    for fn in fns:
      prefix, result = fn(state)
      old_result = fn(prev_state)[1]
      diff = highlight_diff(old_result, result)
      if diff != result:
        print("  New", prefix, diff)


def print_ai_actions(state, outputs, action):
  print("AI suggestions:")
  probs = jax.nn.softmax(outputs.logits)
  action_probs, action_ids = jax.lax.top_k(probs, k=20)
  to_print = collections.Counter()
  for p, action_id in zip(action_probs, action_ids):
    a = game.Action(action_id=action_id, rerolls=jnp.array([-1, -1, -1, -1], dtype=jnp.int32))
    if action_id == probs.size - 1:
      to_print[f"Reroll {game.Dice(state.current_dice().values, action.rerolls).to_list()}"] += p
    else:
      die, slot, coffee = state.unpack_action(a)
      to_print[f"{die} {(str(coffee) + ' ') if coffee != 0 else ''}{SLOT_NAMES[slot]}"] += p
  for s, p in to_print.most_common(5):
    if p < 0.01:
      break
    print(f"  {float(100 * p):0.1f}% {s}")


SLOT_PREFIXES = {slot_type.__name__[:2].lower(): slot_type for slot_type in map(type, game.SLOTS)}
SLOT_PREFIXES.update({"ge": game.LandingGear, "ker": game.Fuel, "con": game.Coffee})
SLOT_NAMES = [type(slot).__name__ for slot in game.SLOTS]


def find_slot(slot_name, die, state):
  for prefix, slot_type in SLOT_PREFIXES.items():
    if slot_name.startswith(prefix):
      break
  else:
    raise ValueError(f"Invalid slot name: {slot_name}")
  if state.ice_rule and slot_type == game.Brakes:
    slot_type = game.IceBrakes
  for i, slot in enumerate(game.SLOTS):
    if (isinstance(slot, slot_type) and ~state.is_on[i] and ~state.is_filled[i] and
        die in slot.allowed_values and ((slot.for_pilot and state.is_pilot_turn) or
                                        (slot.for_copilot and ~state.is_pilot_turn))):
      return i
  raise ValueError(f"No {slot_type.__name__} slot is available for die {die}")


def get_human_action(state, history):
  print("\nChoose an action! Examples: 2 gear, 3 -2 eng, reroll 2 3, undo")
  current_dice = state.current_dice()
  while True:
    try:
      sys.stdout.flush()
      user_input = input("> ").strip().lower()
      parts = user_input.split()

      if len(parts) == 0:
        print("Error empty input")
        continue
      if state.result != game.SAFE and not user_input.startswith("undo"):
        print("Error: can only undo after game is over")
        continue

      if user_input.startswith("undo"):
        if len(history) > 0:
          state, _ = history.pop()
          print("Undid last action.")
          return state, None, history
        else:
          print("Error: cannot undo further - at beginning of game.")
          continue
      else:
        assert state.result == game.SAFE, "Can only do undo action if the game is over"
      if user_input.startswith("set"):
        dice = [int(d) for d in parts[1:]]
        assert len(dice) == current_dice.num(), "Can't change number of dice"
        dice = game.make_dice(jnp.array(dice, dtype=jnp.int32))
        if state.is_pilot_turn:
          state = state.replace(pilot_dice=dice)
        else:
          state = state.replace(copilot_dice=dice)
        print("Setting dice to", dice)
        return state, None, history
      elif user_input.startswith("re"):
        dice_to_reroll = [int(d) for d in parts[1:]]
        assert all(d in current_dice.values for d in dice_to_reroll), f"Don't have {dice_to_reroll}"
        dice_to_reroll = collections.Counter(dice_to_reroll)
        rerolls = []
        for d, c in zip(current_dice.values, current_dice.counts):
          assert c >= dice_to_reroll[int(d)], f"Don't have {dice_to_reroll}"
          rerolls.append(dice_to_reroll[int(d)])
        rerolls = jnp.array(rerolls, dtype=jnp.int32)
        action = game.Action(action_id=jnp.array(game.N_SLOTS * 5 * 4, dtype=jnp.int32),
                             rerolls=jnp.array(rerolls, dtype=jnp.int32))
      else:
        die = int(parts[0])
        if len(parts) == 3:
          coffee = int(parts[1])
          slot_name = parts[2]
        else:
          coffee = 0
          slot_name = parts[1]
        die_result = die + coffee

        assert 1 <= die <= 6, f"Die {die} is not between 1 and 6"
        assert 1 <= die_result <= 6, f"Die result {die_result} is not between 1 and 6"
        assert die in current_dice.values, f"Die {die} is not in current dice"

        slot = find_slot(slot_name, die_result, state)
        action = state.make_action(
          jnp.array(die, dtype=jnp.int32), jnp.array(slot, dtype=jnp.int32),
          jnp.array(coffee, dtype=jnp.int32), jnp.array([-1, -1, -1, -1], jnp.int32))

      mask = featurization.featurize(configure.Config(), state).action_mask
      if mask[action.action_id] == 0:
        print(f"Error: d={die} c={coffee} s={SLOT_NAMES[slot]} is invalid!")
        print("Valid actions:")
        for i, mask in enumerate(mask[:-1]):
          if mask != 0:
            die, slot, coffee = map(int, state.unpack_action(
              game.Action(action_id=jnp.array(i, dtype=jnp.int32),
                          rerolls=jnp.array([-1, -1, -1, -1], dtype=jnp.int32))))
            print(f"  d={die} c={coffee} s={SLOT_NAMES[slot]}")
        continue
      return state, action, history
    except (ValueError, IndexError, AssertionError) as e:
      print(f"Error parsing input: {str(e)}")


@functools.cache
def load_ai(path):
  print("Loading model...")
  config = configure.Config(**utils.load_json(path + 'kwargs.json'))
  train_state = learning.make_train_state(config, random.PRNGKey(0))
  return learning.load_checkpoint(train_state, config.checkpoint_dir)


@jax.jit
def run_ai(train_state, state):
  print("Compiling model...")
  outputs = train_state.apply_fn(train_state.params, jax.tree.map(lambda x: x[None], state))
  #action = outputs.sample_action(key)
  action = outputs.get_top_action()
  return jax.tree.map(lambda x: x[0], (outputs, action))


def play_game(seed=0, airport=airports.PBH, abilities=['mastery', 'control'], human_pilot=False,
              human_copilot=False, log_game=False):
  key = random.PRNGKey(seed)
  state = airport(key, **{a: jnp.array(True) for a in abilities})
  history = []
  special_action = False
  ai_only = (not human_pilot) and (not human_copilot)

  if not (human_pilot and human_copilot):
    train_state = load_ai('./data/aiviator/')

  while True:
    if log_game:
      if state.result != game.SAFE:
        print("The game is over, but you can \"undo\" to continue or Ctrl+D to exit.")
      else:
        print_state(state, not ai_only and not human_pilot, not ai_only and not human_copilot)

    human_turn = ((human_pilot and state.is_pilot_turn) or (human_copilot and ~state.is_pilot_turn)
                  or (state.result != game.SAFE))
    if human_turn or special_action:
      state, action, history = get_human_action(state, history)
      if action is None:
        special_action = True
        continue
      special_action = False
    else:
      special_action = False
      outputs, action = run_ai(train_state, state)
      if log_game and ai_only:
          print()
          print_ai_actions(state, outputs, action)

    if log_game:
      print()
      if action.rerolls.sum() < 0:
        die, slot, coffee = state.unpack_action(action)
        print(f"Placing {die} " +
              (f"(using {coffee} to make {die + coffee}) " if coffee != 0 else "") +
              f"on {SLOT_NAMES[slot]}")
      else:
        reroll_list = game.Dice(state.current_dice().values, action.rerolls).to_list()
        print(f"Rerolling {reroll_list}")

    history.append((state, action))
    state = game.do_action(state, action)

    if state.result != game.SAFE:
      if log_game:
        print(60 * "-")
        print("Congratulations!" if state.result == game.WIN else "Game over!")
        print({
              int(game.WIN): "You landed safely!",
              int(game.COLLISION): "You crashed while advancing!",
              int(game.OVERSHOT): "You overshot the airport!",
              int(game.SPIN): "You entered a spin!",
              int(game.CRASH_LANDING): "You crashed while landing!",
              int(game.OUT_OF_FUEL): "You ran out of fuel!",
            }[int(state.result)])
        print(60 * "-")
        print()
      if human_pilot or human_copilot:
        continue
      else:
        history.append((state, None))
        return state.result, history

    if log_game:
      prev_state = history[-1][0]
      if action.rerolls.sum() >= 0 and (human_turn or ai_only):
        print("  New Dice:", green((
          state.current_dice() if prev_state.anticipation_reroll else state.ally_dice()).to_list()))
      elif (isinstance(game.SLOTS[slot], game.Swap) and state.pilot_swap > 0 and
            state.copilot_swap > 0):
        print(f"  Swapping {blue(state.pilot_swap)} and {orange(state.copilot_swap)}")
      if state.track_left != prev_state.track_left:
        steps = str(prev_state.track_left - state.track_left)
        print("  Moving " + steps + " space" + ("s" if steps != "1" else ""))
      if state.altitude != prev_state.altitude:
        print(60 * "-")
        print("  End of turn!")
        print(60 * "-")
      print_diff(state, prev_state)


def main():
  seed = int(time.time())
  while True:
    print("What would you like to do?")
    print("[1] Play as the pilot with the AI playing the copilot")
    print("[2] Play as the copilot with the AI playing the pilot")
    print("[3] Watch the AI play")
    game_mode = input("> ")
    assert game_mode in ["1", "2", "3"], "Invalid game mode"

    print("Which airport would you like to play on?")

    airports_dict = {
      "PBH": airports.PBH,
      "KEF": airports.KEF,
      "KUL": airports.KUL,
      "PBH_RED": airports.PBH_RED,
      "GIG": airports.GIG,
      "TGU": airports.TGU,
      "OSL": airports.OSL,
    }
    airport_names = list(airports_dict.keys())
    for i, airport_name in enumerate(airport_names):
      print(f"[{i + 1}] {airport_name}")
    airport_choice = input("> ")
    airport = airports_dict[airport_names[int(airport_choice) - 1]]
    assert int(airport_choice) in range(1, len(airport_names) + 1), "Invalid airport choice"

    print("Which abilities would you like to play with?")
    ability_options = list(itertools.combinations(
      ['working_together', 'mastery', 'control', 'anticipation'], 2))
    for i, abilities in enumerate(ability_options):
      print(f"[{i + 1}] {', '.join(abilities)}")
    abilities_choice = input("> ")
    abilities = ability_options[int(abilities_choice) - 1]
    assert int(abilities_choice) in range(1, len(ability_options) + 1), "Invalid abilities choice"

    play_game(airport=airport, abilities=abilities, seed=seed, log_game=True,
              human_pilot=game_mode == "1", human_copilot=game_mode == "2")


if __name__ == "__main__":
  main()
