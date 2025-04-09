import itertools

import jax
import jax.numpy as jnp
import jax.random as random

import airports
import game
import learning
import play_game


def main():
  train_state = play_game.load_ai('./data/aiviator/')

  batch_size = 256
  for airport, airport_name in [
      (airports.KEF, 'KEF'), (airports.KUL, 'KUL'), (airports.PBH, 'PBH'),
      (airports.PBH_RED, 'PBH_RED'), (airports.GIG, 'GIG'), (airports.TGU, 'TGU'),
      (airports.OSL, 'OSL')]:
    print(airport_name)
    make_state = jax.jit(jax.vmap(airport))
    for selected_abilities in itertools.combinations(
        ['working_together', 'mastery', 'anticipation', 'control'], 2):
      abilities = {
        key: jnp.array([key in selected_abilities] * batch_size)
        for key in ['working_together', 'mastery', 'anticipation', 'control']
      }
      wins, losses = 0, 0
      # start = time.time()
      for seed in range(0, 1024 // batch_size):
        key = random.PRNGKey(seed)

        key, init_key = random.split(key)

        states = make_state(key=random.split(init_key, batch_size), **abilities)
        scores = jnp.zeros(batch_size)

        history = []
        for _ in range(learning.MAX_TURNS):
          # key, step_key = random.split(key)
          states, scores, result = learning.trajectory_step(train_state, states, scores, None)
          history.append(result)

        for r in history[-1][0].result:
          if r == game.WIN:
            wins += 1
          else:
            losses += 1

      print(f" {selected_abilities}: {wins}/{wins + losses} = {100 * wins / (wins + losses):0.1f}%")
      #print(time.time() - start)


if __name__ == '__main__':
  main()
