import json
import os
import pickle
import shutil

import jax
import jax.numpy as jnp
import jax.tree_util as jtu


def load_pickle(path):
  with open(path, "rb") as f:
    return pickle.load(f)


def write_pickle(o, path):
  if "/" in path:
    mkdir(path.rsplit("/", 1)[0])
  with open(path, "wb") as f:
    pickle.dump(o, f, -1)


def load_json(path):
  with open(path, "r") as f:
    return json.load(f)


def write_json(o, path):
  if "/" in path:
    mkdir(path.rsplit("/", 1)[0])
  with open(path, "w") as f:
    json.dump(o, f, indent=2)


def mkdir(path):
  if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)


def rmrf(path):
  if os.path.exists(path):
    if os.path.isdir(path):
      shutil.rmtree(path)
    else:
      os.remove(path)


def rmkdir(path):
  rmrf(path)
  mkdir(path)


def where_pytree(cond, t1, t2):
  return jax.tree.map(lambda x, y: jnp.where(cond, x, y), t1, t2)


def has_weak_type(tree):
  return any(getattr(leaf, "weak_type", False) for leaf in jax.tree_util.tree_leaves(tree))


def pytrees_match(tree1, tree2, atol=1e-6, rtol=1e-6):
  differences = []

  def compare_fn(path, val1, val2):
    val1, val2 = jnp.asarray(val1), jnp.asarray(val2)
    if not jnp.allclose(val1, val2, atol=atol, rtol=rtol):
        differences.append((path, val1, val2))

  #try:
  jtu.tree_map_with_path(compare_fn, tree1, tree2)
  #except ValueError as e:
  #    print(f"Error: {e}")
  #    return False

  if differences:
    print("Differences found:")
    for path, val1, val2 in differences:
      print(f"Path: {path}, Value 1: {val1}, Value 2: {val2}")
    return False

  return True
