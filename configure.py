import glob
import json
import os
import shutil

import ml_collections

import utils


class Config(ml_collections.ConfigDict):
  def __init__(self, write=False, overwrite=False, **kwargs):
    super().__init__()

    self.experiment_name = 'debug'

    self.num_layers = 4
    self.num_heads = 4
    self.hidden_dim = 256
    self.mlp_dim = 1024

    self.value_weight = 10.0
    self.entropy_weight = 1e-4
    self.learning_rate = 5e-5
    self.gae_lambda = 0.9
    self.discount_factor = 0.99

    self.ppo = True
    self.ppo_clip_ratio = 0.2
    self.use_history = True
    self.use_ally_dice_for_value = True
    self.curriculum_threshold = 60

    self.batch_size = 128
    self.train_batch_size = 256
    self.num_steps = 10000
    self.metrics_every = 5

    self.init_checkpoint = None

    self.update(**kwargs)

    self.code_location = '.'
    self.experiment_dir = f'./data/{self.experiment_name}/'
    self.code_dir = self.experiment_dir + 'code/'
    self.config_path = self.experiment_dir + 'config.json'
    self.kwargs_path = self.experiment_dir + 'kwargs.json'
    self.history_path = self.experiment_dir + 'history.pkl'
    self.train_history_path = self.experiment_dir + 'train_history.pkl'
    self.checkpoint_dir = os.path.abspath(self.experiment_dir + 'checkpoint')

    if write:
      if os.path.exists(self.code_dir) and not overwrite:
        response = input(f"Directory {self.code_dir} exists. Overwrite? [Y/n]: ").strip().lower()
        if response not in ['', 'y', 'yes']:
          raise Exception(f"Aborted: {self.code_dir} exists")

      utils.mkdir(self.code_dir)
      py_files = glob.glob(os.path.join(self.code_location, "*.py"))
      ipynb_files = glob.glob(os.path.join(self.code_location, "*.ipynb"))
      for file_path in py_files + ipynb_files:
          shutil.copy2(file_path, os.path.join(self.code_dir, os.path.basename(file_path)))
      utils.write_json(json.loads(self.to_json_best_effort()), self.config_path)
      utils.write_json(kwargs, self.kwargs_path)


  def __hash__(self):
      return hash(id(self))