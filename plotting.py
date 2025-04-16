# contrastive/utils.py

from acme.wrappers import base
import numpy as np

class TrajectoryLoggerWrapper(base.EnvironmentWrapper):
    """Logs (step, x, y, reward) to a CSV on every env.reset() and env.step()."""

    def __init__(self, env, log_path="trajectory.csv"):
        super().__init__(env)
        self.log_path = log_path
        self._file = None
        self._step_cnt = 0

    def _open(self):
        if self._file:
            self._file.close()
        self._file = open(self.log_path, "w")
        self._file.write("step,x,y,reward\n")
        self._file.flush()
        self._step_cnt = 0

    def reset(self):
        ts = self._environment.reset()
        if self._file is None:
            self._open()

        x, y = ts.observation[:2]
        # coalesce None → 0.0
        raw_r = ts.reward
        r = float(raw_r) if raw_r is not None else 0.0

        self._file.write(f"{self._step_cnt},{x:.6f},{y:.6f},{r:.6f}\n")
        self._file.flush()
        self._step_cnt = 1
        return ts

    def step(self, action):
        ts = self._environment.step(action)
        x, y = ts.observation[:2]
        raw_r = ts.reward
        r = float(raw_r) if raw_r is not None else 0.0

        self._file.write(f"{self._step_cnt},{x:.6f},{y:.6f},{r:.6f}\n")
        self._file.flush()
        self._step_cnt += 1
        return ts

