from collections import deque
import numpy as np

class VarianceTracker():
  def __init__(self, model_inits):
    self.trackers = [[] for v in range(model_inits)]

  def update(self, validation_psnrs):
    for i, psnr in enumerate(validation_psnrs):
      self.trackers[i].append(psnr)

  def validation_variance(self):
    validation_var = np.mean([np.var(t) for t in self.trackers])
    return validation_var

