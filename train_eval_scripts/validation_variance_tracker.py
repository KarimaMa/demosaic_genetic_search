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

class LRTracker():
  def __init__(self, initial_lr, variance_max, variance_min):
    self.proposed_lrs = [initial_lr]
    self.seen_variances = []
    self.variance_max = variance_max
    self.variance_min = variance_min

  def update_lr(self, validation_var):
    self.seen_variances += [validation_var]
    current_lr = self.proposed_lrs[-1]
    current_var = self.seen_variances[-1]
    if len(self.proposed_lrs) > 1:
      prev_lr = self.proposed_lrs[-2]
      prev_var = self.seen_variances[-2]
    else:
      prev_lr = None
    if validation_var > self.variance_max:
      if prev_lr:
        if prev_lr < current_lr: # overshot lr, step back halfway
          new_lr = (current_lr - prev_lr)/2 + prev_lr
        else:
          # validation variance is still above the variance max after decreasing learning rate
          if current_var > prev_var: # decreasing lr didn't help - variance actually went up, stick to current lr
            new_lr = current_lr
          else: # variance decreased after decreasing lr but still not below max threshold
            variance_decrease_ratio = (prev_var-current_var)/(prev_var)
            lr_decrease_ratio = (prev_lr-current_lr)/(prev_lr)
            if (variance_decrease_ratio / lr_decrease_ratio) < 0.5: # diminishing returns from decreasing lr - little impact on variance
              new_lr = prev_lr
            else:
              new_lr = current_lr / 2
      else: # this is the first search step, halve the learning rate
        new_lr = current_lr / 2
    elif validation_var < self.variance_min:
      if prev_lr:
        if prev_lr > current_lr: # undershot lr, step up halfway
          new_lr = prev_lr - (prev_lr - current_lr)/2
        else:
          # validation variance is still below the variance min even after increasing learning rate
          if current_var < prev_var: # increasing lr didn't help - variance actually went down, stick to current lr
            new_lr = current_lr
          else: # variance increased after increasing lr but still not above min threshold
            new_lr = current_lr * 2
      else: # this is the first search step, double the learning rate
        new_lr = current_lr * 2
    else: # current learning rate is acceptable
      new_lr = current_lr

    self.proposed_lrs += [new_lr]
    return new_lr
