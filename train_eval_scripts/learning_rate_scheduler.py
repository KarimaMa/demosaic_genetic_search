from collections import deque
import numpy as np
import torch

span = 5

class Scheduler():
  def __init__(self, optimizer, max_var, min_var):
    self.optimizer = optimizer
    self.max_var = max_var
    self.min_var = min_var
    self.validation_tracker = deque(maxlen=span)
    self.ticks = 0

  def update_validation_tracker(self, validation_psnr):
      self.validation_tracker.append(validation_psnr)

  def step(self):
    self.ticks += 1
    if len(self.validation_tracker) < span:
      return    

    validation_var = np.var(list(self.validation_tracker))
    print(f"validation var {validation_var}")
    if validation_var > self.min_var and validation_var < self.max_var:
      return 
    elif self.ticks < span:
      return 

    if validation_var < self.min_var:
      factor = 1.33
    else:
      factor = 1.0/1.33

    for param_group in self.optimizer.param_groups:
      old_lr = param_group['lr']
      param_group['lr'] *= factor
      new_lr = param_group['lr']
      print(f"changing learning rate: old lr {old_lr} new lr {new_lr}")
 
    self.ticks = 0 # wait another 10 steps before changing learning rate again


class LRScheduler():
  def __init__(self, args, model, max_var, min_var):
    self.lr = args.learning_rate
    self.model = model
    self.args = args
    if args.adam:
      self.optimizer = torch.optim.Adam(model.parameters(),
          lr=args.learning_rate,
          weight_decay=args.weight_decay)
    elif args.sgd:
      self.optimizer = torch.optim.SGD(model.parameters(),
          args.learning_rate,
          momentum=0.9,
          weight_decay=args.weight_decay)
    self.max_var = max_var
    self.min_var = min_var
    self.validation_tracker = deque(maxlen=span) 
    self.ticks = 0

  def update_validation_tracker(self, validation_psnr):
      self.validation_tracker.append(validation_psnr)

  def step(self, logger):
    self.ticks += 1
    if len(self.validation_tracker) < span:
      return 

    validation_var = np.var(list(self.validation_tracker))
    logger.info(f"validation var {validation_var}")
    
    if validation_var > self.min_var and validation_var < self.max_var:
      return 
    elif self.ticks < span:
      return 

    if validation_var < self.min_var:
      factor = 1.33
    else:
      factor = 1.0/1.33

    old_lr = self.lr
    self.lr *= factor
    logger.info(f"changing learning rate from {old_lr} to {self.lr}")
    # create new optimizer
    if self.args.adam:
      self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.args.weight_decay)
    elif self.args.sgd:
      self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.args.weight_decay)
      
    self.ticks = 0 # wait another span steps before changing learning rate again
    return 




