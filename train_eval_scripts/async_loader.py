
  
import os, sys
from threading import Thread
from queue import Queue

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader



class AsynchronousLoader(object):
  """
  Class for asynchronously loading from CPU memory to device memory
  Parameters
  ----------
  dataset: PyTorch Dataset
      The PyTorch dataset we're loading
  device: PyTorch Device
      The PyTorch device we are loading to
  batch_size: Integer
      The batch size to load in
  pin_memory: Boolean
      Whether to use CUDA pinned memory
      Note that this should *always* be set to True for asynchronous loading to CUDA devices
  workers: Integer
      Number of worker processes to use for loading from storage and collating the batches in CPU memory
  queue_size: Integer
      Size of the que used to store the data loaded to the device
  """
  def __init__(self, dataset, device, batch_size = 1, pin_memory = True, shuffle = False, workers = 10, queue_size = 10, **kwargs):
    self.dataset = dataset
    self.device = device
    self.batch_size = batch_size
    self.workers = workers
    self.pin_memory = pin_memory
    self.shuffle = shuffle
    self.queue_size = queue_size

    # Use PyTorch's DataLoader for collating samples and stuff since it's nicely written and parallelrised
    self.dataloader = DataLoader(dataset, batch_size = batch_size, pin_memory = pin_memory, shuffle = shuffle, num_workers = workers, **kwargs)

    self.load_stream = torch.cuda.Stream(device = device)
    self.queue = Queue(maxsize = self.queue_size)

    self.idx = 0

  def load_loop(self): # The loop that will load into the queue in the background
    for i, sample in enumerate(self.dataloader):
      self.queue.put(self.load_instance(sample))

  def load_instance(self, sample): # Recursive loading for each instance based on torch.utils.data.default_collate
    if torch.is_tensor(sample):
      with torch.cuda.stream(self.load_stream):
        return sample.to(self.device, non_blocking = True)
    else:
      out = []
      for s in sample:
        if isinstance(s, tuple):
          out.append([self.load_instance(x) for x in s])
        else:
          out.append(self.load_instance(s))
      return out

  def __iter__(self):
    assert self.idx == 0, 'idx must be 0 at the beginning of __iter__. Are you trying to run the same instance more than once in parallel?'
    self.idx = 0
    self.worker = Thread(target = self.load_loop)
    self.worker.start()
    return self


  def __next__(self):
    # If we've reached the number of batches to return or the queue is empty and the worker is dead then exit
    if (not self.worker.is_alive() and self.queue.empty()) or self.idx >= len(self.dataloader):
      self.idx = 0
      self.queue.join()
      self.worker.join()
      raise StopIteration
    else: # Otherwise return the next batch
      out = self.queue.get()
      self.queue.task_done()
      self.idx += 1
    return out

  def __len__(self):
    return len(self.dataloader)


