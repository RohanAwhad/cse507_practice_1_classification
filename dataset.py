# dataset.py
import glob
import multiprocessing as mp
import os
import pickle
import random
import threading
import time
import torch

from abc import ABC, abstractmethod
from collections import deque
from typing import Optional

class Dataset(ABC):
  @abstractmethod
  def next_batch(self):
    pass

  @abstractmethod
  def reset(self):
    pass

class CheXpertDatasetLoaderLite(Dataset):
  def __init__(self, root: str, split: str, batch_size: int, process_rank: int, world_size: int, use_worker: bool = False, prefetch_size: int = 1):
    self.root = root
    self.batch_size = batch_size
    self.split = split
    self.process_rank = process_rank
    self.world_size = world_size

    self.files = glob.glob(os.path.join(root, f'{split}_ds', f'shard_*.pkl'))
    self.shard_size = self._get_shard_size()
    self.curr_file_ptr = None
    
    # multiprocessing
    self.use_worker = use_worker
    self.prefetch_size = prefetch_size
    self.workers: list[mp.Process] = []
    self.prefetch_thread = None

    self.offset = self.batch_size * self.process_rank
    self.step = self.batch_size * self.world_size
    assert self.step <= self.shard_size, "Batch size * world size must be less than or equal to shard size"

    self.reset()

  def next_batch(self):
    if self.use_worker: return self._next_batch_from_queue()
    return self._next_batch()

  def _get_shard_size(self): 
    with open(self.files[0], 'rb') as f:
      return len(pickle.load(f))

  def load_shard(self, file_ptr):
    with open(self.files[file_ptr], 'rb') as f:
      return pickle.load(f)

  def _next_batch(self):
    batch = self.curr_shard[self.curr_idx:self.curr_idx+self.step]
    self.curr_idx += self.step
    # drop last batch if it's smaller than batch_size
    if (self.curr_idx + self.step) >= len(self.curr_shard):
      self.curr_file_ptr = (self.curr_file_ptr + 1) % len(self.files)  # cycle through files if necessary
      self.curr_idx = self.offset
      self.curr_shard = self.load_shard(self.curr_file_ptr)
    images, labels = zip(*batch)
    labels = torch.tensor(labels)
    label_masks = (labels != -1).long()
    return {'image': torch.stack(images), 'label': labels.long(), 'label_mask': label_masks}

  def reset(self):
    if not self.use_worker:
      if self.curr_file_ptr is None or self.curr_file_ptr != 0:  # loading shard is costly, and this op is common during overfitting or hyperparameter tuning
        self.curr_file_ptr = 0
        self.curr_shard = self.load_shard(self.curr_file_ptr)
      self.curr_idx = self.offset

    else:
      # for efficiency reasons, we build a queue to prefetch batches
      self.prefetch_queue = mp.Queue(maxsize=self.prefetch_size)
      if len(self.workers) > 0:
        for worker in self.workers:
          worker.terminate()
          worker.close()
      self.workers = []
      worker = mp.Process(target=self.worker)
      worker.start()
      self.workers.append(worker)


  def worker(self):
    if self.curr_file_ptr is None or self.curr_file_ptr != 0:  # loading shard is costly, and this op is common during overfitting or hyperparameter tuning
      self.curr_file_ptr = 0
      self.curr_shard = self.load_shard(self.curr_file_ptr)
    self.curr_idx = self.offset
    self._fill_queue()

  def _fill_queue(self):
    while True:
      # add a batch to the queue
      # multiprocessing
      if self.prefetch_queue.full(): time.sleep(0.25)
      else: self.prefetch_queue.put(self._next_batch())

  def _next_batch_from_queue(self):
    # multiprocessing
    while self.prefetch_queue.empty():
      time.sleep(0.25)
    return self.prefetch_queue.get()
