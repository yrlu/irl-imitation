# Experience Replay
# Following paper: Playing Atari with Deep Reinforcement Learning
#     https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
#
# ---
# @author Yiren Lu
# @email luyiren [at] seas [dot] upenn [dot] edu
#
# MIT License


import numpy as np
import random
from collections import namedtuple


Step = namedtuple('Step','cur_step action next_step reward done')


class ExpReplay():
  """Experience replay"""


  def __init__(self, mem_size, start_mem=None, state_size=[84, 84], kth=4, drop_rate=0.2, batch_size=32):
    # k = -1 for sending raw state
    self.state_size = state_size
    self.drop_rate = drop_rate
    self.mem_size = mem_size
    self.start_mem = start_mem
    if start_mem == None:
      self.start_mem = mem_size/20
    self.kth = kth
    self.batch_size = batch_size
    self.mem = []
    self.total_steps = 0


  def add_step(self, step):
    """
    Store episode to memory and check if it reaches the mem_size. 
    If so, drop [self.drop_rate] of the oldest memory

    args
      step      namedtuple Step, where step.cur_step and step.next_step are of size {state_size}
    """
    self.mem.append(step)
    self.total_steps = self.total_steps + 1
    while len(self.mem) > self.mem_size:
      self.mem = self.mem[int(len(self.mem)*self.drop_rate):]


  def get_last_state(self):
    if len(self.mem) > abs(self.kth):
      if self.kth == -1:
        return self.mem[-1].cur_step
      if len(self.state_size) == 1:
        return [s.cur_step for s in self.mem[-abs(self.kth):]]
      last_state = np.stack([s.cur_step for s in self.mem[-abs(self.kth):]], axis=len(self.state_size))
      return np.stack([s.cur_step for s in self.mem[-abs(self.kth):]], axis=len(self.state_size))
    return []


  def sample(self, num=None):
    """Randomly draw [num] samples"""
    if num == None:
      num = self.batch_size
    if len(self.mem) < self.start_mem:
      return []
    sampled_idx = random.sample(range(abs(self.kth),len(self.mem)), num)
    samples = []
    for idx in sampled_idx:
      steps = self.mem[idx-abs(self.kth):idx]
      cur_state = np.stack([s.cur_step for s in steps], axis=len(self.state_size))
      next_state = np.stack([s.next_step for s in steps], axis=len(self.state_size))
      # handle special cases
      if self.kth == -1:
        cur_state = steps[0].cur_step
        next_state = steps[0].next_step
      elif len(self.state_size) == 1:
        cur_state = [steps[0].cur_step]
        next_state = [steps[0].next_step]
      reward = steps[-1].reward
      action = steps[-1].action
      done = steps[-1].done
      samples.append(Step(cur_step=cur_state, action=action, next_step=next_state, reward=reward, done=done))
    return samples




