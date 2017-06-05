# 1D Gridworld
#
# ---
# @author Yiren Lu
# @email luyiren [at] seas [dot] upenn [dot] edu
#
# MIT License

import numpy as np
from utils import *

class GridWorld1D(object):
  """
  1D grid world environment (without terminal states)
  """

  def __init__(self, rewards, terminals, move_rand=0.0):
    """
    inputs:
      rewards     1d float array - contains rewards
      terminals   a set of all the terminal states
    """
    self.n_states = len(rewards)
    self.rewards = rewards
    self.terminals = terminals
    self.actions = [-1, 1]
    self.n_actions = len(self.actions)
    self.move_rand = move_rand


  def get_reward(self, state):
    return self.rewards[state]


  def get_transition_states_and_probs(self, state, action):
    """
    inputs: 
      state       int - state
      action      int - action

    returns
      a list of (state, probability) pair
    """
    if action < 0 or action >= self.n_actions:
      # invalid input
      return []

    if self.is_terminal(state):
      return [(state, 1.0)]

    if self.move_rand == 0:
      if state+self.actions[action] < 0 or state+self.actions[action] >= self.n_states:
        return [(state, 1.0)]
      return [(state+self.actions[action], 1.0)]
    else:
      mov_probs = np.zeros(3)
      mov_probs[1+self.actions[action]] += 1 - self.move_rand 
      for i in range(3):
        mov_probs[i] += self.move_rand/3

      if state == 0:
        mov_probs[1] += mov_probs[0]
        mov_probs[0] = 0
      if state == self.n_states - 1:
        mov_probs[1] += mov_probs[2]
        mov_probs[2] = 0

      res = []
      for i in range(3):
        if mov_probs[i] != 0:
          res.append((state-1+i, mov_probs[i]))
      return res


  def is_terminal(self, state):
    if state in self.terminals:
      return True
    else:
      return False

  ##############################################
  # Stateful Functions For Model-Free Leanring #
  ##############################################

  def reset(self, start_pos):
    self._cur_state = start_pos

  def get_current_state(self):
    return self._cur_state

  def step(self, action):
    """
    Step function for the agent to interact with gridworld
    inputs: 
      action        action taken by the agent
    returns
      current_state current state
      action        input action
      next_state    next_state
      reward        reward on the next state
      is_done       True/False - if the agent is already on the terminal states
    """
    if self.is_terminal(self._cur_state):
      self._is_done = True
      return self._cur_state, action, self._cur_state, self.get_reward(self._cur_state), True

    st_prob = self.get_transition_states_and_probs(self._cur_state, action)

    rand_idx = np.random.choice(np.arange(0, len(st_prob)), p=[prob for st, prob in st_prob])
    last_state = self._cur_state
    next_state = st_prob[rand_idx][0]
    reward = self.get_reward(last_state)
    self._cur_state = next_state
    return last_state, action, next_state, reward, False

  #######################
  # Some util functions #
  #######################

  def get_transition_mat(self):
    """
    get transition dynamics of the gridworld

    return:
      P_a         NxNxN_ACTIONS transition probabilities matrix - 
                    P_a[s0, s1, a] is the transition prob of 
                    landing at state s1 when taking action 
                    a at state s0
    """
    N_STATES = self.n_states
    N_ACTIONS = len(self.actions)
    P_a = np.zeros((N_STATES, N_STATES, N_ACTIONS))
    for si in range(N_STATES):
      for a in range(N_ACTIONS):
        probs = self.get_transition_states_and_probs(si, a)

        for sj, prob in probs:
          # Prob of si to sj given action a
          P_a[si, sj, a] = prob
    return P_a


  def generate_demonstrations(self, policy, n_trajs=100, len_traj=20, rand_start=False, start_pos=0):
    """gatheres expert demonstrations

    inputs:
    gw          Gridworld - the environment
    policy      Nx1 matrix
    n_trajs     int - number of trajectories to generate
    rand_start  bool - randomly picking start position or not
    start_pos   2x1 list - set start position, default [0,0]
    returns:
    trajs       a list of trajectories - each element in the list is a list of Steps representing an episode
    """

    trajs = []
    for i in range(n_trajs):
      if rand_start:
        # override start_pos
        start_pos = np.random.randint(0, self.n_states)

      episode = []
      self.reset(start_pos)
      cur_state = start_pos
      cur_state, action, next_state, reward, is_done = self.step(int(policy[cur_state]))
      episode.append(Step(cur_state=cur_state, action=self.actions[action], next_state=next_state, reward=reward, done=is_done))
      # while not is_done:
      for _ in range(1,len_traj):
          cur_state, action, next_state, reward, is_done = self.step(int(policy[cur_state]))
          episode.append(Step(cur_state=cur_state, action=self.actions[action], next_state=next_state, reward=reward, done=is_done))
          if is_done:
              break
      trajs.append(episode)
    return trajs


