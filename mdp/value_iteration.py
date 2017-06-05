# Value iteration agent
# Model-based learning which requires mdp.
#
# ---
# @author Yiren Lu
# @email luyiren [at] seas [dot] upenn [dot] edu
#
# MIT License

import math
import numpy as np


def value_iteration(P_a, rewards, gamma, error=0.01, deterministic=True):
  """
  static value iteration function. Perhaps the most useful function in this repo
  
  inputs:
    P_a         NxNxN_ACTIONS transition probabilities matrix - 
                              P_a[s0, s1, a] is the transition prob of 
                              landing at state s1 when taking action 
                              a at state s0
    rewards     Nx1 matrix - rewards for all the states
    gamma       float - RL discount
    error       float - threshold for a stop
    deterministic   bool - to return deterministic policy or stochastic policy
  
  returns:
    values    Nx1 matrix - estimated values
    policy    Nx1 (NxN_ACTIONS if non-det) matrix - policy
  """
  N_STATES, _, N_ACTIONS = np.shape(P_a)

  values = np.zeros([N_STATES])

  # estimate values
  while True:
    values_tmp = values.copy()

    for s in range(N_STATES):
      v_s = []
      values[s] = max([sum([P_a[s, s1, a]*(rewards[s] + gamma*values_tmp[s1]) for s1 in range(N_STATES)]) for a in range(N_ACTIONS)])

    if max([abs(values[s] - values_tmp[s]) for s in range(N_STATES)]) < error:
      break


  if deterministic:
    # generate deterministic policy
    policy = np.zeros([N_STATES])
    for s in range(N_STATES):
      policy[s] = np.argmax([sum([P_a[s, s1, a]*(rewards[s]+gamma*values[s1]) 
                                  for s1 in range(N_STATES)]) 
                                  for a in range(N_ACTIONS)])

    return values, policy
  else:
    # generate stochastic policy
    policy = np.zeros([N_STATES, N_ACTIONS])
    for s in range(N_STATES):
      v_s = np.array([sum([P_a[s, s1, a]*(rewards[s] + gamma*values[s1]) for s1 in range(N_STATES)]) for a in range(N_ACTIONS)])
      policy[s,:] = np.transpose(v_s/np.sum(v_s))
    return values, policy




class ValueIterationAgent(object):

  def __init__(self, mdp, gamma, iterations=100):
    """
    The constructor builds a value model from mdp using dynamic programming
    
    inputs:
      mdp       markov decision process that is required by value iteration agent definition: 
                https://github.com/stormmax/reinforcement_learning/blob/master/envs/mdp.py
      gamma     discount factor
    """
    self.mdp = mdp
    self.gamma = gamma
    states = mdp.get_states()
    # init values
    self.values = {}

    for s in states:
      if mdp.is_terminal(s):
        self.values[s] = mdp.get_reward(s)
      else:
        self.values[s] = 0

    # estimate values
    for i in range(iterations):
      values_tmp = self.values.copy()

      for s in states:
        if mdp.is_terminal(s):
          continue

        actions = mdp.get_actions(s)
        v_s = []
        for a in actions:
          P_s1sa = mdp.get_transition_states_and_probs(s, a)
          R_sas1 = [mdp.get_reward(s1) for s1 in [p[0] for p in P_s1sa]]
          v_s.append(sum([P_s1sa[s1_id][1] * (mdp.get_reward(s) + gamma *
                                              values_tmp[P_s1sa[s1_id][0]]) for s1_id in range(len(P_s1sa))]))
        # V(s) = max_{a} \sum_{s'} P(s'| s, a) (R(s,a,s') + \gamma V(s'))
        self.values[s] = max(v_s)

  def get_values(self):
    """
    returns
      a dictionary {<state, value>}
    """
    return self.values

  def get_q_values(self, state, action):
    """
    returns qvalue of (state, action)
    """
    return sum([P_s1_s_a * (self.mdp.get_reward_sas(s, a, s1) + self.gamma * self.values[s1])
                for s1, P_s1_s_a in self.mdp.get_transition_states_and_probs(state, action)])

  def eval_policy_dist(self, policy, iterations=100):
    """
    evaluate a policy distribution
    returns
      a map {<state, value>}
    """
    values = {}
    states = self.mdp.get_states()
    for s in states:
      if self.mdp.is_terminal(s):
        values[s] = self.mdp.get_reward(s)
      else:
        values[s] = 0

    for i in range(iterations):
      values_tmp = values.copy()

      for s in states:
        if self.mdp.is_terminal(s):
          continue
        actions = self.mdp.get_actions(s)
        # v(s) = \sum_{a\in A} \pi(a|s) (R(s,a,s') + \gamma \sum_{s'\in S}
        # P(s'| s, a) v(s'))
        values[s] = sum([policy[s][i][1] * (self.mdp.get_reward(s) + self.gamma * sum([s1_p * values_tmp[s1]
                                                                                       for s1, s1_p in self.mdp.get_transition_states_and_probs(s, actions[i])]))
                         for i in range(len(actions))])
    return values


  def get_optimal_policy(self):
    """
    returns
      a dictionary {<state, action>}
    """
    states = self.mdp.get_states()
    policy = {}
    for s in states:
      policy[s] = [(self.get_action(s), 1)]
    return policy


  def get_action_dist(self, state):
    """
    args
      state    current state
    returns
      a list of {<action, prob>} pairs representing the action distribution on state
    """
    actions = self.mdp.get_actions(state)
    # \sum_{s'} P(s'|s,a)*(R(s,a,s') + \gamma v(s'))
    v_a = [sum([s1_p * (self.mdp.get_reward_sas(state, a, s1) + self.gamma * self.values[s1])
                for s1, s1_p in self.mdp.get_transition_states_and_probs(state, a)])
           for a in actions]

    # I exponentiated the v_s^a's to make them positive
    v_a = [math.exp(v) for v in v_a]
    return [(actions[i], v_a[i] / sum(v_a)) for i in range(len(actions))]

  def get_action(self, state):
    """
    args
      state    current state
    returns
      an action to take given the state
    """
    actions = self.mdp.get_actions(state)
    v_s = []
    for a in actions:
      P_s1sa = self.mdp.get_transition_states_and_probs(state, a)
      R_sas1 = [self.mdp.get_reward(s1) for s1 in [p[0] for p in P_s1sa]]
      v_s.append(sum([P_s1sa[s1_id][1] *
                      (self.mdp.get_reward(state) +
                       self.gamma *
                       self.values[P_s1sa[s1_id][0]]) for s1_id in range(len(P_s1sa))]))
    a_id = v_s.index(max(v_s))
    return actions[a_id]







