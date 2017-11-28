'''
Implementation of maximum entropy inverse reinforcement learning in
  Ziebart et al. 2008 paper: Maximum Entropy Inverse Reinforcement Learning
  https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf

Acknowledgement:
  This implementation is largely influenced by Matthew Alger's maxent implementation here:
  https://github.com/MatthewJA/Inverse-Reinforcement-Learning/blob/master/irl/maxent.py

By Yiren Lu (luyirenmax@gmail.com), May 2017
'''
import numpy as np
import mdp.gridworld as gridworld
import mdp.value_iteration as value_iteration
import img_utils
from utils import *


def compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=True):
  """compute the expected states visition frequency p(s| theta, T) 
  using dynamic programming

  inputs:
    P_a     NxNxN_ACTIONS matrix - transition dynamics
    gamma   float - discount factor
    trajs   list of list of Steps - collected from expert
    policy  Nx1 vector (or NxN_ACTIONS if deterministic=False) - policy

  
  returns:
    p       Nx1 vector - state visitation frequencies
  """
  N_STATES, _, N_ACTIONS = np.shape(P_a)

  T = len(trajs[0])
  # mu[s, t] is the prob of visiting state s at time t
  mu = np.zeros([N_STATES, T]) 

  for traj in trajs:
    mu[traj[0].cur_state, 0] += 1
  mu[:,0] = mu[:,0]/len(trajs)

  for s in range(N_STATES):
    for t in range(T-1):
      if deterministic:
        mu[s, t+1] = sum([mu[pre_s, t]*P_a[pre_s, s, int(policy[pre_s])] for pre_s in range(N_STATES)])
      else:
        mu[s, t+1] = sum([sum([mu[pre_s, t]*P_a[pre_s, s, a1]*policy[pre_s, a1] for a1 in range(N_ACTIONS)]) for pre_s in range(N_STATES)])
  p = np.sum(mu, 1)
  return p



def maxent_irl(feat_map, P_a, gamma, trajs, lr, n_iters):
  """
  Maximum Entropy Inverse Reinforcement Learning (Maxent IRL)

  inputs:
    feat_map    NxD matrix - the features for each state
    P_a         NxNxN_ACTIONS matrix - P_a[s0, s1, a] is the transition prob of 
                                       landing at state s1 when taking action 
                                       a at state s0
    gamma       float - RL discount factor
    trajs       a list of demonstrations
    lr          float - learning rate
    n_iters     int - number of optimization steps

  returns
    rewards     Nx1 vector - recoverred state rewards
  """
  N_STATES, _, N_ACTIONS = np.shape(P_a)

  # init parameters
  theta = np.random.uniform(size=(feat_map.shape[1],))

  # calc feature expectations
  feat_exp = np.zeros([feat_map.shape[1]])
  for episode in trajs:
    for step in episode:
      feat_exp += feat_map[step.cur_state,:]
  feat_exp = feat_exp/len(trajs)

  # training
  for iteration in range(n_iters):
  
    if iteration % (n_iters/20) == 0:
      print 'iteration: {}/{}'.format(iteration, n_iters)
    
    # compute reward function
    rewards = np.dot(feat_map, theta)

    # compute policy
    _, policy = value_iteration.value_iteration(P_a, rewards, gamma, error=0.01, deterministic=False)
    
    # compute state visition frequences
    svf = compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=False)
    
    # compute gradients
    grad = feat_exp - feat_map.T.dot(svf)

    # update params
    theta += lr * grad

  rewards = np.dot(feat_map, theta)
  # return sigmoid(normalize(rewards))
  return normalize(rewards)

def value(policy, n_states, transition_probabilities, reward, discount,
                    threshold=1e-2):
    """
    FROM https://github.com/MatthewJA/Inverse-Reinforcement-Learning/blob/master/irl/value_iteration.py#L10

    Find the value function associated with a policy.

    policy: List of action ints for each state.
    n_states: Number of states. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    -> Array of values for each state
    """
    v = np.zeros(n_states)

    diff = float("inf")
    while diff > threshold:
        diff = 0
        for s in range(n_states):
            vs = v[s]
            a = policy[s]
            v[s] = sum(transition_probabilities[s, a, k] *
                       (reward[k] + discount * v[k])
                       for k in range(n_states))
            diff = max(diff, abs(vs - v[s]))

    return v

def expected_value_diff(P_a, rewards, true_rewards, gamma, p_start, optimal_value, policy, error=0.01, deterministic=True):
  v = value(policy, P_a.shape[0], P_a.transpose(0, 2, 1), true_rewards, gamma)

  return optimal_value.dot(p_start) - v.dot(p_start)
