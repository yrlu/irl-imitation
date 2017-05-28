import numpy as np
from cvxopt import matrix, solvers
import mdp.gridworld as gridworld
import mdp.value_iteration as value_iteration
import img_utils
from utils import *


def compute_state_visition_freq(P_a, gamma, reward, trajs, policy):
  """compute the states visition frequency p(s| theta, T)"""

  N_STATES, _, N_ACTIONS = np.shape(P_a)


  T = len(trajs[0])
  # mu[s, t] is the prob of visiting state s at time t
  mu = np.zeros([N_STATES, T]) 
  # init mu_1(s)
  # mu[0, 0] = T

  for traj in trajs:
    mu[traj[0].cur_state, 0] += 1
  mu[:,0] = mu[:,0]/len(trajs)

  for t in range(1, T):
    for s in range(N_STATES):
      mu[s, t] += sum([mu[pre_s, t-1]*P_a[pre_s, s, int(policy[pre_s])] for pre_s in range(N_STATES)])


  # for s in range(N_STATES):
    # for t in range(T-1):
      # mu[s, t+1] = sum([mu[pre_s, t]*P_a[pre_s, s, int(policy[pre_s])] for pre_s in range(N_STATES)])
  p = np.sum(mu, 1)
  return p


def maxent(feat_map, P_a, gamma, trajs, lr, n_iters):
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
  """
  N_STATES, _, N_ACTIONS = np.shape(P_a)

  # init parameters
  # theta = np.ones([feat_map.shape[1]])/N_STATES
  theta = np.random.uniform(size=(feat_map.shape[1],))
  

  # calc feature expectations
  feat_exp = np.zeros([feat_map.shape[1]])
  for episode in trajs:
    for step in episode:
      feat_exp += feat_map[step.cur_state,:]
  feat_exp = feat_exp/len(trajs)
  print feat_exp

  # training
  for iteration in range(n_iters):
    print iteration
    
    # compute reward function
    rewards = np.dot(feat_map, theta)

    # print rewards
    # compute policy
    _, policy = value_iteration.value_iteration(P_a, rewards, gamma, error=0.01, deterministic=True)
    
    # print policy
    # policy[-1] = 4
    # compute state visition frequences
    svf = compute_state_visition_freq(P_a, gamma, rewards, trajs, policy)
    
    # compute gradients
    grad = feat_exp - feat_map.T.dot(svf)
    # for debug
    # if iteration % 50 == 0:
    #   H = 5
    #   W = 5
    #   print np.reshape(rewards, (H,W), order='F')
    #   img_utils.heatmap2d(np.reshape(normalize(rewards)*10, (H,W), order='F'), 'Reward Map - Recovered')
    #   # print np.reshape(grad, (H,W), order='F')
    #   print np.reshape(P_a[-1, :, 0], (H,W), order='F')
    #   print np.reshape(P_a[-1, :, 1], (H,W), order='F')

    # update params
    theta += lr * grad

  rewards = np.dot(feat_map, theta)
  # return normalize(rewards)
  return rewards


