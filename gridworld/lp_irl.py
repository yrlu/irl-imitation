'''
Implementation of linear programming inverse reinforcement learning in
  Ng & Russell 2000 paper: Algorithms for Inverse Reinforcement Learning
  http://ai.stanford.edu/~ang/papers/icml00-irl.pdf

By Yiren Lu (luyirenmax@gmail.com), May 2017
'''
import numpy as np
from cvxopt import matrix, solvers


def normalize(vals):
  """
  normalize to (0, max_val)
  input:
    vals: 1d array
  """
  min_val = np.min(vals)
  max_val = np.max(vals)
  return (vals - min_val) / (max_val - min_val)


def lp_irl(trans_probs, policy, gamma=0.5, l1=10, R_max=10):
  """
  inputs:
    trans_probs       NxNxN_ACTIONS transition matrix
    policy            policy vector / map
    R_max             maximum possible value of recoverred rewards
    gamma             RL discount factor
    l1                l1 regularization lambda
  returns:
    rewards           Nx1 reward vector
  """
  print np.shape(trans_probs)
  N_STATES, _, N_ACTIONS = np.shape(trans_probs)
  print N_STATES, N_ACTIONS
  # Formulate a linear IRL problem
  A = np.zeros([2 * N_STATES * (N_ACTIONS + 1), 3 * N_STATES])
  b = np.zeros([2 * N_STATES * (N_ACTIONS + 1)])
  c = np.zeros([3 * N_STATES])

  for i in range(N_STATES):
    a_opt = policy[i]
    tmp_inv = np.linalg.inv(np.identity(N_STATES) - gamma * trans_probs[:, :, a_opt])

    cnt = 0
    for a in range(N_ACTIONS):
      if a != a_opt:
        A[i * (N_ACTIONS - 1) + cnt, :N_STATES] = - \
            np.dot(trans_probs[i, :, a_opt] - trans_probs[i, :, a], tmp_inv)
        A[N_STATES * (N_ACTIONS - 1) + i * (N_ACTIONS - 1) + cnt, :N_STATES] = - \
            np.dot(trans_probs[i, :, a_opt] - trans_probs[i, :, a], tmp_inv)
        A[N_STATES * (N_ACTIONS - 1) + i * (N_ACTIONS - 1) + cnt, N_STATES + i] = 1
        cnt += 1

  for i in range(N_STATES):
    A[2 * N_STATES * (N_ACTIONS - 1) + i, i] = 1
    b[2 * N_STATES * (N_ACTIONS - 1) + i] = R_max

  for i in range(N_STATES):
    A[2 * N_STATES * (N_ACTIONS - 1) + N_STATES + i, i] = -1
    b[2 * N_STATES * (N_ACTIONS - 1) + N_STATES + i] = 0

  for i in range(N_STATES):
    A[2 * N_STATES * (N_ACTIONS - 1) + 2 * N_STATES + i, i] = 1
    A[2 * N_STATES * (N_ACTIONS - 1) + 2 * N_STATES + i, 2 * N_STATES + i] = -1

  for i in range(N_STATES):
    A[2 * N_STATES * (N_ACTIONS - 1) + 3 * N_STATES + i, i] = 1
    A[2 * N_STATES * (N_ACTIONS - 1) + 3 * N_STATES + i, 2 * N_STATES + i] = -1

  for i in range(N_STATES):
    c[N_STATES:2 * N_STATES] = -1
    c[2 * N_STATES:] = l1

  sol = solvers.lp(matrix(c), matrix(A), matrix(b))
  rewards = sol['x'][:N_STATES]
  rewards = normalize(rewards) * R_max
  return rewards
