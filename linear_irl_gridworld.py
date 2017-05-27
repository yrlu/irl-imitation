import numpy as np
import matplotlib.pyplot as plt
import argparse

import img_utils
from mdp import gridworld
from mdp import value_iteration
from lp_irl import *

PARSER = argparse.ArgumentParser(description=None)
PARSER.add_argument('-l', '--l1', default=10, type=float, help='l1 regularization')
PARSER.add_argument('-g', '--gamma', default=0.5, type=float, help='discount factor')
PARSER.add_argument('-r', '--r_max', default=10, type=float, help='maximum value of reward')
PARSER.add_argument('-a', '--act_random', default=0.3, type=float, help='probability of acting randomly')
ARGS = PARSER.parse_args()
print ARGS


GAMMA = ARGS.gamma
ACT_RAND = ARGS.act_random
R_MAX = ARGS.r_max
L1 = ARGS.l1


def main():
  """
  Recover gridworld reward using linear programming IRL
  """

  H = 10
  W = 10
  N_STATES = H * W
  N_ACTIONS = 5

  # init the gridworld
  grid = [['0', '0', '0', '0', '0', '0', '0', '0', '0', '0'],
          ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0'],
          ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0'],
          ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0'],
          ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0'],
          ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0'],
          ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0'],
          ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0'],
          ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0'],
          ['0', '0', '0', '0', '0', '0', '0', '0', '0', str(R_MAX)]]

  gw = gridworld.GridWorld(grid, {(H - 1, W - 1)}, 1 - ACT_RAND)

  # solve the MDP using value iteration
  vi = value_iteration.ValueIterationAgent(gw, GAMMA, 100)

  r_mat = gw.get_reward_mat()
  print 'show rewards map. any key to continue'
  img_utils.heatmap2d(r_mat, 'Reward Map - Ground Truth')

  v_mat = gw.get_values_mat(vi.get_values())
  print 'show values map. any key to continue'
  img_utils.heatmap2d(v_mat, 'Value Map - Ground Truth')

  # Construct transition matrix
  P_a = np.zeros((N_STATES, N_STATES, N_ACTIONS))

  for si in range(N_STATES):
    statei = gw.idx2pos(si)
    for a in range(N_ACTIONS):
      probs = gw.get_transition_states_and_probs(statei, a)
      for statej, prob in probs:
        sj = gw.pos2idx(statej)
        # Prob of si to sj given action a
        P_a[si, sj, a] = prob

  # display policy and value in gridworld just for debug use
  gw.display_policy_grid(vi.get_optimal_policy())
  gw.display_value_grid(vi.values)

  # setup policy
  policy = np.zeros(N_STATES)
  for i in range(N_STATES):
    policy[i] = vi.get_action(gw.idx2pos(i))

  # solve for the rewards
  rewards = lp_irl(P_a, policy, gamma=GAMMA, l1=L1, R_max=R_MAX)

  # display recoverred rewards
  print 'show recoverred rewards map. any key to continue'
  img_utils.heatmap2d(np.reshape(rewards, (H, W), order='F'), 'Reward Map - Recovered')
  img_utils.heatmap3d(np.reshape(rewards, (H, W), order='F'), 'Reward Map - Recovered')


if __name__ == "__main__":
  main()
