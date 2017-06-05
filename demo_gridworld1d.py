import numpy as np
import matplotlib.pyplot as plt
import argparse

import img_utils
from mdp import gridworld1d
from mdp import value_iteration
from lp_irl import *
from maxent_irl import *
from deep_maxent_irl import *


PARSER = argparse.ArgumentParser(description=None)
PARSER.add_argument('-ns', '--n_states', default=100, type=int, help='number of states in the 1d gridworld')
PARSER.add_argument('-g', '--gamma', default=0.5, type=float, help='discount factor')
PARSER.add_argument('-a', '--act_random', default=0.0, type=float, help='probability of acting randomly')
PARSER.add_argument('-t', '--n_trajs', default=500, type=int, help='number of expert trajectories')
PARSER.add_argument('-l', '--l_traj', default=20, type=int, help='length of expert trajectory')
PARSER.add_argument('--rand_start', dest='rand_start', action='store_true', help='when sampling trajectories, randomly pick start positions')
PARSER.add_argument('--no-rand_start', dest='rand_start',action='store_false', help='when sampling trajectories, fix start positions')
PARSER.set_defaults(rand_start=True)
PARSER.add_argument('-lr', '--learning_rate', default=0.02, type=float, help='learning rate')
PARSER.add_argument('-ni', '--n_iters', default=20, type=int, help='number of iterations')
ARGS = PARSER.parse_args()
print ARGS


GAMMA = ARGS.gamma
ACT_RAND = ARGS.act_random
R_MAX = 1 # the constant r_max does not affect much the recoverred reward distribution
N_STATES = ARGS.n_states
N_ACTIONS = 2
N_TRAJS = ARGS.n_trajs
L_TRAJ = ARGS.l_traj
RAND_START = ARGS.rand_start
LEARNING_RATE = ARGS.learning_rate
N_ITERS = ARGS.n_iters
SIGMA = 0.5


def to_plot(map, n=N_STATES):
  return np.repeat(np.reshape(map, [n,1]), 10, axis=1)

def feat(s):
  feat_vec = np.zeros(N_STATES)
  for i in range(N_STATES):
    # by approximity
    feat_vec[i] = np.exp(-abs(s-i)/(2*SIGMA**2))
  return feat_vec

def main():
  # init the gridworld
  rmap_gt = np.zeros(N_STATES)
  rmap_gt[N_STATES-5] = R_MAX
  rmap_gt[10] = R_MAX

  gw = gridworld1d.GridWorld1D(rmap_gt, {}, ACT_RAND)
  P_a = gw.get_transition_mat()
  values_gt, policy_gt = value_iteration.value_iteration(P_a, rmap_gt, GAMMA, error=0.01, deterministic=True)

  # gradient rewards 
  rmap_gt = values_gt
  gw = gridworld1d.GridWorld1D(rmap_gt, {}, ACT_RAND)
  P_a = gw.get_transition_mat()
  values_gt, policy_gt = value_iteration.value_iteration(P_a, rmap_gt, GAMMA, error=0.01, deterministic=True)

  # np.random.seed(1)
  trajs = gw.generate_demonstrations(policy_gt, n_trajs=N_TRAJS, len_traj=L_TRAJ, rand_start=RAND_START)  
  
  # feat_map = np.eye(N_STATES)
  feat_map = np.array([feat(s) for s in range(N_STATES)])
  test_irl_algorithms(gw, P_a, rmap_gt, policy_gt, trajs, feat_map)



def test_irl_algorithms(gw, P_a, rmap_gt, policy_gt, trajs, feat_map):
  print 'LP IRL training ..'
  rewards_lpirl = lp_irl(P_a, policy_gt, gamma=0.3, l1=10, R_max=R_MAX)
  print 'Max Ent IRL training ..'
  rewards_maxent = maxent_irl(feat_map, P_a, GAMMA, trajs, LEARNING_RATE*2, N_ITERS*2)
  print 'Deep Max Ent IRL training ..'
  rewards = deep_maxent_irl(feat_map, P_a, GAMMA, trajs, LEARNING_RATE, N_ITERS)    
  values, _ = value_iteration.value_iteration(P_a, rewards, GAMMA, error=0.01, deterministic=True)

  # plots
  plt.figure(figsize=(20,8))
  plt.subplot(1, 4, 1)
  img_utils.heatmap2d(to_plot(rmap_gt), 'Rewards Map - Ground Truth', block=False, text=False)
  plt.subplot(1, 4, 2)
  img_utils.heatmap2d(to_plot(rewards_lpirl), 'Reward Map - LP', block=False, text=False)
  plt.subplot(1, 4, 3)
  img_utils.heatmap2d(to_plot(rewards_maxent), 'Reward Map - Maxent', block=False, text=False)
  plt.subplot(1, 4, 4)
  img_utils.heatmap2d(to_plot(rewards), 'Reward Map - Deep Maxent', block=False, text=False)
  plt.show()


if __name__ == "__main__":
  main()