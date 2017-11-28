import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import namedtuple


import img_utils
from mdp import gridworld
from mdp import value_iteration
from deep_maxent_irl import *
from maxent_irl import *
from utils import *
from lp_irl import *

Step = namedtuple('Step','cur_state action next_state reward done')


PARSER = argparse.ArgumentParser(description=None)
PARSER.add_argument('-hei', '--height', default=5, type=int, help='height of the gridworld')
PARSER.add_argument('-wid', '--width', default=5, type=int, help='width of the gridworld')
PARSER.add_argument('-g', '--gamma', default=0.9, type=float, help='discount factor')
PARSER.add_argument('-a', '--act_random', default=0.3, type=float, help='probability of acting randomly')
PARSER.add_argument('-t', '--n_trajs', default=200, type=int, help='number of expert trajectories')
PARSER.add_argument('-l', '--l_traj', default=20, type=int, help='length of expert trajectory')
PARSER.add_argument('--rand_start', dest='rand_start', action='store_true', help='when sampling trajectories, randomly pick start positions')
PARSER.add_argument('--no-rand_start', dest='rand_start',action='store_false', help='when sampling trajectories, fix start positions')
PARSER.set_defaults(rand_start=True)
PARSER.add_argument('-lr', '--learning_rate', default=0.02, type=float, help='learning rate')
PARSER.add_argument('-ni', '--n_iters', default=20, type=int, help='number of iterations')
PARSER.add_argument('-s', '--sparse', default=False, action='store_true', help='flag to use sparse tensors in tf')
ARGS = PARSER.parse_args()
print ARGS


GAMMA = ARGS.gamma
ACT_RAND = ARGS.act_random
R_MAX = 1 # the constant r_max does not affect much the recoverred reward distribution
H = ARGS.height
W = ARGS.width
N_TRAJS = ARGS.n_trajs
L_TRAJ = ARGS.l_traj
RAND_START = ARGS.rand_start
LEARNING_RATE = ARGS.learning_rate
N_ITERS = ARGS.n_iters


def generate_demonstrations(gw, policy, n_trajs=100, len_traj=20, rand_start=False, start_pos=[0,0]):
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
      start_pos = [np.random.randint(0, gw.height), np.random.randint(0, gw.width)]

    episode = []
    gw.reset(start_pos)
    cur_state = start_pos
    cur_state, action, next_state, reward, is_done = gw.step(int(policy[gw.pos2idx(cur_state)]))
    episode.append(Step(cur_state=gw.pos2idx(cur_state), action=action, next_state=gw.pos2idx(next_state), reward=reward, done=is_done))
    # while not is_done:
    for _ in range(len_traj):
        cur_state, action, next_state, reward, is_done = gw.step(int(policy[gw.pos2idx(cur_state)]))
        episode.append(Step(cur_state=gw.pos2idx(cur_state), action=action, next_state=gw.pos2idx(next_state), reward=reward, done=is_done))
        if is_done:
            break
    trajs.append(episode)
  return trajs


def main():
  N_STATES = H * W
  N_ACTIONS = 5

  rmap_gt = np.zeros([H, W])
  rmap_gt[H-1, W-1] = R_MAX
  rmap_gt[0, W-1] = R_MAX
  rmap_gt[H-1, 0] = R_MAX

  gw = gridworld.GridWorld(rmap_gt, {}, 1 - ACT_RAND)

  rewards_gt = np.reshape(rmap_gt, H*W, order='F')
  P_a = gw.get_transition_mat()

  values_gt, policy_gt = value_iteration.value_iteration(P_a, rewards_gt, GAMMA, error=0.01, deterministic=True)
  
  # use identity matrix as feature
  feat_map = np.eye(N_STATES)

  trajs = generate_demonstrations(gw, policy_gt, n_trajs=N_TRAJS, len_traj=L_TRAJ, rand_start=RAND_START)

  mu = np.zeros([N_STATES])

  for traj in trajs:
    mu[traj[0].cur_state] += 1
  mu = mu / len(trajs)
  
  print 'Deep Max Ent IRL training ..'
  t = time.time()
  rewards = deep_maxent_irl(feat_map, P_a, GAMMA, trajs, LEARNING_RATE, N_ITERS, ARGS.sparse)
  print('time for dirl', time.time() - t)

  values, policy = value_iteration.value_iteration(P_a, rewards, GAMMA, error=0.01, deterministic=True)

  print('evd', expected_value_diff(P_a, rewards, rewards_gt, GAMMA, mu, values_gt, policy))

  # plots
  plt.figure(figsize=(20,4))
  plt.subplot(1, 4, 1)
  img_utils.heatmap2d(np.reshape(rewards_gt, (H,W), order='F'), 'Rewards Map - Ground Truth', block=False)
  plt.subplot(1, 4, 2)
  img_utils.heatmap2d(np.reshape(values_gt, (H,W), order='F'), 'Value Map - Ground Truth', block=False)
  plt.subplot(1, 4, 3)
  img_utils.heatmap2d(np.reshape(rewards, (H,W), order='F'), 'Reward Map - Recovered', block=False)
  plt.subplot(1, 4, 4)
  img_utils.heatmap2d(np.reshape(values, (H,W), order='F'), 'Value Map - Recovered', block=False)
  plt.show()




if __name__ == "__main__":
  main()
