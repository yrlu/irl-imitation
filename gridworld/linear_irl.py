import img_utils 
import numpy as np
import gridworld
import value_iteration
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt


H = 10
W = 10
N_STATES = H*W
N_ACTIONS = 5
GAMMA = 0.5
# with probability of ACT_RAND not following the action given
ACT_RAND = 0.3
R_MAX = 10


def get_values_mat(gridworld, values):
  """
  inputs:
    values: a dictionary {<state, value>}
  """
  shape = np.shape(gridworld.grid)
  v_mat = np.zeros(shape)
  for i in range(shape[0]):
    for j in range(shape[1]):
      v_mat[i,j] = values[(i,j)]
  return v_mat

def get_reward_mat(gridworld):
  """
  Get reward matrix from gridworld
  """
  shape = np.shape(gridworld.grid)
  r_mat = np.zeros(shape)
  for i in range(shape[0]):
    for j in range(shape[1]):
      r_mat[i,j] = float(gridworld.grid[i][j])
  return r_mat


def pos2idx(pos, H=H, W=W):
  """
  input: 
    column-major 2d position
  returns:
    1d index 
  """
  return pos[0] + pos[1]*H

def idx2pos(idx, H=H, W=W):
  """
  input:
    1d idx
  returns:
    2d column-major position
  """
  return (idx % H, idx / H)

# grid = [['0', '0'],
#         ['0', str(R_MAX)]]

# grid = [['0', '0', '0'],
#         ['0', '0', '0'],
#         ['0', '0', str(R_MAX)]]

# grid = [['0', '0', '0', '0', '0'],
#         ['0', '0', '0', '0', '0'],
#         ['0', '0', '0', '0', '0'],
#         ['0', '0', '0', '0', '0'],
#         ['0', '0', '0', '0', str(R_MAX)]]


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


gw = gridworld.GridWorld(grid, {(H-1, W-1)}, 1-ACT_RAND)
vi = value_iteration.ValueIterationAgent(gw, GAMMA, 100)
gw.display_policy_grid(vi.get_optimal_policy())

v_mat = get_values_mat(gw, vi.get_values())

print 'show values map. any key to continue'
img_utils.heatmap2d(v_mat, 'Value Map - Ground Truth')

r_mat = get_reward_mat(gw)
print 'show rewards map. any key to continue'
img_utils.heatmap2d(r_mat, 'Reward Map - Ground Truth')

P_a = np.zeros((N_STATES, N_STATES, N_ACTIONS))
print P_a.shape

for si in range(N_STATES):
  statei = idx2pos(si)
  for a in range(N_ACTIONS):
    probs = gw.get_transition_states_and_probs(statei, a)
    for statej, prob in probs:
      sj = pos2idx(statej)
      # Prob of si to sj given action a
      P_a[si, sj, a] = prob

# for debug
print '-----'
print gw.get_transition_states_and_probs((H-1, W-1), 0)
dirs = {0: 'r', 1: 'l', 2: 'd', 3: 'u', 4: 's'}
print dirs
print np.reshape(P_a[pos2idx([H-2,H-2]),:,vi.get_action([H-2,H-2])], (H,W), order='F')
print P_a.shape
gw.display_value_grid(vi.values)

# Formulate a linear IRL problem
A = np.zeros([2*N_STATES*(N_ACTIONS+1), 3*N_STATES])
b = np.zeros([2*N_STATES*(N_ACTIONS+1)])
c = np.zeros([3*N_STATES])

for i in range(N_STATES):
  a_opt = vi.get_action(idx2pos(i))
  tmp_inv = np.linalg.inv(np.identity(N_STATES) - GAMMA*P_a[:,:,a_opt])

  cnt = 0
  for a in range(N_ACTIONS):
    if a != a_opt:
      A[i*(N_ACTIONS-1)+cnt,:N_STATES] = -np.dot(P_a[i,:,a_opt] - P_a[i,:,a], tmp_inv)
      A[N_STATES*(N_ACTIONS-1) + i*(N_ACTIONS-1)+cnt,:N_STATES] = -np.dot(P_a[i,:,a_opt] - P_a[i,:,a], tmp_inv)
      A[N_STATES*(N_ACTIONS-1) + i*(N_ACTIONS-1)+cnt, N_STATES + i] = 1
      cnt += 1

for i in range(N_STATES):
  A[2*N_STATES*(N_ACTIONS-1) + i, i] = 1
  b[2*N_STATES*(N_ACTIONS-1) + i] = R_MAX

for i in range(N_STATES):
  A[2*N_STATES*(N_ACTIONS-1) + N_STATES + i, i] = -1
  b[2*N_STATES*(N_ACTIONS-1) + N_STATES + i] = 0

for i in range(N_STATES):
  A[2*N_STATES*(N_ACTIONS-1) + 2*N_STATES + i, i] = 1
  A[2*N_STATES*(N_ACTIONS-1) + 2*N_STATES + i, 2*N_STATES + i] = -1

for i in range(N_STATES):
  A[2*N_STATES*(N_ACTIONS-1) + 3*N_STATES + i, i] = 1
  A[2*N_STATES*(N_ACTIONS-1) + 3*N_STATES + i, 2*N_STATES + i] = -1

l1 = 10
for i in range(N_STATES):
  c[N_STATES:2*N_STATES] = -1
  c[2*N_STATES:] = l1

sol=solvers.lp(matrix(c),matrix(A),matrix(b))

rewards = sol['x'][:N_STATES]

def normalize(vals):
  """
  normalize to (0, max_val)
  input: 
    vals: 1d array
  """
  min_val = np.min(vals)
  max_val = np.max(vals)
  return (vals - min_val)/(max_val - min_val)

rewards = normalize(rewards)
print rewards
img_utils.heatmap2d(np.reshape(rewards*R_MAX, (H,W), order='F'), 'Reward Map - Recovered')
img_utils.heatmap3d(np.reshape(rewards*R_MAX, (H,W), order='F'), 'Reward Map - Recovered')
