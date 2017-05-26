import img_utils 
import numpy as np
import gridworld
import value_iteration
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt

# test heatmap util function
# hm = np.random.random((16, 16))
# img_utils.heatmap2d(hm)




H = 10
W = 10
N_STATES = H*W
N_ACTIONS = 5
GAMMA = 0.8
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

# plt.figure(figsize=(10,8))

print 'show values map. any key to continue'
# plt.subplot(2,2,1)
img_utils.heatmap2d(v_mat, 'Value Map - Ground Truth')

r_mat = get_reward_mat(gw)
print 'show rewards map. any key to continue'
# plt.subplot(2,2,2)
img_utils.heatmap2d(r_mat, 'Reward Map - Ground Truth')


# print gw.get_transition_states_and_probs((0,0), 0)
# print gw.get_transition_states_and_probs((0,0), 1)
# print gw.get_transition_states_and_probs((0,0), 2)
# print gw.get_transition_states_and_probs((0,0), 3)
# print gw.get_transition_states_and_probs((0,0), 4)
# print gw.get_actions((0,0))

# construct the P_a* 
P_a = np.zeros((N_STATES, N_STATES, N_ACTIONS))
print P_a.shape

# print pos2idx((0,0))
# print pos2idx((0,1))
# print pos2idx((1,0))
# print idx2pos(0)
# print idx2pos(1)
# assert(pos2idx((0,0))==0)
# assert(pos2idx((0,1))==2)
# assert(pos2idx((1,0))==1)
# assert(idx2pos(0)==(0,0))
# assert(idx2pos(1)==(1,0))
# assert(idx2pos(2)==(0,1))
# assert(pos2idx(idx2pos(2))==2)
# assert(pos2idx(idx2pos(3))==3)
# assert(pos2idx(idx2pos(1))==1)



for si in range(N_STATES):
  statei = idx2pos(si)
  for a in range(N_ACTIONS):
    probs = gw.get_transition_states_and_probs(statei, a)
    for statej, prob in probs:
      sj = pos2idx(statej)
      # Prob of si to sj given action a
      P_a[si, sj, a] = prob
print '-----'
print gw.get_transition_states_and_probs((H-1, W-1), 0)
# print gw.get_transition_states_and_probs([3,4], vi.get_action([3,4]))
dirs = {0: 'r', 1: 'l', 2: 'd', 3: 'u', 4: 's'}
print dirs
print np.reshape(P_a[pos2idx([H-2,H-2]),:,vi.get_action([H-2,H-2])], (H,W), order='F')
print np.reshape(P_a[pos2idx([4,3]),:,vi.get_action([4,3])], (H,W), order='F')
print np.reshape(P_a[pos2idx([4,4]),:,vi.get_action([4,4])], (H,W), order='F')
print P_a.shape

gw.display_value_grid(vi.values)

# print vi.get_action(idx2pos(2))


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
# print sol['x']

# print A
# print A.shape
# print b
# print b.shape
# print c
# print c.shape

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
# print rewards
# img_utils.heatmap2d(np.reshape(sol['x'][:N_STATES], (H,W), order='F'))
# plt.subplot(2,2,3)
img_utils.heatmap2d(np.reshape(rewards*R_MAX, (H,W), order='F'), 'Reward Map - Recovered')
img_utils.heatmap3d(np.reshape(rewards*R_MAX, (H,W), order='F'), 'Reward Map - Recovered')

# # formulate linear IRL problem
# A = np.zeros((2*N_STATES*(N_ACTIONS-1) + N_STATES*2, N_STATES*2))
# b = np.zeros((2*N_STATES*(N_ACTIONS-1) + N_STATES*2))
# c = np.zeros((2*N_STATES))

# for si in range(N_STATES):
#   # get optimal action
#   a_opt = vi.get_action(idx2pos(si))
#   tmp_inv = np.linalg.inv(np.identity(N_STATES) - GAMMA*P_a[:,:,a_opt])
#   i = 0
#   for a in range(N_ACTIONS):
#     if a != a_opt:
#       A[si*(N_ACTIONS-1)+i,:N_STATES] = -np.dot(np.reshape(P_a[si,:,a_opt] - P_a[si,:,a], (1,N_STATES)), tmp_inv)
#       A[si*(N_ACTIONS-1)+ N_STATES*(N_ACTIONS-1) +i,:N_STATES] = -np.dot(np.reshape(P_a[si,:,a_opt] - P_a[si,:,a], (1,N_STATES)), tmp_inv)
#       A[si*(N_ACTIONS-1)+ N_STATES*(N_ACTIONS-1) +i, N_STATES + si] = 1
#       i += 1

# for i in range(N_STATES):
#   A[N_STATES*(N_ACTIONS-1)*2+i,i] = 1
#   b[N_STATES*(N_ACTIONS-1)*2+i] = R_MAX

# for i in range(N_STATES):
#   A[N_STATES*(N_ACTIONS-1)*2+N_STATES+i,i] = -1
#   b[N_STATES*(N_ACTIONS-1)*2+N_STATES+i] = R_MAX

# for i in range(N_STATES):
#   # c[i] = 1
#   c[N_STATES + i] = -1




# print A.shape
# print A
# print b.shape
# print b

# sol=solvers.lp(matrix(c),matrix(A),matrix(b))
# print sol['x'][:N_STATES]

# img_utils.heatmap2d(np.reshape(sol['x'][:N_STATES], (H,W), order='F'))

# raw_input()
