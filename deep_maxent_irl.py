import numpy as np
import tensorflow as tf
import mdp.gridworld as gridworld
import mdp.value_iteration as value_iteration
import img_utils
import tf_utils
from utils import *



class DeepIRLFC:


  def __init__(self, n_input, lr, n_h1=400, n_h2=300, name='deep_irl_fc'):
    self.n_input = n_input
    self.lr = lr
    self.n_h1 = n_h1
    self.n_h2 = n_h2
    self.name = name

    self.sess = tf.Session()
    self.input_s, self.reward, self.theta = self._build_network(self.name)
    # self.optimizer = tf.train.AdamOptimizer(lr)
    self.optimizer = tf.train.GradientDescentOptimizer(lr)
    
    # self.demo_svf = tf.placeholder(tf.float32, [None, self.n_input])
    self.grad_r = tf.placeholder(tf.float32, [None])
    self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.theta]) 
    # self.loss = tf.reduce_sum(self.reward*self.grad_r) + 0.02*self.l2_loss
    self.grad_l2 = tf.gradients(self.l2_loss, self.theta)
    self.grad_theta = tf.gradients(self.reward, self.theta)
    self.grad_theta = [-self.grad_r*g for g in self.grad_theta]
    # self.grad_theta = tf.gradients(self.reward, self.theta, -self.grad_r)
    # self.grad_theta = [tf.scalar_mul(self.grad_r, g) for g in self.grad_theta]
    # print self.grad_theta
    # print self.reward.get_shape()
    # print self.grad_r.get_shape()
    # print zip(self.grad_theta, self.theta)
    # self.grad = [tf.add(0.02*self.grad_l2[i], self.grad_theta[i]) for i in range(len(self.grad_l2))]
    # self.grad = tf.add(self.grad_l2, self.grad_theta)
    # self.loss = tf.reduce_sum(self.reward*self.grad_r)
    # self.optimize = tf.train.AdamOptimizer().minimize(self.loss)
    self.optimize = self.optimizer.apply_gradients(zip(self.grad_theta, self.theta))
    self.sess.run(tf.global_variables_initializer())
    # var_grad = tf.gradients(self.reward, [self.theta])[0]


  def _build_network(self, name):
    input_s = tf.placeholder(tf.float32, [None, self.n_input])
    with tf.variable_scope(name):
      fc1 = tf_utils.fc(input_s, self.n_h1, scope="fc1", activation_fn=tf.nn.relu)
        # initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
      # fc1 = tf.nn.batch_normalization(fc1, 0, 1, 0, 1, 1e-10)
      # fc2 = tf_utils.fc(fc1, self.n_h2, scope="fc2", activation_fn=tf.nn.relu,
        # initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
      # fc2 = tf.nn.batch_normalization(fc2, 0, 1, 0, 1, 1e-10)
      # reward = tf_utils.fc(input_s, 1, scope="reward", activation_fn=tf.sigmoid)
      reward = tf_utils.fc(fc1, 1, scope="reward")
      # reward = tf.nn.batch_normalization(reward, 0, 1, 0, 1, 1e-10)
      # reward = tf.contrib.layers.fully_connected(input_s, num_outputs=1, activation_fn=None)
    theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
    # print theta
    # print theta[0].get_shape()
    # print theta[1].get_shape()
    return input_s, reward, theta


  def get_theta(self):
    return self.sess.run(self.theta)


  def get_rewards(self, states):
    rewards = self.sess.run(self.reward, feed_dict={self.input_s: states})
    # print np.reshape(rewards, (3,3)).T
    return rewards


  def apply_grads(self, feat_map, grad_r):
    _, grad_theta = self.sess.run([self.optimize, self.grad_theta], feed_dict={self.grad_r: grad_r, self.input_s: feat_map})
    return grad_theta



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


def demo_svf(trajs, n_states):
  """
  compute state visitation frequences from demonstrations
  
  input:
    trajs   list of list of Steps - collected from expert
  returns:
    p       Nx1 vector - state visitation frequences   
  """

  p = np.zeros(n_states)
  for traj in trajs:
    for step in traj:
      p[step.cur_state] += 1
  p = p/len(trajs)
  return p



def deep_maxent_irl(feat_map, P_a, gamma, trajs, lr, n_iters):
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

  # tf.set_random_seed(1)

  N_STATES, _, N_ACTIONS = np.shape(P_a)

  # init nn model
  nn_r = DeepIRLFC(feat_map.shape[1], lr, 3, 3)

  # find state visitation frequencies using demonstrations
  mu_D = demo_svf(trajs, N_STATES)
  # print 'mu_D:', np.reshape(mu_D, (5,5)).T 
  # training 
  for iteration in range(n_iters):
    # if iteration % (n_iters/20) == 0:
    print 'iteration: {}'.format(iteration)
    # print np.shape(feat_map)
    # compute the reward matrix
    rewards = nn_r.get_rewards(feat_map)
    # rewards = normalize(rewards)
    print np.reshape(rewards, (5,5), order='F')
    # print nn_r.get_theta()[0]
    # print np.reshape(nn_r.get_theta()[0], (5,5), order='F')
    # img_utils.heatmap2d(np.reshape(rewards, (5,5), order='F'), '', block=True, fig_num=1)
    # compute policy 
    _, policy = value_iteration.value_iteration(P_a, rewards, gamma, error=0.01, deterministic=True)
    # img_utils.heatmap2d(np.reshape(rewards, (3,3), order='F'), '', block=True)
    # compute expected svf
    mu_exp = compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=True)
    # print policy
    # print mu_exp
    # compute gradients on rewards:
    grad_r = mu_D - mu_exp
    # print grad_r
    # print np.sum(mu_exp)
    # print 'mu_exp:', np.reshape(mu_exp, (5,5)).T 
    # print grad_r
    # print np.reshape(grad_r, (3,3)).T
    # print np.shape(grad_r)
    # print np.shape(feat_map)

    # apply gradients to the neural network
    # acc_grad = np.zeros(N_STATES)
    for i in range(N_STATES):
      grad_theta = nn_r.apply_grads([feat_map[i,:]], [grad_r[i]])
      # print grad_theta
      # acc_grad += grad_theta[0]
    # nn_r.apply_grads(feat_map, grad_r)
    # print np.reshape(acc_grad, (5,5), order='F')

    # print np.reshape(grad_theta[0], (5,2), order='F')
    # img_utils.heatmap2d(np.reshape(grad_theta[0], (5,2), order='F'), '', block=True, fig_num=2)


  rewards = nn_r.get_rewards(feat_map)
  return normalize(rewards)
  # return rewards





