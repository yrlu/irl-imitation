import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
import time

import mdp.gridworld as gridworld
import mdp.value_iteration as value_iteration
import img_utils
import tf_utils
from utils import *



class DeepIRLFC:


  def __init__(self, n_input, n_actions, lr, T, n_h1=400, n_h2=300, l2=10, deterministic=False, sparse=False, name='deep_irl_fc'):
    self.n_input = n_input
    self.lr = lr
    self.n_h1 = n_h1
    self.n_h2 = n_h2
    self.name = name
    self.sparse = sparse
    self.deterministic = deterministic

    self.sess = tf.Session()
    self.input_s, self.reward, self.theta = self._build_network(self.name)

    # value iteration
    if sparse:
        self.P_a = tf.sparse_placeholder(tf.float32, shape=(n_input, n_actions, n_input))
    else:
        self.P_a = tf.placeholder(tf.float32, shape=(n_input, n_actions, n_input))
    self.gamma = tf.placeholder(tf.float32)
    self.epsilon = tf.placeholder(tf.float32)
    self.values, self.policy = self._vi(self.reward)

    # state visitation frequency
    self.T = T
    self.mu = tf.placeholder(tf.float32, n_input, name='mu_placerholder')

    self.svf = self._svf(self.policy)

    self.optimizer = tf.train.GradientDescentOptimizer(lr)
    
    self.grad_r = tf.placeholder(tf.float32, [n_input, 1])
    self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.theta])
    self.grad_l2 = tf.gradients(self.l2_loss, self.theta)

    self.grad_theta = tf.gradients(self.reward, self.theta, -self.grad_r)
    # apply l2 loss gradients
    self.grad_theta = [tf.add(l2*self.grad_l2[i], self.grad_theta[i]) for i in range(len(self.grad_l2))]
    self.grad_theta, _ = tf.clip_by_global_norm(self.grad_theta, 100.0)

    self.grad_norms = tf.global_norm(self.grad_theta)
    self.optimize = self.optimizer.apply_gradients(zip(self.grad_theta, self.theta))
    self.sess.run(tf.global_variables_initializer())


  def _build_network(self, name):
    input_s = tf.placeholder(tf.float32, [self.n_input, self.n_input])
    with tf.variable_scope(name):
      fc1 = tf_utils.fc(input_s, self.n_h1, scope="fc1", activation_fn=tf.nn.elu,
        initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
      fc2 = tf_utils.fc(fc1, self.n_h2, scope="fc2", activation_fn=tf.nn.elu,
        initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
      reward = tf_utils.fc(fc2, 1, scope="reward")
    theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
    return input_s, reward, theta

  def _vi(self, rewards):

      rewards = tf.squeeze(rewards)

      def body(i, c, t):
          old_values = t.read(i)
          if self.sparse:
              new_values = tf.sparse_reduce_max(
                  tf.sparse_reduce_sum_sparse(self.P_a * (rewards + self.gamma * old_values), axis=2), axis=1)
          else:
            new_values = tf.reduce_max(tf.reduce_sum(self.P_a * (rewards + self.gamma * old_values), axis=2), axis=1)

          c = tf.reduce_max(tf.abs(new_values - old_values)) > self.epsilon
          c.set_shape(())
          t = t.write(i + 1, new_values)
          return i + 1, c, t

      def condition(i, c, t):
          return c

      t = tf.TensorArray(dtype=tf.float32, size=350, clear_after_read=True)
      t = t.write(0, tf.constant(0, dtype=tf.float32, shape=(self.n_input,)))

      i, _, values = tf.while_loop(condition, body, [0, True, t], parallel_iterations=1, back_prop=False,
                                   name='VI_loop')
      values = values.read(i)

      if self.deterministic:
          if self.sparse:
              policy = tf.argmax(tf.sparse_tensor_to_dense(tf.sparse_reduce_sum_sparse(self.P_a * (rewards + self.gamma * values), axis=2)), axis=1)
          else:
              policy = tf.argmax(tf.reduce_sum(self.P_a * (rewards + self.gamma * values), axis=2), axis=1)
      else:
          if self.sparse:
              policy = tf.sparse_tensor_to_dense(
                  tf.sparse_reduce_sum_sparse(self.P_a * (rewards + self.gamma * values), axis=2))
          else:
              policy = tf.reduce_sum(self.P_a * (rewards + self.gamma * values), axis=2)

          policy = tf.nn.softmax(policy)

      return values, policy

  def _svf(self, policy):
      if self.deterministic:
        r = tf.range(self.n_input, dtype=tf.int64)
        expanded = tf.expand_dims(r, 1)
        tiled = tf.tile(expanded, [1, self.n_input])
        indices = tf.stack([tiled] + tf.meshgrid(r, policy), axis=2)
        P_a_cur_policy = tf.gather_nd(self.P_a, indices)

      # if deterministic:
      #   mu[start:end, t + 1] = np.sum(mu[:, t, np.newaxis] * P_az[:, start:end], axis=0)
      # else:
      #   mu[start:end, t + 1] = np.sum(np.sum(mu[:, t, np.newaxis, np.newaxis] * (P_a[:, :, start:end] * policy[:, :, np.newaxis]), axis=1), axis=0)

      cur_mu = self.mu
      mu = self.mu
      with tf.variable_scope('svf'):
          if self.deterministic:
              for t in range(self.T - 1):
                  cur_mu = tf.reduce_sum(cur_mu * P_a_cur_policy, axis=1)
                  mu += cur_mu
          else:
              for t in range(self.T - 1):
                  cur_mu = tf.reduce_sum(tf.reduce_sum(tf.tile(tf.expand_dims(cur_mu, 1), [1, tf.shape(policy)[1]]) * tf.transpose(self.P_a, (0, 2, 1)) * policy, axis=2), axis=1)
                  mu += cur_mu

      return mu


  def get_theta(self):
    return self.sess.run(self.theta)

  def get_rewards(self, states):
    rewards = self.sess.run(self.reward, feed_dict={self.input_s: states})
    return rewards

  def get_policy(self, states, P_a, gamma, epsilon=0.01):
    return self.sess.run([self.reward, self.values, self.policy],
                         feed_dict={self.input_s: states, self.P_a: P_a, self.gamma: gamma, self.epsilon: epsilon})

  def get_policy_svf(self, states, P_a, gamma, p_start_state, epsilon=0.01):
      return self.sess.run([self.reward, self.values, self.policy, self.svf],
                           feed_dict={self.input_s: states, self.P_a: P_a, self.gamma: gamma, self.mu: p_start_state, self.epsilon: epsilon})

  def apply_grads(self, feat_map, grad_r):
    grad_r = np.reshape(grad_r, [-1, 1])
    feat_map = np.reshape(feat_map, [-1, self.n_input])
    _, grad_theta, l2_loss, grad_norms = self.sess.run([self.optimize, self.grad_theta, self.l2_loss, self.grad_norms], 
      feed_dict={self.grad_r: grad_r, self.input_s: feat_map})
    return grad_theta, l2_loss, grad_norms


def start_state_probs(trajs, n_states):
    p_start_state = np.zeros([n_states])

    for traj in trajs:
        p_start_state[traj[0].cur_state] += 1
    p_start_state = p_start_state[:] / len(trajs)

    return p_start_state

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
  tt = time.time()
  N_STATES, _, N_ACTIONS = np.shape(P_a)

  T = len(trajs[0])
  # mu[s, t] is the prob of visiting state s at time t
  mu = np.zeros([N_STATES, T])

  mu[:, 0] = start_state_probs(trajs, N_STATES)

  num_cpus = multiprocessing.cpu_count()
  chunk_size = N_STATES // num_cpus

  if deterministic:
    P_az = P_a[np.arange(0, N_STATES), :, policy]
  else:
    P_a = P_a.transpose(0, 2, 1)

  def step(t, start, end):
      if deterministic:
        mu[start:end, t + 1] = np.sum(mu[:, t, np.newaxis] * P_az[:, start:end], axis=0)
      else:
        mu[start:end, t + 1] = np.sum(np.sum(mu[:, t, np.newaxis, np.newaxis] * (P_a[:, :, start:end] * policy[:, :, np.newaxis]), axis=1), axis=0)

  with ThreadPoolExecutor(max_workers=num_cpus) as e:
    for t in range(T - 1):
      futures = list()
      for i in range(N_STATES):
          futures.append(e.submit(step, t, i, min(N_STATES, i + chunk_size)))

      for f in futures:
          # Force throwing an exception if thrown by step()
          f.result()

  # for t in range(T - 1):
  #   mu[:, t+1] = (mu[:, t]*P_a[np.arange(0, N_STATES), :, policy]).sum(axis=1)

  p = np.sum(mu, 1)

  print(time.time() - tt)
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

def deep_maxent_irl(feat_map, P_a, gamma, trajs, lr, n_iters, sparse):
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
  nn_r = DeepIRLFC(feat_map.shape[1], N_ACTIONS, lr, len(trajs), 3, 3, deterministic=False, sparse=sparse)

  # find state visitation frequencies using demonstrations
  mu_D = demo_svf(trajs, N_STATES)
  p_start_state = start_state_probs(trajs, N_STATES)

  P_a_t = P_a.transpose(0, 2, 1)
  if sparse:
    mask = P_a_t > 0
    indices = np.argwhere(mask)
    P_a_t = tf.SparseTensorValue(indices, P_a_t[mask], P_a_t.shape)

  # training 
  for iteration in range(n_iters):
    if iteration % (n_iters/10) == 0:
      print 'iteration: {}'.format(iteration)

    # compute the reward matrix
    # rewards = nn_r.get_rewards(feat_map)

    # compute policy
    # _, policy = value_iteration.value_iteration(P_a, rewards, gamma, error=0.01, deterministic=False)

    # compute rewards and policy at the same time
    t = time.time()
    #rewards, _, policy = nn_r.get_policy(feat_map, P_a_t, gamma, 0.01)
    #print('tensorflow VI', time.time() - t)
    
    # compute expected svf
    #mu_exp = compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=False)

    rewards, _, policy, mu_exp = nn_r.get_policy_svf(feat_map, P_a_t, gamma, p_start_state, 0.01)
    # compute gradients on rewards:
    grad_r = mu_D - mu_exp

    # apply gradients to the neural network
    grad_theta, l2_loss, grad_norm = nn_r.apply_grads(feat_map, grad_r)
    

  rewards = nn_r.get_rewards(feat_map)
  # return sigmoid(normalize(rewards))
  return normalize(rewards)





