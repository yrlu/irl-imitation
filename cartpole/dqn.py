# Deep Q-learning agent with q-value approximation
# Following paper: Playing Atari with Deep Reinforcement Learning
#     https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
#
# ---
# @author Yiren Lu
# @email luyiren [at] seas [dot] upenn [dot] edu
#
# MIT License


import gym
import numpy as np
import random
import tensorflow as tf
import tf_utils


class DQNAgent():
  """
  DQN Agent with a 2-hidden-layer fully-connected q-network that acts epsilon-greedily.
  """

  def __init__(self,
    session,
    epsilon=0.5, 
    epsilon_anneal = 0.01,
    end_epsilon=0.1,
    lr=0.5, 
    gamma=0.99,
    state_size=4,
    action_size=2,
    scope="dqn",
    n_hidden_1=20,
    n_hidden_2=20,
    ):
    """
    args
      epsilon           exploration rate
      epsilon_anneal    linear decay rate per call of epsilon_decay() function
      end_epsilon       lowest exploration rate
      lr                learning rate
      gamma             discount factor
      state_size        network input size
      action_size       network output size
    """
    self.epsilon = epsilon
    self.epsilon_anneal = epsilon_anneal
    self.end_epsilon = end_epsilon
    self.lr = lr
    self.gamma = gamma
    self.state_size = state_size
    self.action_size = action_size
    self.scope = scope
    self.n_hidden_1 = n_hidden_1
    self.n_hidden_2 = n_hidden_2
    self._build_qnet()
    self.sess = session

  def _build_qnet(self):
    """
    Build q-network
    """
    with tf.variable_scope(self.scope):
      self.state_input = tf.placeholder(tf.float32, [None, self.state_size])
      self.action = tf.placeholder(tf.int32, [None])
      self.target_q = tf.placeholder(tf.float32, [None])

      fc1 = tf_utils.fc(self.state_input, n_output=self.n_hidden_1, activation_fn=tf.nn.relu)
      fc2 = tf_utils.fc(fc1, n_output=self.n_hidden_2, activation_fn=tf.nn.relu)
      self.q_values = tf_utils.fc(fc2, self.action_size, activation_fn=None)

      action_mask = tf.one_hot(self.action, self.action_size, 1.0, 0.0)
      q_value_pred = tf.reduce_sum(self.q_values * action_mask, 1)

      self.loss = tf.reduce_mean(tf.square(tf.subtract(self.target_q, q_value_pred)))
      self.optimizer = tf.train.AdamOptimizer(self.lr)
      self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

  def get_action_values(self, state):
    actions = self.sess.run(self.q_values, feed_dict={self.state_input: [state]})
    return actions

  def get_optimal_action(self, state):
    actions = self.sess.run(self.q_values, feed_dict={self.state_input: [state]})
    return actions.argmax()

  def get_action(self, state):
    """
    Epsilon-greedy action

    args
      state           current state      
    returns
      an action to take given the state
    """
    if np.random.random() < self.epsilon:
      # act randomly
      return np.random.randint(0, self.action_size)
    else:
      return self.get_optimal_action(state)

  def epsilon_decay(self):    
    if self.epsilon > self.end_epsilon:
      self.epsilon = self.epsilon - self.epsilon_anneal

  def learn_epoch(self, exprep, num_steps):
    """
    Deep Q-learing: train qnetwork for num_steps, for each step, sample a batch from exprep

    Args
      exprep:         experience replay
      num_steps:      num of steps
    """
    for i in xrange(num_steps):
      self.learn_batch(exprep.sample())

  def learn_batch(self, batch_steps):
    """
    Deep Q-learing: train qnetwork with the input batch
    Args
      batch_steps:    a batch of sampled namedtuple Step, where Step.cur_step and 
                      Step.next_step are of shape {self.state_size}
      sess:           tf session
    Returns 
      batch loss (-1 if input is empty)
    """
    if len(batch_steps) == 0:
      return -1

    next_state_batch = [s.next_step for s in batch_steps]
    q_values = self.sess.run(self.q_values, feed_dict={self.state_input: next_state_batch})

    max_q_values = q_values.max(axis=1)
    # compute target q value
    target_q = np.array([s.reward + self.gamma*max_q_values[i]*(1-s.done) for i,s in enumerate(batch_steps)])
    target_q = target_q.reshape([len(batch_steps)])
    
    # minimize the TD-error
    cur_state_batch = [s.cur_step for s in batch_steps]
    actions = [s.action for s in batch_steps]
    l, _, = self.sess.run([self.loss, self.train_op], feed_dict={ self.state_input: cur_state_batch,
                                                                  self.target_q: target_q,
                                                                  self.action: actions })
    return l

