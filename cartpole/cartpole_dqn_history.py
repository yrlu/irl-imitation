'''To collect dqn expert history on carpole environment. No training'''
import argparse
import gym
import numpy as np
import sys
import tensorflow as tf
import dqn
import exp_replay
from exp_replay import Step
import matplotlib.pyplot as plt
import os
import pickle


PARSER = argparse.ArgumentParser(description=None)
PARSER.add_argument('-d', '--device', default='cpu', type=str, help='choose device: cpu/gpu')
PARSER.add_argument('-e', '--episodes', default=150, type=int, help='number of episodes')
PARSER.add_argument('-m', '--model_dir', default='cartpole-model/', type=str, help='model directory')
PARSER.add_argument('-hf', '--history_file', default='cartpole-model/history.p', type=str, help='history file path')
ARGS = PARSER.parse_args()
print ARGS


DEVICE = ARGS.device
NUM_EPISODES = ARGS.episodes
ACTIONS = {0:0, 1:1}
MAX_STEPS = 300
FAIL_PENALTY = 0
EPSILON = 1
EPSILON_DECAY = 0.01
END_EPSILON = 0.1
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.9
BATCH_SIZE = 32
MEM_SIZE = 1e4
START_MEM = 1e2
STATE_SIZE = [4]
EPOCH_SIZE = 100

MODEL_DIR = ARGS.model_dir
MODEL_PATH = MODEL_DIR + 'model'
MEMORY_PATH = MODEL_DIR + 'memory.p'
HISTORY_PATH = ARGS.history_file

env = gym.make('CartPole-v0')
exprep = exp_replay.ExpReplay(mem_size=MEM_SIZE, start_mem=START_MEM, state_size=STATE_SIZE, kth=-1, batch_size=BATCH_SIZE)

sess = tf.Session()
with tf.device('/{}:0'.format(DEVICE)):
  agent = dqn.DQNAgent(session=sess, epsilon=EPSILON, epsilon_anneal=EPSILON_DECAY, end_epsilon=END_EPSILON, 
        lr=LEARNING_RATE, gamma=DISCOUNT_FACTOR, state_size=4, 
        action_size=len(ACTIONS), n_hidden_1=10, n_hidden_2=10)

sess.run(tf.initialize_all_variables())
saver = tf.train.Saver()
if os.path.isdir(MODEL_DIR):
  saver.restore(sess, MODEL_PATH)
  agent.epsilon = agent.end_epsilon
  print 'restored model'

else:
  print 'Wrong model directory. Abort'
  exit



history = []

# Display:
print 'press ctrl-c to stop'
for e in range(NUM_EPISODES):
  episode = []
  cur_state = env.reset()
  done = False
  t = 0
  while not done:
    env.render()
    t = t+1
    action = agent.get_optimal_action(cur_state)
    q_values = agent.get_action_values(cur_state)
    # print np.amax(q_values), np.amin(q_values), np.amax(q_values) - np.amin(q_values)
    next_state, reward, done, info = env.step(action)
    episode.append(Step(cur_step=cur_state, action=action, next_step=next_state, reward=reward, done=done))
    cur_state = next_state
    if done:
      print("Episode {} finished after {} timesteps".format(e, t+1))
      break
  history.append(episode)

pickle.dump(exprep, open(HISTORY_PATH, "wb"))
print 'history saved'



