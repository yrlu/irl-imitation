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
PARSER.add_argument('-m', '--model_dir', default='cartpole-model', type=str, help='model directory (no slash at the end)')
PARSER.add_argument('-t', '--train', default=False, type=str, help='train for [number of episodes] IF MODEL EXISTS')
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

TRAIN = ARGS.train

MODEL_DIR = ARGS.model_dir
MODEL_PATH = MODEL_DIR + '/model'
MEMORY_PATH = MODEL_DIR + '/memory.p'


def train(agent, exprep, env):
  for i in xrange(NUM_EPISODES):
    cur_state = env.reset()
    for t in xrange(MAX_STEPS):
      action = agent.get_action(cur_state)
      next_state, reward, done, info = env.step(action)
      if done:
        reward = FAIL_PENALTY
        exprep.add_step(Step(cur_step=cur_state, action=action, next_step=next_state, reward=reward, done=done))
        print("Episode {} finished after {} timesteps".format(i, t + 1))
        yield t + 1 
        break
      exprep.add_step(Step(cur_step=cur_state, action=action, next_step=next_state, reward=reward, done=done))
      cur_state = next_state
      if t == MAX_STEPS - 1:
        print("Episode {} finished after {} timesteps".format(i, t + 1))
        yield t + 1
    agent.epsilon_decay()
    agent.learn_epoch(exprep, EPOCH_SIZE)
    print 'epsilon: {}'.format(agent.epsilon)


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
  if TRAIN:
    exprep = pickle.load(open(MEMORY_PATH,"rb"))
    history = [e_length for e_length in train(agent, exprep, env)]
    saver.save(sess, MODEL_PATH)
    pickle.dump(exprep, open(MEMORY_PATH, "wb"))
    print 'saved model'
    # plot
    import matplotlib.pyplot as plt
    avg_reward = [np.mean(history[i*10:(i+1)*10]) for i in xrange(int(len(history)/10))]
    f_reward = plt.figure(1)
    plt.plot(np.linspace(0, len(history), len(avg_reward)), avg_reward)
    plt.ylabel('Episode length')
    plt.xlabel('Training episodes')
    f_reward.show()
    print 'press enter to continue'
    raw_input()
    plt.close()

else:
  os.makedirs(MODEL_DIR)
  history = [e_length for e_length in train(agent, exprep, env)]
  saver.save(sess, MODEL_PATH)
  pickle.dump(exprep, open(MEMORY_PATH, "wb"))
  print 'saved model'
  # plot
  import matplotlib.pyplot as plt
  avg_reward = [np.mean(history[i*10:(i+1)*10]) for i in xrange(int(len(history)/10))]
  f_reward = plt.figure(1)
  plt.plot(np.linspace(0, len(history), len(avg_reward)), avg_reward)
  plt.ylabel('Episode length')
  plt.xlabel('Training episodes')
  f_reward.show()
  print 'press enter to continue'
  raw_input()
  plt.close()


# Display:
print 'press ctrl-c to stop'
while True:
  cur_state = env.reset()
  done = False
  t = 0
  while not done:
    env.render()
    t = t+1
    action = agent.get_optimal_action(cur_state)
    next_state, reward, done, info = env.step(action)
    cur_state = next_state
    if done:
      print("Episode finished after {} timesteps".format(t+1))
      break
    





