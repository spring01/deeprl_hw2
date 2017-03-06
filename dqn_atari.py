#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import random
import subprocess

from copy import deepcopy

import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute)
from keras.models import Model, Sequential
from keras.optimizers import Adam

import deeprl_hw2 as tfrl
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.objectives import mean_huber_loss
from deeprl_hw2.preprocessors import AtariPreprocessor, HistoryPreprocessor
from deeprl_hw2.policy import *


def create_model(window, input_shape, num_actions,
                 model_name='q_network'):  # noqa: D103
    """Create the Q-network model.

    Use Keras to construct a keras.models.Model instance (you can also
    use the SequentialModel class).

    We highly recommend that you use tf.name_scope as discussed in
    class when creating the model and the layers. This will make it
    far easier to understnad your network architecture if you are
    logging with tensorboard.

    Parameters
    ----------
    window: int
      Each input to the network is a sequence of frames. This value
      defines how many frames are in the sequence.
    input_shape: tuple(int, int)
      The expected input image size.
    num_actions: int
      Number of possible actions. Defined by the gym environment.
    model_name: str
      Useful when debugging. Makes the model show up nicer in tensorboard.

    Returns
    -------
    keras.models.Model
      The Q-model.
    """
    pass


def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    
    subprocess.call(["mkdir", "-p", parent_dir])
    #~ os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    return parent_dir


def main():  # noqa: D103
    parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
    parser.add_argument('--env', default='Breakout-v0', help='Atari env name')
    parser.add_argument(
        '-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    
    parser.add_argument('--input_shape', nargs=2, type=int, default=None,
                        help='Input shape')
    parser.add_argument('--num_frame', default=4, type=int,
                        help='Number of frames in a state')
    parser.add_argument('--discount', default=0.99, type=float,
                        help='Discount factor gamma')
    parser.add_argument('--target_update_freq', default=1.0, type=float,
                        help='Frequency to update the target network')
    parser.add_argument('--num_burn_in', default=1000, type=int,
                        help='Number of samples filled in memory before update')
    parser.add_argument('--train_freq', default=1, type=int,
                        help='How often you actually update your Q-Network')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='How many samples in each minibatch')
    
    parser.add_argument('--learning_rate', default=1e-4, type=float,
                        help='Learning rate alpha')
    parser.add_argument('--explore_prob', default=0.05, type=float,
                        help='Exploration probability in epsilon-greedy')
    
    args = parser.parse_args()
    args.input_shape = tuple(args.input_shape)

    args.output = get_output_folder(args.output, args.env)
    
    
    import gym
    env = gym.make(args.env)
    #~ env.reset()
    
    num_actions = len(env.get_action_meanings())
    len_input = np.prod(args.input_shape) * args.num_frame
    
    #~ if args.model == 'LinQN':
    model = Sequential()
    model.add(Dense(num_actions, input_dim=len_input))
    model.compile(loss='mse', optimizer='rmsprop')
    
    q_network = {}
    q_network['online'] = model
    q_network['target'] = deepcopy(q_network['online'])
    
    proc = {}
    proc['atari'] = AtariPreprocessor(args.input_shape)
    proc['history'] = HistoryPreprocessor(args.input_shape)
    memory = None
    
    policy = {}
    policy['train'] = GreedyEpsilonPolicy(args.explore_prob, num_actions)
    
    agent = DQNAgent(model, proc, memory, policy,
                     args.discount, args.target_update_freq,
                     args.num_burn_in, args.train_freq, args.batch_size)
    
    
    
    
    #~ observation, reward, done, info = env.step(env.action_space.sample())
    #~ done = False
    #~ proc_obs = proc.process_state_for_network(observation)
    #~ frame_buffer = [proc_obs for _ in range(args.num_frame)]
    
    for _ in range(100):
        env.reset()
        observation, reward, done, info = env.step(env.action_space.sample())
        proc_obs = proc.process_state_for_network(observation)
        frame_buffer = [proc_obs for _ in range(args.num_frame)]
        state_batch = []
        target_batch = []
        while not done:
        #~ for _ in range(500):
            env.render()
            state_vec = np.concatenate([f.ravel() for f in frame_buffer])
            state_vec = state_vec.reshape(1, -1)
            #~ import pdb; pdb.set_trace()
            q_values = model.predict(state_vec)
            action = np.argmax(q_values)
            observation, reward, done, info = env.step(action)
            
            state_batch.append(state_vec)
            
            target = np.zeros(num_actions)
            if done:
                target[action] = reward
            else:
                target[action] = reward + args.discount * np.max(q_values)
            
            target_batch.append(target)
            
            proc_obs = proc.process_state_for_network(observation)
            frame_buffer.pop(0)
            frame_buffer.append(proc_obs)
        
        state_batch = np.vstack(state_batch)
        target_batch = np.vstack(target_batch)
        #~ import pdb; pdb.set_trace()
        print model.train_on_batch(state_batch, target_batch)
        
        
        
    

    # here is where you should start up a session,
    # create your DQN agent, create your model, etc.
    # then you can run your fit method.

if __name__ == '__main__':
    main()
