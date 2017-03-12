#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import sys
import random
import subprocess

import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute, Lambda, Merge, merge)
from keras import backend as K
from keras.initializations import normal, identity
from keras.models import Model, Sequential
from keras.optimizers import Adam

import deeprl_hw2 as tfrl
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.objectives import mean_huber_loss
from deeprl_hw2.preprocessors import AtariPreprocessor
from deeprl_hw2.policy import *

import gym
from collections import deque
import cPickle as pickle

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
    if model_name == 'linear':
        model_input_shape = (np.prod(input_shape) * window,)
        model = Sequential()
        model.add(Dense(num_actions, input_dim=model_input_shape[0]))
    elif model_name == 'dqn':
        model_input_shape = tuple([window] + list(input_shape))
        model = Sequential()
        conv1 = Convolution2D(32, 8, 8, subsample=(4, 4),
            init='uniform',
            border_mode='same', input_shape=model_input_shape)
        model.add(conv1)
        model.add(Activation('relu'))
        conv2 = Convolution2D(64, 4, 4, subsample=(2, 2),
            init='uniform',
            border_mode='same')
        model.add(conv2)
        model.add(Activation('relu'))
        conv3 = Convolution2D(64, 3, 3, subsample=(2, 2),
            init='uniform',
            border_mode='same')
        model.add(conv3)
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512, init='uniform'))
        model.add(Activation('relu'))
        model.add(Dense(num_actions, init='uniform'))
    elif model_name == 'dueling':
        model_input_shape = tuple([window] + list(input_shape))
        inputs = Input(shape=model_input_shape)
        conv1 = Convolution2D(32, 8, 8, subsample=(4, 4), init='uniform',
            border_mode='same', activation='relu')(inputs)
        conv2 = Convolution2D(64, 4, 4, subsample=(2, 2), init='uniform',
            border_mode='same', activation='relu')(conv1)
        conv3 = Convolution2D(64, 3, 3, subsample=(1, 1), init='uniform',
            border_mode='same', activation='relu')(conv2)
        feature = Flatten()(conv3)
        value1 = Dense(512, init='uniform')(feature)
        value2 = Dense(1, init='uniform')(value1)
        advantage1 = Dense(512, init='uniform')(feature)
        advantage2 = Dense(num_actions, init='uniform')(advantage1)
        mean_advantage2 = Lambda(lambda x: K.mean(x, axis=1))(advantage2)
        ones = K.ones([1, num_actions])
        exp_mean_advantage2 = Lambda(lambda x: K.dot(K.expand_dims(x, dim=1), -ones))(mean_advantage2)
        sum_adv = merge([exp_mean_advantage2, advantage2], mode='sum')
        
        exp_value2 = Lambda(lambda x: K.dot(x, ones))(value2)
        q_value = merge([exp_value2, sum_adv], mode='sum')
        
        #~ import pdb; pdb.set_trace()
        model = Model(input=inputs, output=q_value)
    return model, model_input_shape


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
    subprocess.call(["mkdir", "-p", parent_dir])
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
    parser.add_argument('--replay_buffer_size', default=0, type=int,
                        help='Replay buffer size')
    parser.add_argument('--target_reset_interval', default=10000, type=int,
                        help='Interval to reset the target network')
    parser.add_argument('--num_burn_in', default=1000, type=int,
                        help='Number of samples filled in memory before update')
    parser.add_argument('--train_freq', default=1, type=int,
                        help='How often you actually update your Q-Network')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='How many samples in each minibatch')
    
    parser.add_argument('--learning_rate', default=1e-6, type=float,
                        help='Learning rate alpha')
    parser.add_argument('--explore_prob', default=0.05, type=float,
                        help='Exploration probability in epsilon-greedy')
    parser.add_argument('--num_train', default=5000000, type=int,
                        help='Number of training sampled interactions with the environment')
    parser.add_argument('--max_episode_length', default=None, type=int,
                        help='Number of training sampled interactions with the environment')
    
    parser.add_argument('--model_name', default='linear', type=str,
                        help='Model name')
    parser.add_argument('--read_weight', default=None, type=str,
                        help='Read weight from $read_weight/online_weight.save')
    
    parser.add_argument('--eval_interval', default=100000, type=int,
                        help='Evaluation interval')
    parser.add_argument('--eval_episodes', default=20, type=int,
                        help='Number of episodes in evaluation')
    
    parser.add_argument('--double_net', default=False, type=bool,
                        help='Invode double Q net')
    
    args = parser.parse_args()
    args.input_shape = tuple(args.input_shape)
    
    args.output = get_output_folder(args.output, args.env)
    
    
    env = gym.make(args.env)
    
    num_actions = env.action_space.n
    
    opt_adam = Adam(lr=args.learning_rate)
    
    model_online, model_input_shape = create_model(args.num_frame, args.input_shape,
        num_actions, model_name=args.model_name)
    model_target, model_input_shape = create_model(args.num_frame, args.input_shape,
        num_actions, model_name=args.model_name)
    
    q_network = {}
    q_network['online'] = model_online
    q_network['target'] = model_target
    
    proc = AtariPreprocessor(args.input_shape)
    
    if args.replay_buffer_size == 0:
        memory = None
    else:
        memory = deque(maxlen=(args.replay_buffer_size / args.num_frame))
    
    policy = {}
    policy['train'] = LinearDecayGreedyEpsilonPolicy(1.0,
                                                     args.explore_prob,
                                                     args.num_train)
    policy['evaluation'] = GreedyEpsilonPolicy(0.0)
    state_shape = tuple([args.num_frame] + list(args.input_shape))
    
    agent = DQNAgent(state_shape, model_input_shape,
                     q_network, proc, memory, policy,
                     args.discount, args.target_reset_interval,
                     args.num_burn_in, args.train_freq, args.batch_size,
                     args.eval_interval, args.eval_episodes, args.double_net)
    
    agent.compile(opt_adam, mean_huber_loss)
    
    #~ try:
    if args.read_weight:
        weight_read_name = os.path.join(args.read_weight)
        with open(weight_read_name, 'rb') as save:
            saved_weights, agent.memory = pickle.load(save)
        agent.q_network['online'].set_weights(saved_weights)
        agent.q_network['target'].set_weights(saved_weights)
        print 'weights & memory read from {:s}'.format(weight_read_name)
    print '########## training #############'
    agent.fit(env, args.num_train, args.max_episode_length)
    #~ except:
        #~ pass
    
    weight_save_name = os.path.join(args.output, 'online_weight.save')
    with open(weight_save_name, 'wb') as save:
        weights = q_network['online'].get_weights()
        pickle.dump((weights, agent.memory), save, protocol=pickle.HIGHEST_PROTOCOL)
    print 'weights & memory written to {:s}'.format(weight_save_name)
    

    # here is where you should start up a session,
    # create your DQN agent, create your model, etc.
    # then you can run your fit method.

if __name__ == '__main__':
    main()
