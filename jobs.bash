#!/bin/bash

python -u dqn_atari.py --env SpaceInvaders-v0 --output ./output --input_shape 84 84 --replay_buffer_size 0 --model_name linear --num_train 500000 | tee linear_no_memory.out

python -u dqn_atari.py --env SpaceInvaders-v0 --output ./output --input_shape 84 84 --replay_buffer_size 100000 --model_name linear --num_train 500000 --target_reset_interval 10000 | tee linear_memory.out

python -u dqn_atari.py --env SpaceInvaders-v0 --output ./output --input_shape 84 84 --replay_buffer_size 100000 --model_name linear --num_train 500000 --target_reset_interval 10000 --double_net True | tee linear_double.out

python -u dqn_atari.py --env SpaceInvaders-v0 --output ./output --input_shape 84 84 --replay_buffer_size 100000 --model_name dqn --num_train 500000 --target_reset_interval 10000 | tee dqn.out

python -u dqn_atari.py --env SpaceInvaders-v0 --output ./output --input_shape 84 84 --replay_buffer_size 100000 --model_name dqn --num_train 500000 --target_reset_interval 10000 --double_net True | tee dqn_double.out

python -u dqn_atari.py --env SpaceInvaders-v0 --output ./output --input_shape 84 84 --replay_buffer_size 100000 --model_name dueling --double_net True --num_train 500000 --target_reset_interval 10000 | tee dqn_dueling.out
