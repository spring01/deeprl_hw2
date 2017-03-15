#!/bin/bash

python -i -u dqn_atari.py --env SpaceInvaders-v0 --output ./output --input_shape 84 84 --replay_buffer_size 0 --model_name linear --num_train 5000000 > linear_no_memory.out

python -i -u dqn_atari.py --env SpaceInvaders-v0 --output ./output --input_shape 84 84 --replay_buffer_size 100000 --model_name linear --num_train 5000000 --target_reset_interval 10000 > linear_memory.out

python -i -u dqn_atari.py --env SpaceInvaders-v0 --output ./output --input_shape 84 84 --replay_buffer_size 100000 --model_name linear --num_train 5000000 --target_reset_interval 10000 --double_net > linear_double.out

python -i -u dqn_atari.py --env SpaceInvaders-v0 --output ./output --input_shape 84 84 --replay_buffer_size 100000 --model_name dqn --num_train 5000000 --target_reset_interval 10000 > dqn.out

python -i -u dqn_atari.py --env SpaceInvaders-v0 --output ./output --input_shape 84 84 --replay_buffer_size 100000 --model_name dqn --num_train 5000000 --target_reset_interval 10000 --double_net > dqn_double.out

python -i -u dqn_atari.py --env SpaceInvaders-v0 --output ./output --input_shape 84 84 --replay_buffer_size 100000 --model_name dueling --num_train 5000000 --target_reset_interval 10000 > dqn_dueling.out
