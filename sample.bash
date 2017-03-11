#!/bin/bash

python -i dqn_atari.py --env SpaceInvaders-v0 --output ./output --input_shape 84 84 --replay_buffer_size 10000 --model_name dqn --num_train 5000000 --target_reset_interval 5000

