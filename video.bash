#!/bin/bash

# model_full=dqn_double, model=dqn
model_full=$1
model=$2

output=./final_res/${model_full}/cherry_pick_0_3.out
python -u dqn_atari.py --env SpaceInvaders-v0 --output ./output --input_shape 84 84 --replay_buffer_size 0 --num_train 1 --target_reset_interval 10000 --double_net True --do_render True --eval_episodes 100 --model_name ${model} --make_video_cherry ./output/${model_full}_0_3/video > $output
max_video_src=`grep 'max reward video' $output | sed -e "s/.*: //"`
median_video_src=`grep 'median reward video' $output | sed -e "s/.*: //"`
cp -r $max_video_src ./final_res/${model_full}/max_video_0_3
cp -r $median_video_src ./final_res/${model_full}/median_video_0_3

output=./final_res/${model_full}/cherry_pick_1_3.out
python -u dqn_atari.py --env SpaceInvaders-v0 --output ./output --input_shape 84 84 --replay_buffer_size 0 --num_train 1 --target_reset_interval 10000 --double_net True --read_weight ./final_res/${model}/online_weight_600000.save --do_render True --eval_episodes 100 --model_name ${model} --make_video_cherry ./output/${model_full}_1_3/video > $output
max_video_src=`grep 'max reward video' $output | sed -e "s/.*: //"`
median_video_src=`grep 'median reward video' $output | sed -e "s/.*: //"`
cp -r $max_video_src ./final_res/${model_full}/max_video_1_3
cp -r $median_video_src ./final_res/${model_full}/median_video_1_3

output=./final_res/${model_full}/cherry_pick_2_3.out
python -u dqn_atari.py --env SpaceInvaders-v0 --output ./output --input_shape 84 84 --replay_buffer_size 0 --num_train 1 --target_reset_interval 10000 --double_net True --read_weight ./final_res/${model}/online_weight_1300000.save --do_render True --eval_episodes 100 --model_name ${model} --make_video_cherry ./output/${model_full}_2_3/video > $output
max_video_src=`grep 'max reward video' $output | sed -e "s/.*: //"`
median_video_src=`grep 'median reward video' $output | sed -e "s/.*: //"`
cp -r $max_video_src ./final_res/${model_full}/max_video_2_3
cp -r $median_video_src ./final_res/${model_full}/median_video_2_3

output=./final_res/${model_full}/cherry_pick_3_3.out
python -u dqn_atari.py --env SpaceInvaders-v0 --output ./output --input_shape 84 84 --replay_buffer_size 0 --num_train 1 --target_reset_interval 10000 --double_net True --read_weight ./final_res/${model}/online_weight.save --do_render True --eval_episodes 100 --model_name ${model} --make_video_cherry ./output/${model_full}_3_3/video > $output
max_video_src=`grep 'max reward video' $output | sed -e "s/.*: //"`
median_video_src=`grep 'median reward video' $output | sed -e "s/.*: //"`
cp -r $max_video_src ./final_res/${model_full}/max_video_3_3
cp -r $median_video_src ./final_res/${model_full}/median_video_3_3

