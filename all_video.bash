#!/bin/bash

bash video.bash linear_no_memory linear
bash video.bash linear_memory linear
bash video.bash linear_double linear
bash video.bash dqn dqn
bash video.bash dqn_double dqn
bash video.bash dqn_dueling dueling

