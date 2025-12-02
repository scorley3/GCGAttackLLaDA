#!/bin/bash
# echo "Task 1: Hopefully working with 128 mc num and 128 batch size"
# SECONDS=0
# python3 main.py --prompts_start 0 --prompts_end 20 --iterations 3 --k 16 --batch_size 128 --mc_num 128 --use_qwen False --seed_llada False --prefill_string True --change_prefix True --only True --log_path "hopefully_working_128bs.json" --plot_path "hopefully_working_128bs"
# echo "Task Complete in $SECONDS seconds"
echo "Task 2"
SECONDS=0
python3 main.py --prompts_start 300 --prompts_end 320 --iterations 3 --k 2 --batch_size 25 --mc_num 64 --use_qwen False --seed_llada False --prefill_string True --change_prefix True --only True --log_path "beepboop.json" --plot_path "loss_plots/beep_boop"
echo "Task Complete in $SECONDS seconds"

# echo "Task 2"
# SECONDS=0
# python3 main.py --prompts_start 1 --prompts_end 20 --iterations 8 --k 16 --batch_size 128 --mc_num 128 --use_qwen False --seed_llada False --prefill_string True --change_prefix True --only True --log_path "hopefully_working_128bs_8iter1.json" --plot_path "hopefully_working_128bs_8iter1"
# echo "Task Complete in $SECONDS seconds"

# echo "Task 3"
# SECONDS=0
# python3 main.py --prompts_start 0 --prompts_end 20 --iterations 8 --k 8 --batch_size 16 --mc_num 128 --use_qwen False --seed_llada False --prefill_string True --change_prefix True --only True --log_path "hopefully_working_lowerparams.json" --plot_path "hopefully_working_lowerparams"
# echo "Task Complete in $SECONDS seconds"

#echo -e "Subject: tasks completed" | msmtp neyroud2@illinois.edu
echo -e "Subject: tasks completed" | msmtp scorley3@illinois.edu