#!/bin/bash
echo "Task 1: Llada generation with 100 iterations and small mc_num WITH prefill"
SECONDS=0
python3 main.py --prompts_start 0 --prompts_end 100 --iterations 100 --k 256 --batch_size 128 --use_qwen False --seed_llada False --prefill_string True --log_path "single_mc_log.json" --plot_path "single_mc_Loss_plot_prompt" --mc_num 16
echo "Task Complete in $SECONDS seconds"
echo "Task 2: Llada generation with low iterations and mid sized mc and low k WITH prefill"
SECONDS=0
python3 main.py --prompts_start 0 --prompts_end 100 --iterations 20 --k 10 --batch_size 128 --use_qwen False --seed_llada False --prefill_string True --log_path "single_mc_log.json" --plot_path "single_mc_Loss_plot_prompt" --mc_num 16
echo "Task Complete in $SECONDS seconds"

echo -e "Subject:mc_num and k script done \n\nI WROTE A SECOND SCRIPT INCASE THE FIRST FINISHED SHELVE MEEEEE" | msmtp neyroud2@illinois.edu
echo -e "Subject: mc_num and k script done\n\nI WROTE A SECOND SCRIPT INCASE THE FIRST FINISHED SHELVE MEEEEE" | msmtp scorley3@illinois.edu
# echo -e "Subject: SCRIPT IS DONE\n\nTHE SCRIPT IS DONE SHELVE MEEEEE SHELVE MEEEEEEEE" | msmtp rubenzneyroud@gmail.com
