#!/bin/bash
echo "Task 1: Llada generation with 16 iters, 16 candidates, 16 batch size WITH prefill (efficiency test)"
SECONDS=0
python3 main.py --prompts_start 0 --prompts_end 5 --iterations 16 --k 16 --batch_size 16 --use_qwen False --seed_llada False --prefill_string True --log_path "attack_log_small_nums.json" --plot_path "Loss_plot_test_small_params_prompt"
echo "Task Complete in $SECONDS seconds"

echo "Task 2: Llada generation with 32 for each param WITH prefill"
SECONDS=0
python3 main.py --prompts_start 6 --prompts_end 10 --iterations 32 --k 32 --batch_size 32 --use_qwen False --seed_llada False --prefill_string True --log_path "attack_log_32.json" --plot_path "Loss_plot_test32_prompt"
echo "Task Complete in $SECONDS seconds"

echo "Task 3: Llada generation 32s with 64 batch size just for fun WITH prefill"
SECONDS=0
python3 main.py --prompts_start 103 --prompts_end 105 --iterations 32 --k 32 --batch_size 64 --use_qwen False --seed_llada False --prefill_string True --log_path "attack_log_vibes.json" --plot_path "Loss_plot_testvibes_prompt"
echo "Task Complete in $SECONDS seconds"

echo -e "Subject: find out if sam's scripts are fire!" | msmtp neyroud2@illinois.edu
echo -e "Subject: did this finish before the car ride is over?" | msmtp scorley3@illinois.edu
# echo -e "Subject: SCRIPT IS DONE\n\nTHE SCRIPT IS DONE SHELVE MEEEEE SHELVE MEEEEEEEE" | msmtp rubenzneyroud@gmail.com
