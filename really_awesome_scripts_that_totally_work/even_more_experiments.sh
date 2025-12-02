#!/bin/bash
echo "Task 1"
SECONDS=0
python3 main.py --prompts_start 0 --prompts_end 1 --iterations 128 --k 8 --batch_size 32 --use_qwen False --seed_llada False --prefill_string True --log_path "lowk.json" --plot_path "lowk"
echo "Task Complete in $SECONDS seconds"

echo "Task 2"
SECONDS=0
python3 main.py --prompts_start 6 --prompts_end 7 --iterations 128 --k 16 --batch_size 64 --use_qwen False --seed_llada False --prefill_string True --log_path "attack_log_bs64.json" --plot_path "Loss_plot_bs64_prompt"
echo "Task Complete in $SECONDS seconds"

# echo "Task 3: Llada generation 32s with 64 batch size just for fun WITH prefill"
# SECONDS=0
# python3 main.py --prompts_start 103 --prompts_end 105 --iterations 32 --k 32 --batch_size 64 --use_qwen False --seed_llada False --prefill_string True --log_path "attack_log_vibes.json" --plot_path "Loss_plot_testvibes_prompt"
# echo "Task Complete in $SECONDS seconds"

echo -e "Subject: find out if sam's scripts are fire!" | msmtp neyroud2@illinois.edu
# echo -e "Subject: did this finish before the car ride is over?" | msmtp scorley3@illinois.edu
# echo -e "Subject: SCRIPT IS DONE\n\nTHE SCRIPT IS DONE SHELVE MEEEEE SHELVE MEEEEEEEE" | msmtp rubenzneyroud@gmail.com
