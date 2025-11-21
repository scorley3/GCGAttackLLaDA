#!/bin/bash
echo "Task 1: Llada generation with 100 iterations and batch size 128 WITH prefill"
SECONDS=0
python3 main.py --prompts_start 100 --prompts_end 103 --iterations 5 --k 256 --batch_size 128 --use_qwen False --seed_llada False --prefill_string True --log_path "attack_log_bash1.json" --plot_path "Loss_plot_test1_prompt"
echo "Task Complete in $SECONDS seconds"

echo "Task 2: Llada generation with 100 iterations and batch size 50 WITH prefill"
SECONDS=0
python3 main.py --prompts_start 103 --prompts_end 105 --iterations 100 --k 256 --batch_size 50 --use_qwen False --seed_llada False --prefill_string True --log_path "attack_log_bash2.json" --plot_path "Loss_plot_test2_prompt"
echo "Task Complete in $SECONDS seconds"

# echo -e "Subject: SCRIPT IS DONE\n\nTHE SCRIPT IS DONE SHELVE MEEEEE SHELVE MEEEEEEEE" | msmtp neyroud2@illinois.edu
# echo -e "Subject: AAAA FIRE FIRE FIRE HELPPPPP\n\ndid I get your attention? the only thing thats fire is the output(I hope)\nshelve that thang" | msmtp scorley3@illinois.edu
# echo -e "Subject: SCRIPT IS DONE\n\nTHE SCRIPT IS DONE SHELVE MEEEEE SHELVE MEEEEEEEE" | msmtp rubenzneyroud@gmail.com
