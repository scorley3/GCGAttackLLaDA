#!/bin/bash
echo "Task 1: Llada generation with 100 iterations and batch size 128 WITH prefill"
SECONDS=0
python3 main.py --prompts_start 100 --prompts_end 103 --iterations 100 --k 256 --batch_size 128 --use_qwen False --seed_llada False --prefill_string True --log_path "attack_log_bash.json"
echo "Task Complete in $SECONDS seconds"

echo "Task 2: Llada generation with 100 iterations and batch size 50 WITH prefill"
SECONDS=0
python3 main.py --prompts_start 103 --prompts_end 105 --iterations 100 --k 256 --batch_size 50 --use_qwen False --seed_llada False --prefill_string True --log_path "attack_log_bash.json"
echo "Task Complete in $SECONDS seconds"

sudo shutdown -h now