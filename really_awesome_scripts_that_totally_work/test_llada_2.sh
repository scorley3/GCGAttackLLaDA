echo "Task 2: Llada generation with 100 iterations and batch size 50 WITH prefill"
SECONDS=0
python3 main.py --prompts_start 104 --prompts_end 105 --iterations 100 --k 256 --batch_size 50 --use_qwen False --seed_llada False --prefill_string True --log_path "attack_log_bash20_2.json" --plot_path "Loss_plot_test20_2_prompt"
echo "Task Complete in $SECONDS seconds"

echo "Task 3: Llada generation 50s just for fun WITH prefill"
SECONDS=0
python3 main.py --prompts_start 103 --prompts_end 105 --iterations 50 --k 50 --batch_size 50 --use_qwen False --seed_llada False --prefill_string True --log_path "attack_log_bash30.json" --plot_path "Loss_plot_test30_prompt"
echo "Task Complete in $SECONDS seconds"

echo -e "Subject: SCRIPT IS DONE\n\nTHE SCRIPT IS DONE SHELVE MEEEEE SHELVE MEEEEEEEE" | msmtp neyroud2@illinois.edu
echo -e "Subject: AAAA FIRE FIRE FIRE HELPPPPP\n\ndid I get your attention? the only thing thats fire is the output(I hope)\nshelve that thang" | msmtp scorley3@illinois.edu
# echo -e "Subject: SCRIPT IS DONE\n\nTHE SCRIPT IS DONE SHELVE MEEEEE SHELVE MEEEEEEEE" | msmtp rubenzneyroud@gmail.com
