set +e
L=/work/repro/log.txt; : > $L
r(){ echo "### $*" >> $L; "$@" >> $L 2>&1; echo "[rc=$?]" >> $L; }
r python reject_sampling_overflow.py orig 3531
r python reject_sampling_overflow.py orig 3532
r python reject_sampling_overflow.py orig 4000
r python reject_sampling_overflow.py fixed 4000
echo "### sanitizer orig 3532" >> $L
compute-sanitizer --tool memcheck --print-limit 3 python reject_sampling_overflow.py orig 3532 2>&1 | grep -E "Invalid|ERROR SUMMARY|wrong rows|is out of bounds|Address 0x|speculative_sampling" | head -12 >> $L
echo "### sanitizer fixed 3532" >> $L
compute-sanitizer --tool memcheck python reject_sampling_overflow.py fixed 3532 2>&1 | grep -E "ERROR SUMMARY|wrong rows" >> $L
r python moe_topk_sum_grid.py 65535
r python moe_topk_sum_grid.py 65536
cat $L
