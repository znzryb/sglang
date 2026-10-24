set +e
L=/work/repro/log_aot.txt; : > $L
r(){ echo "### $*" >> $L; timeout 900 "$@" >> $L 2>&1; echo "[rc=$?]" >> $L; }
r python apply_shuffle_mul_sum_overflow.py 37440
r python apply_shuffle_mul_sum_overflow.py 37500
echo "### sanitizer apply_shuffle_mul_sum 37500" >> $L
compute-sanitizer --tool memcheck --print-limit 2 python apply_shuffle_mul_sum_overflow.py 37500 2>&1 | grep -E "Invalid|out of bounds|ERROR SUMMARY|wrong rows|at .*apply|Host Frame.*apply" | head -8 >> $L
r python moe_sum_reduce_gridy.py 524280 bf16
r python moe_sum_reduce_gridy.py 524296 bf16
r python moe_sum_reduce_gridy.py 262140 fp16
r python moe_sum_reduce_gridy.py 262152 fp16
cat $L
