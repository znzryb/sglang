# Verify DeepSeek AWQ startup (default config) + chat + GSM8K. usage: awq_verify.sh [num_examples]
N=${1:-100}; M=TechxGenus/DeepSeek-V2-Lite-Chat-AWQ; H="python3 /work/repro/httpreq.py"; LOG=/tmp/srv.log
JC=/root/.cache/sgl/sglang-jit
cleanup(){ [ -n "$PID" ] && kill $PID 2>/dev/null; sleep 2; pkill -9 -f sglang.launch_server 2>/dev/null; mkdir -p $JC && cp -a /root/.cache/sglang/jit/. $JC/ 2>/dev/null; sync; echo "[stage] cleanup done (jit cache saved: $(ls $JC/sm89 2>/dev/null | wc -l) kernels)"; }
trap cleanup EXIT
echo "[stage] import-check: $(grep -n 'kernel_api_logging' /sgl/python/sglang/srt/models/deepseek_common/utils.py | head -1)"
mkdir -p /root/.cache/sglang/jit && cp -a $JC/. /root/.cache/sglang/jit/ 2>/dev/null; echo "[stage] jit cache restored: $(ls /root/.cache/sglang/jit/sm89 2>/dev/null | wc -l) kernels, nproc=$(nproc)"
echo "[stage] download"; timeout 600 hf download $M --quiet >/dev/null || { echo "[FAIL] download"; exit 1; }
echo "[stage] launch"
python3 -m sglang.launch_server --model-path $M --trust-remote-code --host 127.0.0.1 --port 30000 > $LOG 2>&1 &
PID=$!; last_size=0; still=0
for i in $(seq 1 180); do   # <= 15 min to become healthy
  $H GET http://127.0.0.1:30000/health >/dev/null 2>&1 && break
  kill -0 $PID 2>/dev/null || { echo "[FAIL] server exited"; grep -nE "Error|Traceback" $LOG | tail -20; exit 1; }
  size=$(stat -c %s $LOG); if [ "$size" = "$last_size" ]; then still=$((still+5)); else still=0; last_size=$size; fi
  [ $((i % 6)) -eq 0 ] && echo "[wait] $((i*5))s no-new-log=${still}s: $(tail -c 400 $LOG | tr '\r\n' '  ' | tail -c 150)"
  sleep 5
done
$H GET http://127.0.0.1:30000/health >/dev/null 2>&1 || { echo "[FAIL] not healthy in 15 min"; exit 1; }
echo "[stage] server up"; grep -E "Load weight end|Capture .* end" $LOG | cut -c1-200
echo "[stage] chat"
timeout 120 $H POST http://127.0.0.1:30000/v1/chat/completions "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17*23? Answer with the number only.\"}],\"max_tokens\":32,\"temperature\":0}" | head -c 800; echo
echo "[stage] gsm8k n=$N"
timeout 360 /opt/sgleval/bin/sgl-eval run gsm8k --base-url http://127.0.0.1:30000/v1 --num-examples $N 2>&1 | grep -E "Accuracy|Invalid|Latency|rror" | head -6
echo "[stage] done"
