(
set -euo pipefail
cd /home/project/vllm_fastllm
umask 077

exec 9>test/distributed/results/.pd_v62_replay_launch.lock
flock -n 9 || { echo '已有最小复验启动操作，未重复提交。'; exit 1; }
shopt -s nullglob
pd_existing=(test/distributed/results/*_pd_v62_replay_*)
if ((${#pd_existing[@]})); then
    printf '已有复验目录，请先查看结果，未重复提交：\n%s\n' "${pd_existing[@]}"
    exit 0
fi
flock -n test/distributed/results/.pd_tp2_tp1_pull.lock true || {
    echo '已有 PD 测试持有互斥锁，未提交新请求。'; exit 1;
}

pd_python=/venv/main/bin/python3
pd_source=project/pd_test_commands/PD-V62_replay_minimal.py
pd_generation=test/distributed/results/20261128T110401Z_pd_generation_d98fff95
test -x "$pd_python"
test -f "$pd_source"
test -f build-process-tp/_pd_core.so
test -f "$pd_generation/summary.json"
pd_run="$PWD/test/distributed/results/$(date -u +%Y%m%dT%H%M%SZ)_pd_v62_replay_$("$pd_python" -c 'import secrets; print(secrets.token_hex(4))')"
mkdir "$pd_run"
cp "$pd_source" "$pd_run/run.py"
cp "$pd_generation/summary.json" "$pd_run/source.generation.summary.json"

cat > "$pd_run/command.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
pd_directory=$(cd -- "$(dirname -- "$0")" && pwd)
trap 'pd_code=$?; echo "$pd_code" > "$pd_directory/exit_code"' EXIT
cd /home/project/vllm_fastllm
export FT_PD_CORE_LIBRARY="$PWD/build-process-tp/_pd_core.so"
export PYTHONPATH="$PWD/tools${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$PWD/build-process-tp${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1
/venv/main/bin/python3 -u "$pd_directory/run.py"
SH
nohup bash "$pd_run/command.sh" > "$pd_run/stdout.log" 2> "$pd_run/stderr.log" < /dev/null 9>&- &
echo "$!" > "$pd_run/supervisor.pid"
echo "本次目录：$pd_run"
echo '已后台提交；请根据 exit_code 和 summary.md 判断结果，不要再次启动。'
)
