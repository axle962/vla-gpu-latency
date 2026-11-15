#!/usr/bin/env bash
# Stop this user's LIBERO runner shells and eval processes, matching the exact program path
# (never this script's own command line).
pids=$(ps -u "$USER" -o pid=,args= | awk '($2=="bash" && $3=="scripts/run_libero.sh") || ($2 ~ /python$/ && $3=="workloads/libero_eval.py") {print $1}')
[ -n "$pids" ] && kill $pids
sleep 5
echo "$(hostname -s): $(ps -u "$USER" -o args= | awk '$3=="scripts/run_libero.sh" || $3=="workloads/libero_eval.py"' | wc -l) left"
