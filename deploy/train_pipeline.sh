#!/usr/bin/env bash
# Overnight pipeline: wait for prepare_asym, smoke-test, then train in segments. After each segment
# (default 2,000 steps) training stops, the reference comparison (vs ElectronGirl pilot_r32 / AnticPan
# Role-Ref-KV / base) and the held-out loss gap run, and training resumes from the saved state.
# Env: GPUS (comma list, default 0,1), PORT (rendezvous port), VARIANTS (configs/train_asym_<name>.json),
#      FIRST (first run name, default v1a; "none" to skip), PY (python), NOLDPATH=1 to drop LD_LIBRARY_PATH.
set -u
cd ~/anima-ref-adapter-research
PY=${PY:-$HOME/comfy-venv/bin/python}
PREP=${PREP:-$HOME/train/asym_prep}
RUNS=${RUNS:-$HOME/train/runs}      # fast scratch is fine: every finished segment is mirrored to PERSIST
PERSIST=${PERSIST:-}                # optional persistent copy of RUNS (rsync after each segment)
LOG=~/logs/train
SEG=${SEGMENT_STEPS:-2000}
GPUS=${GPUS:-0,1}
PORT=${PORT:-29511}
NGPU=$(echo $GPUS | tr ',' '\n' | wc -l)
G0=$(echo $GPUS | cut -d, -f1)
G1=$(echo $GPUS | cut -d, -f2)
REFS="$HOME/competitors/electrongirl/showcase/ref_girl.jpg $HOME/competitors/electrongirl/showcase/ref_man.jpg"
[ "${NOLDPATH:-0}" = 1 ] && unset LD_LIBRARY_PATH
mkdir -p $LOG $RUNS ~/train/compare

while [ ! -s $PREP/index.json ] || [ -n "$(ls $PREP/*.partial 2>/dev/null)" ]; do sleep 60; done
echo "$(date -u +%T) prep ready (gpus $GPUS)"

compare() {  # name adapter
  local name=$1 a=$2 step
  step=$(basename $a .safetensors | sed 's/adapter_//')
  [ -s ~/train/compare/${name}_$step/summary.json ] && return 0
  CUDA_VISIBLE_DEVICES=$G1 $PY train/compare_refs.py --prep $PREP --ours $a --cases 16 --extra-refs $REFS \
      --out ~/train/compare/${name}_$step > $LOG/compare_${name}_$step.log 2>&1 &
  local p1=$!
  CUDA_VISIBLE_DEVICES=$G0 $PY train/eval_asym.py --prep $PREP --ours $a --out ~/train/compare/${name}_$step \
      > $LOG/eval_${name}_$step.log 2>&1 || true
  wait $p1 && echo "$(date -u +%T) compare $name $step done" || echo "$(date -u +%T) compare $name $step FAILED"
}

run_variant() {  # name config
  local name=$1 cfg=$2 out=$RUNS/$1
  mkdir -p $out
  echo "$(date -u +%T) smoke $name"
  CUDA_VISIBLE_DEVICES=$G0 $PY train/train_asym.py --prep $PREP --config $cfg --out /tmp/smoke_$name --smoke \
      > $LOG/${name}_smoke.log 2>&1 || { echo "$(date -u +%T) smoke $name FAILED"; tail -30 $LOG/${name}_smoke.log; return 1; }
  local total=$($PY -c "import json;print(json.load(open('$cfg'))['train']['steps'])")
  local done_steps=0
  while [ $done_steps -lt $total ]; do
    local last=$(ls $out/state_*.pt 2>/dev/null | tail -1) resume=""
    [ -n "$last" ] && resume="--resume $last" && done_steps=$(basename $last .pt | sed 's/state_0*//')
    [ -z "$done_steps" ] && done_steps=0
    [ $done_steps -ge $total ] && break
    local stop=$((done_steps + SEG)); [ $stop -gt $total ] && stop=$total
    echo "$(date -u +%T) train $name $done_steps -> $stop"
    touch $out/.training
    CUDA_VISIBLE_DEVICES=$GPUS $PY -m torch.distributed.run --nproc_per_node $NGPU --master_port $PORT train/train_asym.py \
        --prep $PREP --config $cfg --out $out $resume --stop-at $stop >> $LOG/${name}.log 2>&1 \
        || { rm -f $out/.training; echo "$(date -u +%T) train $name FAILED"; tail -40 $LOG/${name}.log; return 1; }
    rm -f $out/.training
    [ -n "$PERSIST" ] && mkdir -p $PERSIST && rsync -a $out $PERSIST/ && echo "$(date -u +%T) mirrored $name to $PERSIST"
    done_steps=$stop
    compare $name $out/adapter_$(printf %06d $stop).safetensors
  done
  echo "$(date -u +%T) $name finished"
}

FIRST=${FIRST:-v1a}
[ "$FIRST" != none ] && run_variant $FIRST configs/train_asym$([ $FIRST = v1a ] || echo _$FIRST).json
for v in ${VARIANTS:-}; do
  run_variant $v ${CONFIG_OVERRIDE:-configs/train_asym_$v.json}
done
echo "$(date -u +%T) pipeline done"
