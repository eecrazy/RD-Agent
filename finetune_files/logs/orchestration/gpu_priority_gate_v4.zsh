#!/usr/bin/env zsh

# Keep all GPUs busy while giving search/SFT work priority over opportunistic
# snapshot validation.  When primary work is waiting, evict only the number of
# validation jobs needed to satisfy that demand, choosing the youngest jobs so
# that already-completed validation compute is preserved.

setopt NO_NOMATCH

ROOT=/data/github/RD-Agent
LOCK_DIR=$ROOT/finetune_files/gpu_leases/locks
STATE_DIR=$ROOT/finetune_files/logs/orchestration
LOCK_PATH=$STATE_DIR/gpu_priority_gate_v4.lock
PID_PATH=$STATE_DIR/gpu_priority_gate_v4.pid
LOG_PATH=${GPU_PRIORITY_LOG_PATH:-$STATE_DIR/gpu_priority_gate_v4.log}
POLL_SECONDS=${GPU_PRIORITY_POLL_SECONDS:-5}
GRACE_TICKS=${GPU_PRIORITY_GRACE_TICKS:-3}
DRY_RUN=${GPU_PRIORITY_DRY_RUN:-0}
MAX_TICKS=${GPU_PRIORITY_MAX_TICKS:-0}

cd $ROOT || exit 1

exec {gate_lock_fd}>$LOCK_PATH
if ! flock -n $gate_lock_fd; then
  print -u2 -- "another gpu_priority_gate_v4 instance owns $LOCK_PATH"
  exit 1
fi
print -r -- $$ >| $PID_PATH

log_line() {
  print -r -- "$(date '+%F %T') GPU_PRIORITY_V4 $*" >> $LOG_PATH
}

resume_validation() {
  local p stat cwd pgid
  typeset -A resumed_pgids
  resumed_pgids=()

  for p in $(pgrep -f '[r]eproduction/ft_agent/run_validation_sweep.py' 2>/dev/null || true); do
    [[ $p != $$ && -d /proc/$p ]] || continue
    stat=$(ps -o stat= -p $p 2>/dev/null | xargs)
    if [[ $stat == *T* ]]; then
      (( DRY_RUN )) || kill -CONT $p 2>/dev/null || true
    fi
  done

  for p in $(pgrep -f '/[o]pencompass/bin/opencompass' 2>/dev/null || true); do
    [[ -d /proc/$p ]] || continue
    cwd=$(readlink /proc/$p/cwd 2>/dev/null || true)
    [[ $cwd == *'/validation_sweep/'* ]] || continue
    pgid=$(ps -o pgid= -p $p 2>/dev/null | xargs)
    [[ -n $pgid && -z $resumed_pgids[$pgid] ]] || continue
    resumed_pgids[$pgid]=1
    stat=$(ps -o stat= -p $p 2>/dev/null | xargs)
    if [[ $stat == *T* ]]; then
      (( DRY_RUN )) || kill -CONT -- -$pgid 2>/dev/null || true
    fi
  done
}

cleaned=0
cleanup() {
  (( cleaned )) && return
  cleaned=1
  if (( ! DRY_RUN )); then
    resume_validation
    rm -f $PID_PATH
  fi
  log_line "exit pid=$$ dry_run=$DRY_RUN"
}
on_signal() {
  cleanup
  trap - EXIT INT TERM HUP
  exit 0
}
trap cleanup EXIT
trap on_signal INT TERM HUP

self=$$
last_state=''
last_heartbeat=0
wait_ticks=0
ticks=0

log_line "start pid=$$ dry_run=$DRY_RUN poll_seconds=$POLL_SECONDS grace_ticks=$GRACE_TICKS"

while true; do
  now=$(date +%s)
  typeset -A owner_pgids primary_pgids handled_pgids
  typeset -a snapshot_rows victims
  owner_pgids=()
  primary_pgids=()
  handled_pgids=()
  snapshot_rows=()
  victims=()
  primary=0
  snapshot=0
  total=0

  while read lease_pid lease_path; do
    [[ -n $lease_pid && -d /proc/$lease_pid ]] || continue
    case $lease_path in
      $LOCK_DIR/gpu-[0-7].lock) ;;
      *) continue ;;
    esac
    pgid=$(ps -o pgid= -p $lease_pid 2>/dev/null | xargs)
    [[ -n $pgid ]] || continue
    owner_pgids[$pgid]=1
    (( total++ ))
    cwd=$(readlink /proc/$lease_pid/cwd 2>/dev/null || true)
    if [[ $cwd == *'/validation_sweep/'* ]]; then
      (( snapshot++ ))
      etimes=$(ps -o etimes= -p $lease_pid 2>/dev/null | xargs)
      [[ $etimes == <-> ]] || etimes=0
      gpu=${lease_path:t:r}
      gpu=${gpu#gpu-}
      snapshot_rows+=("$(printf '%012d:%s:%s:%s' $etimes $pgid $lease_pid $gpu)")
    else
      (( primary++ ))
    fi
  done < <(lslocks -n -o PID,PATH 2>/dev/null)

  for p in $(pgrep -f '/[l]lamafactory-cli train|/[o]pencompass/bin/opencompass' 2>/dev/null || true); do
    [[ $p != $self && -d /proc/$p ]] || continue
    cwd=$(readlink /proc/$p/cwd 2>/dev/null || true)
    [[ $cwd == *'/validation_sweep/'* ]] && continue
    pgid=$(ps -o pgid= -p $p 2>/dev/null | xargs)
    [[ -n $pgid ]] && primary_pgids[$pgid]=1
  done

  high_wait=0
  for pgid in ${(k)primary_pgids}; do
    [[ -n $owner_pgids[$pgid] ]] || (( high_wait++ ))
  done

  free_slots=$(( 8 - total ))
  (( free_slots < 0 )) && free_slots=0
  needed=$(( high_wait - free_slots ))
  (( needed < 0 )) && needed=0
  (( needed > snapshot )) && needed=$snapshot
  changed=0

  if (( high_wait > 0 )); then
    (( wait_ticks++ ))
    mode=drain

    # Freeze validation launchers so a snapshot cannot race a waiting primary
    # process for a newly released lease.
    for p in $(pgrep -f '[r]eproduction/ft_agent/run_validation_sweep.py' 2>/dev/null || true); do
      [[ $p != $self && -d /proc/$p ]] || continue
      stat=$(ps -o stat= -p $p 2>/dev/null | xargs)
      if [[ $stat != *T* ]]; then
        (( DRY_RUN )) || kill -STOP $p 2>/dev/null || true
        (( changed++ ))
      fi
    done

    # Also freeze already-spawned validation process groups that do not yet own
    # a lease.  Leased groups not selected below continue making progress.
    for p in $(pgrep -f '/[o]pencompass/bin/opencompass' 2>/dev/null || true); do
      [[ -d /proc/$p ]] || continue
      cwd=$(readlink /proc/$p/cwd 2>/dev/null || true)
      [[ $cwd == *'/validation_sweep/'* ]] || continue
      pgid=$(ps -o pgid= -p $p 2>/dev/null | xargs)
      [[ -n $pgid && -z $handled_pgids[$pgid] ]] || continue
      handled_pgids[$pgid]=1
      [[ -n $owner_pgids[$pgid] ]] && continue
      stat=$(ps -o stat= -p $p 2>/dev/null | xargs)
      if [[ $stat != *T* ]]; then
        (( DRY_RUN )) || kill -STOP -- -$pgid 2>/dev/null || true
        (( changed++ ))
      fi
    done

    if (( wait_ticks >= GRACE_TICKS && needed > 0 )); then
      killed=0
      # Rows are zero-padded by elapsed time, so lexical order chooses the
      # youngest validation jobs first and minimizes discarded GPU-seconds.
      for row in ${(@on)snapshot_rows}; do
        (( killed < needed )) || break
        IFS=: read etimes pgid lease_pid gpu <<< $row
        [[ -n $pgid && -n $owner_pgids[$pgid] ]] || continue
        victims+=("gpu${gpu}:pg${pgid}:${etimes}s")
        if (( ! DRY_RUN )); then
          kill -TERM -- -$pgid 2>/dev/null || true
          kill -CONT -- -$pgid 2>/dev/null || true
        fi
        (( killed++ ))
        (( changed++ ))
      done
    fi
  else
    wait_ticks=0
    mode=fill
    resume_validation
  fi

  victim_text=${(j:,:)victims}
  [[ -n $victim_text ]] || victim_text=none
  state="$mode primary=$primary snapshot=$snapshot total=$total free=$free_slots high_wait=$high_wait needed=$needed wait_ticks=$wait_ticks victims=$victim_text"
  if [[ $state != $last_state ]] || (( changed > 0 )) || (( now - last_heartbeat >= 30 )); then
    gpu=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null |
      awk -F, '{gsub(/ /,""); printf "%s:%s%%/%sMiB;",$1,$2,$3}')
    log_line "$state changed=$changed gpu[$gpu]"
    last_state=$state
    last_heartbeat=$now
  fi

  (( ticks++ ))
  if (( MAX_TICKS > 0 && ticks >= MAX_TICKS )); then
    break
  fi
  sleep $POLL_SECONDS
done
