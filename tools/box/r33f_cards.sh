#!/bin/bash
# r33f_cards.sh CARD... : each card complete (arms, then simulator) before the next; reruns skip saved sessions
cd /home/hal/dev/interp/janus
SV=${SV:-http://127.0.0.1:18765}
for C in "$@"; do
  case $C in U-733|U-421|O-290|O-158) N=3;; *) N=2;; esac
  for A in A0,F2_pool,P1_active Y1_yoked; do
    timeout 4000 .venv/bin/python -I tools/run_text_arms.py --arms $A --cards $C --samples $N --workers 3 --server $SV 2>&1 | grep -v "^skip"
  done
  timeout 2000 .venv/bin/python -I tools/run_text_arms.py --arms A0,F2_pool,P1_active,Y1_yoked --cards $C --sim --workers 4 --server $SV 2>&1 | grep -v "^skip"
  echo "== $C complete"
done
