#!/bin/bash
# round 36 on one box: setup, then for each line "KEY REPO REVISION UNITS_REPO" of ~/r36/list.txt download the
# checkpoint (at most 2 ahead), record it over ~/r36/pool.npz, delete the weights. Resumable: a KEY whose
# ~/r36/out/KEY.npz exists is skipped. A failed download leaves ~/ck/KEY.fail and the KEY is skipped. Marker ~/r36/all.done
set -x
[ -f ~/setup.done ] || bash ~/janus/tools/box/r36_setup.sh > ~/setup.log 2>&1
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p ~/r36/out ~/ck
(while read K R V U; do
   [ -f ~/r36/out/$K.npz ] && continue
   while [ $(ls ~/ck/*.ok 2>/dev/null | wc -l) -ge 2 ]; do sleep 5; done
   for try in 1 2 3; do
     ~/o/bin/hf download $R --revision $V --local-dir ~/ck/$K --include "*.safetensors" "*.bin" "*.json" < /dev/null > ~/ck/$K.dl.log 2>&1 && { touch ~/ck/$K.ok; break; }
     sleep 30
   done
   [ -f ~/ck/$K.ok ] || touch ~/ck/$K.fail
 done < ~/r36/list.txt) &
while read K R V U; do
  [ -f ~/r36/out/$K.npz ] && continue
  until [ -f ~/ck/$K.ok ] || [ -f ~/ck/$K.fail ]; do sleep 5; done
  [ -f ~/ck/$K.fail ] && continue
  ~/o/bin/python -I ~/janus/tools/r36_record.py record --model ~/ck/$K --units-repo $U --pool ~/r36/pool.npz \
    --out ~/r36/out/$K.tmp.npz --batch 16 < /dev/null > ~/r36/$K.log 2>&1 && mv ~/r36/out/$K.tmp.npz ~/r36/out/$K.npz
  rm -rf ~/ck/$K ~/ck/$K.ok
done < ~/r36/list.txt
wait
touch ~/r36/all.done
