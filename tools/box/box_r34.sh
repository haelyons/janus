#!/bin/bash
# round 34 on the box: setup, record the stimuli, analyse, then Pile cards for the selected neurons. Marker ~/r34.done
set -x
bash ~/olmo_setup.sh > ~/setup.log 2>&1
cd ~/janus
M=~/models/OLMo-2-1124-7B
~/o/bin/python -I tools/r34_record.py --model $M --stimuli runs/r34/stimuli.json --out ~/r34 || exit 1
for R in suffix text; do ~/o/bin/python -I tools/r34_analyze.py --record ~/r34 --target postponed --readout $R --out ~/r34/analysis_$R.json > ~/r34/table_$R.txt || exit 1; done
~/o/bin/python -I - <<'PY' || exit 1
import json, sys
sys.path.insert(0, 'bank')
import olmo_units as ou
a = json.load(open('/home/ubuntu/r34/analysis_suffix.json'))
L = a['best_layer']
units = [{'name': f'olmo:L{L}:n{s["unit"]}', 'kind': 'neuron', 'layer': L, 'index': int(s['unit']), 'w': None} for s in a['selected']]
ou.extract('/home/ubuntu/models/OLMo-2-1124-7B', '/home/ubuntu/pile10k.parquet', units, '/home/ubuntu/r34acts', batch=16)
ou.sim_items.__defaults__ = (240,)
ou.build('/home/ubuntu/pile10k.parquet', '/home/ubuntu/r34acts', select=8)
PY
touch ~/r34.done
