#!/bin/bash
# setup, then the round-33 probe server on 127.0.0.1:8765
bash ~/olmo_setup.sh > ~/setup.log 2>&1
cd ~/janus && exec ~/o/bin/python -I bank/olmo_units.py serve --model ~/models/OLMo-2-1124-7B --units ~/acts/units.json --port 8765
