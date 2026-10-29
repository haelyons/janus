#!/bin/bash
# on-box setup for round 36: torch cu128, transformers, hf_transfer. Marker ~/setup.done.
set -x
cd ~
python3 -m venv ~/o && ~/o/bin/pip install -q -U pip
~/o/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu128
~/o/bin/pip install -q transformers==4.57.1 accelerate pyarrow numpy "huggingface_hub[hf_transfer]"
~/o/bin/python -c "import torch, transformers; print('torch', torch.__version__, 'tf', transformers.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory/2**30)"
touch ~/setup.done
