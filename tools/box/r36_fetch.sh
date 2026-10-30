#!/bin/bash
# copy finished round-36 recordings (non-empty KEY.npz; empty ones are skip placeholders) from the boxes to runs/r36/out
O="-i $HOME/.ssh/lambda_ed25519 -o ConnectTimeout=20"
mkdir -p ~/dev/interp/janus/runs/r36/out
for ip in "$@"; do
  rsync -a --min-size=1 --exclude '*.tmp.npz' -e "ssh $O" ubuntu@$ip:r36/out/ ~/dev/interp/janus/runs/r36/out/ &
done
wait
ls ~/dev/interp/janus/runs/r36/out | tr '\n' ' '; echo
