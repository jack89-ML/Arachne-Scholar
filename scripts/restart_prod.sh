#!/bin/bash
# Restart Arachne-Scholar dashboard (192.168.1.89 produzione, porta 8001)
pkill -f '[u]vicorn dashboard.app'
sleep 2
cd /tmp/arachne_prod/Arachne-Scholar
# cupy/nvidia libs per SpaCy GPU (senza questo export: gpu=False, fallback CPU)
export LD_LIBRARY_PATH="$(ls -d /tmp/arachne_gpu_venv/lib/python3.12/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH}"
nohup /tmp/arachne_gpu_venv/bin/python3 -m uvicorn dashboard.app:app --host 0.0.0.0 --port 8001 > /tmp/arachne_prod/server.log 2>&1 &
sleep 3
pgrep -af '[u]vicorn dashboard.app' || echo 'ERRORE: uvicorn non ripartito'
