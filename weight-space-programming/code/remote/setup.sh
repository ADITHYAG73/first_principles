#!/bin/bash
# EC2 user-data for the weightspace-worker (Ubuntu 22.04 DLAMI, NVIDIA driver preinstalled).
# Installs the idle watchdog FIRST, then the Python env, model, data. Logs to /var/log/wsp-setup.log
set -euxo pipefail
exec > >(tee -a /var/log/wsp-setup.log) 2>&1
BUCKET=__BUCKET__
REGION=__REGION__
MODEL_REV=__MODEL_REV__

mkdir -p /opt/wsp/{logs,models,jobs,data/practice,data/heldout}
touch /opt/wsp/busy
chown -R ubuntu:ubuntu /opt/wsp

# ---- idle watchdog: stop the machine after 15 idle minutes; hard cap 8 h uptime -------------
cat > /usr/local/bin/wsp-watchdog.sh <<'EOF'
#!/bin/bash
B=/opt/wsp/busy
if pgrep -f "worker.py" >/dev/null || pgrep -f "wsp-setup" >/dev/null || pgrep -f "cloud-init" >/dev/null; then
  touch $B
fi
IDLE=$(( $(date +%s) - $(stat -c %Y $B) ))
UP=$(cut -d. -f1 /proc/uptime)
echo "$(date -Is) idle=${IDLE}s uptime=${UP}s" >> /var/log/wsp-watchdog.log
if [ "$IDLE" -gt 900 ] || [ "$UP" -gt 28800 ]; then
  echo "$(date -Is) watchdog: shutting down (idle=${IDLE}s uptime=${UP}s)" >> /var/log/wsp-watchdog.log
  shutdown -h now
fi
EOF
chmod +x /usr/local/bin/wsp-watchdog.sh
cat > /etc/systemd/system/wsp-watchdog.service <<'EOF'
[Unit]
Description=wsp idle watchdog
[Service]
Type=oneshot
ExecStart=/usr/local/bin/wsp-watchdog.sh
EOF
cat > /etc/systemd/system/wsp-watchdog.timer <<'EOF'
[Unit]
Description=run wsp idle watchdog every 2 minutes
[Timer]
OnBootSec=5min
OnUnitActiveSec=2min
[Install]
WantedBy=timers.target
EOF
systemctl daemon-reload
systemctl enable --now wsp-watchdog.timer

# ---- python env -----------------------------------------------------------------------------
sudo -u ubuntu -H bash -lc 'curl -LsSf https://astral.sh/uv/install.sh | sh'
sudo -u ubuntu -H bash -lc "cd /opt/wsp && ~/.local/bin/uv venv -p 3.12 .venv && \
  VIRTUAL_ENV=.venv ~/.local/bin/uv pip install 'vllm==0.32.0' 'math-verify[antlr4_13_2]' \
  flash-linear-attention boto3 pandas pyarrow huggingface_hub"
sudo -u ubuntu -H bash -lc "cd /opt/wsp && .venv/bin/python -c 'import vllm, transformers, torch; print(\"versions\", vllm.__version__, transformers.__version__, torch.__version__, torch.cuda.device_count())'"

# ---- code, data, model ----------------------------------------------------------------------
sudo -u ubuntu -H bash -lc "cd /opt/wsp && .venv/bin/python - <<'PY'
import boto3, os
s3 = boto3.client('s3', region_name='$REGION')
for prefix, dest in [('code/', '/opt/wsp/'), ('practice/', '/opt/wsp/data/practice/')]:
    for o in s3.list_objects_v2(Bucket='$BUCKET', Prefix=prefix).get('Contents', []):
        s3.download_file('$BUCKET', o['Key'], dest + o['Key'][len(prefix):])
PY"
sudo -u ubuntu -H bash -lc "cd /opt/wsp && .venv/bin/hf download Qwen/Qwen3.5-0.8B --revision $MODEL_REV --local-dir models/original"
sudo -u ubuntu -H bash -lc "cd /opt/wsp && .venv/bin/python prepare_heldout.py 32"
# weights into S3 (the run's canonical copy), with checksums
sudo -u ubuntu -H bash -lc "cd /opt/wsp/models/original && sha256sum *.safetensors > SHA256SUMS && cd /opt/wsp && .venv/bin/python - <<'PY'
import boto3, pathlib
s3 = boto3.client('s3', region_name='$REGION')
for p in pathlib.Path('/opt/wsp/models/original').iterdir():
    if p.is_file():
        s3.upload_file(str(p), '$BUCKET', 'model/original/' + p.name)
PY"
touch /opt/wsp/busy
echo "SETUP_DONE $(date -Is)" > /opt/wsp/setup_done
sudo -u ubuntu -H bash -lc "cd /opt/wsp && .venv/bin/python -c \"import boto3; boto3.client('s3', region_name='$REGION').upload_file('/var/log/wsp-setup.log', '$BUCKET', 'status/setup.log'); boto3.client('s3', region_name='$REGION').upload_file('/opt/wsp/setup_done', '$BUCKET', 'status/setup_done')\""
