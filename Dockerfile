FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir \
    --default-timeout=300 \
    --retries=10 \
    -r requirements.txt

# Patch SpikingJelly for NumPy >= 1.24
RUN python - <<'PY'
from pathlib import Path

p = Path('/opt/conda/lib/python3.11/site-packages/spikingjelly/activation_based/auto_cuda/base.py')

txt = p.read_text()

txt = txt.replace(
    'elif value.dtype == np.int:',
    'elif value.dtype == np.int_:'
)

p.write_text(txt)

print("SpikingJelly patched successfully.")
PY

# Verify patch
RUN grep -n "value.dtype ==" \
    /opt/conda/lib/python3.11/site-packages/spikingjelly/activation_based/auto_cuda/base.py

COPY . /app

RUN chmod +x run.sh

CMD ["./run.sh"]