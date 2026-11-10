"""Reproducible Windows setup using official distributions; Python standard library only."""
import os
from pathlib import Path
import subprocess
import time
import urllib.request
import zipfile
from start_server import reachable

ROOT=Path(__file__).resolve().parent
runtime=ROOT/'runtime'
runtime.mkdir(exist_ok=True)
if not (runtime/'ollama.exe').exists():
    archive=ROOT/'ollama-windows-amd64.zip'
    print('Downloading Ollama v0.40.16 (about 1.6 GB)...',flush=True)
    urllib.request.urlretrieve('https://github.com/ollama/ollama/releases/download/v0.40.16/ollama-windows-amd64.zip',archive)
    with zipfile.ZipFile(archive) as z:z.extractall(runtime)
if not (runtime/'cloudflared.exe').exists():
    print('Downloading official cloudflared...',flush=True)
    urllib.request.urlretrieve('https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe',runtime/'cloudflared.exe')
env=os.environ.copy()
env.update(OLLAMA_HOST='127.0.0.1:11435',OLLAMA_MODELS=str(ROOT/'models'),OLLAMA_CONTEXT_LENGTH='4096',OLLAMA_NUM_PARALLEL='1',OLLAMA_NO_CLOUD='true')
child=None
try:
    if not reachable('http://127.0.0.1:11435/api/version'):
        child=subprocess.Popen([str(runtime/'ollama.exe'),'serve'],env=env,creationflags=subprocess.CREATE_NO_WINDOW)
    for _ in range(60):
        if reachable('http://127.0.0.1:11435/api/version'):break
        time.sleep(1)
    else:raise RuntimeError('Ollama did not start.')
    subprocess.run([str(runtime/'ollama.exe'),'pull','qwen3.5:9b'],env=env,check=True)
    print('Ready. Run START_PUBLIC.cmd to create a public link.',flush=True)
finally:
    if child:
        child.terminate()
        child.wait(timeout=15)
