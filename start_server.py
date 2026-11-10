"""Start local inference, web app and optional public tunnel. Stop only owned processes."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
def reachable(url):
    try:
        with urllib.request.urlopen(url,timeout=2) as r:
            return r.status == 200
    except Exception:
        return False

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--public',action='store_true')
    args=parser.parse_args()
    env=os.environ.copy()
    env.update(OLLAMA_HOST='127.0.0.1:11435',OLLAMA_MODELS=str(ROOT/'models'),
               OLLAMA_CONTEXT_LENGTH='4096',OLLAMA_NUM_PARALLEL='1',OLLAMA_MAX_LOADED_MODELS='1',OLLAMA_NO_CLOUD='true')
    children=[]
    handles=[]
    def spawn(argv,name):
        log=open(ROOT/(name+'.log'),'w',encoding='utf-8')
        handles.append(log)
        child=subprocess.Popen(argv,cwd=str(ROOT),env=env,stdout=log,stderr=log,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        children.append(child)
        return child
    try:
        if not reachable('http://127.0.0.1:11435/api/version'):
            spawn([str(ROOT/'runtime'/'ollama.exe'),'serve'],'ollama')
        for _ in range(60):
            if reachable('http://127.0.0.1:11435/api/version'):break
            time.sleep(1)
        else:raise RuntimeError('Ollama failed to start. See ollama.log.')
        if not reachable('http://127.0.0.1:8765/api/health'):
            spawn([sys.executable,str(ROOT/'app.py')],'web')
        for _ in range(30):
            if reachable('http://127.0.0.1:8765/api/health'):break
            time.sleep(1)
        else:raise RuntimeError('Web app failed to start. See web.log.')
        print('LOCAL: http://127.0.0.1:8765',flush=True)
        if args.public:
            tunnel=spawn([str(ROOT/'runtime'/'cloudflared.exe'),'tunnel','--url','http://127.0.0.1:8765','--protocol','http2','--no-autoupdate'],'tunnel')
            for _ in range(90):
                log=(ROOT/'tunnel.log').read_text(encoding='utf-8',errors='replace')
                match=re.search(r'https://[a-z0-9-]+\.trycloudflare\.com',log)
                if match:
                    url=match.group()
                    (ROOT/'PUBLIC_URL.txt').write_text(url+'\n',encoding='utf-8')
                    print('PUBLIC: '+url,flush=True)
                    break
                if tunnel.poll() is not None:raise RuntimeError('Tunnel failed. See tunnel.log.')
                time.sleep(1)
            else:raise RuntimeError('Tunnel address not received. See tunnel.log.')
        if os.name=='nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        print('Keep this process running during evaluation. Ctrl+C stops owned services.',flush=True)
        while True:
            if any(c.poll() is not None for c in children):
                raise RuntimeError('A server process stopped. Check log files and restart.')
            time.sleep(2)
    except KeyboardInterrupt:
        print('Stopping...')
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill()
        for handle in handles:handle.close()
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':main()
