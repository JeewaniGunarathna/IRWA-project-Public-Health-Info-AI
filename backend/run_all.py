# backend/run_all.py
import subprocess, os, time, signal, sys
from pathlib import Path

# Each entry = (name, command-list)
AGENTS = [
    ("retrieval",    ["uvicorn", "app.agents.retrieval_agent.main:app", "--host", "127.0.0.1", "--port", "8005"]),
    ("security",     ["uvicorn", "app.agents.security_agent.main:app",  "--host", "127.0.0.1", "--port", "8090"]),
    ("chat",         ["uvicorn", "app.agents.chat_agent.main:app",      "--host", "127.0.0.1", "--port", "8007"]),
    ("report",       ["uvicorn", "app.agents.report_generator.main:app","--host", "127.0.0.1", "--port", "8003"]),
    ("orchestrator", ["uvicorn", "orchestrator.main:app", "--host", "127.0.0.1", "--port", "8010"]),
]

procs = []

def start_all():
    env = os.environ.copy()
    backend_dir = Path(__file__).parent
    for name, cmd in AGENTS:
        print(f"▶ starting {name}: {' '.join(cmd)}")
        # no --reload to avoid double workers
        p = subprocess.Popen(cmd, cwd=str(backend_dir), env=env)
        procs.append((name, p))
        time.sleep(0.5)  # stagger starts so ports settle

def stop_all(*_):
    print("\n⏹ stopping agents…")
    for name, p in procs:
        if p.poll() is None:
            print(f"⏹ terminating {name}")
            p.terminate()
    time.sleep(2)
    for name, p in procs:
        if p.poll() is None:
            print(f"⚠ force-kill {name}")
            p.kill()

if __name__ == "__main__":
    try:
        start_all()
        print("✅ all agents launched. Press Ctrl+C to stop.")
        # keep parent alive until orchestrator exits
        for name, p in procs:
            if name == "orchestrator":
                p.wait()
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop_all()
