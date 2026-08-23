# run_app.py
"""
Cartpilot-ai Master Application Launcher (100% Pure Python)
Launches both the FastAPI REST Backend (port 8000) and the Streamlit Web UI (port 8501)
together in one command: python run_app.py
"""

import subprocess
import sys
import time
import os

# Suppress HuggingFace progress bars and warnings in sub-processes
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

project_root = os.path.dirname(os.path.abspath(__file__))
python_exe = sys.executable

print("=" * 70)
print("  STARTING VOICE COMMAND SHOPPING ASSISTANT")
print("=" * 70)

# Step 1: Launch FastAPI Backend Server on Port 8000
print("\n[1/2] Launching FastAPI REST Backend (http://127.0.0.1:8000)...")
fastapi_proc = subprocess.Popen(
    [python_exe, "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8000"],
    cwd=project_root
)

time.sleep(3)

# Step 2: Launch Streamlit Web UI on Port 8501
print("\n[2/2] Launching Streamlit Web UI (http://127.0.0.1:8501)...")
streamlit_proc = subprocess.Popen(
    [python_exe, "-m", "streamlit", "run", "app.py"],
    cwd=project_root
)

print("\n" + "=" * 70)
print("  VOICE COMMAND SHOPPING ASSISTANT RUNNING!")
print("  - Streamlit Web UI: http://127.0.0.1:8501")
print("  - Swagger API Docs:  http://127.0.0.1:8000/docs")
print("=" * 70 + "\n")

try:
    streamlit_proc.wait()
except KeyboardInterrupt:
    print("\nReceived shutdown signal...")
finally:
    print("Shutting down CartPilot AI servers...")
    for proc in (fastapi_proc, streamlit_proc):
        if proc and proc.poll() is None:
            proc.terminate()
    print("Shutdown complete.")

