# run_app.py
"""
FoodieBot Master Application Launcher (100% Pure Python)
Launches both the FastAPI REST Backend (port 8000) and the Streamlit Web UI (port 8501)
together in one command: python run_app.py
"""

import subprocess
import sys
import time
import os

project_root = os.path.dirname(os.path.abspath(__file__))
python_exe = sys.executable

print("=" * 70)
print("  STARTING FOODIEBOT FULL-STACK SYSTEM (FASTAPI + STREAMLIT)")
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
print("  FOODIEBOT SYSTEM RUNNING!")
print("  - Streamlit UI:  http://127.0.0.1:8501")
print("  - Swagger API:   http://127.0.0.1:8000/docs")
print("=" * 70 + "\n")

try:
    streamlit_proc.wait()
except KeyboardInterrupt:
    print("\nShutting down FoodieBot servers...")
    fastapi_proc.terminate()
    streamlit_proc.terminate()
    print("Shutdown complete.")
