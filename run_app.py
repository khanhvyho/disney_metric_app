# run_app.py
import os, sys, time, socket, webbrowser, threading
from urllib.request import urlopen

def resource_path(rel_path: str) -> str:
    return os.path.join(getattr(sys, "_MEIPASS", os.path.dirname(__file__)), rel_path)

def find_free_port(start=8501, tries=20):
    p = start
    for _ in range(tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                p += 1
    return start

def ready(base_url: str, timeout=40):
    deadline = time.time() + timeout
    paths = ("/healthz", "/_stcore/health")
    while time.time() < deadline:
        for path in paths:
            try:
                with urlopen(base_url + path, timeout=1.2) as r:
                    if r.status == 200:
                        return True
            except Exception:
                pass
        time.sleep(0.35)
    return False

# --- paths / env ---
APP_PATH = resource_path("app.py")
SRC_PATH = resource_path("src")
BASE_DIR = os.path.dirname(APP_PATH)

if os.path.isdir(SRC_PATH): sys.path.insert(0, SRC_PATH)
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"  # avoid config conflicts
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
os.environ.setdefault("MPLCONFIGDIR", os.path.expanduser("~/.matplotlib"))

# logs
LOG_DIR = os.path.join(os.path.expanduser("~/Library/Logs"), "PartnerMetrics")
os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, "streamlit.log")
log = open(log_path, "a", buffering=1)
sys.stdout = log
sys.stderr = log
print("\n\n==== PartnerMetrics (main-thread launch) ====")
print(f"APP_PATH={APP_PATH}\nSRC_PATH={SRC_PATH}\nBASE_DIR={BASE_DIR}")

# reuse existing server if alive
STATE_DIR = os.path.join(os.path.expanduser("~"), ".partnermetrics")
os.makedirs(STATE_DIR, exist_ok=True)
PORT_FILE = os.path.join(STATE_DIR, "port")
if os.path.exists(PORT_FILE):
    try:
        existing = int(open(PORT_FILE).read().strip())
        if ready(f"http://127.0.0.1:{existing}", timeout=1):
            webbrowser.open_new_tab(f"http://localhost:{existing}")
            sys.exit(0)
    except Exception:
        pass

port = find_free_port(8501)
address = "127.0.0.1"
base = f"http://{address}:{port}"

# open the browser AFTER Streamlit is healthy (runs in background)
def opener():
    if ready(base, timeout=45):
        try: open(PORT_FILE, "w").write(str(port))
        except Exception: pass
        try: webbrowser.open_new_tab(f"http://localhost:{port}")
        except Exception: pass
    else:
        print("[launcher] timed out waiting for /healthz")

threading.Thread(target=opener, daemon=True).start()

# --- run Streamlit IN THE MAIN THREAD ---
from streamlit.web import cli as stcli
sys.argv = [
    "streamlit", "run", APP_PATH,
    "--server.address", address,
    "--server.port", str(port),
    "--server.headless", "true",   # we open the tab ourselves
    "--logger.level", "info",
]
stcli.main()   # <- stays in main thread; no signal error
