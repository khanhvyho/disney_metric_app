# run_app.py
import os
from streamlit.web import bootstrap

HERE = os.path.dirname(os.path.abspath(__file__))
APP_PATH = os.path.join(HERE, "app.py")

# Optional: fix port if 8501 is blocked
# os.environ["STREAMLIT_SERVER_PORT"] = "8501"
# os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

bootstrap.run(APP_PATH, "", [], {})
