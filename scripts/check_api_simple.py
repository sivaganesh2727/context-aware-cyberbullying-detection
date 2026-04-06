import sys, os
# ensure repository root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)
for txt in ["hi","hello","how are you","test","foo"]:
    payload = {"text": txt,
               "include_explanation": True,
               "use_ensemble": False,
               "use_advanced_context": True}
    r = client.post("/detect", json=payload)
    print(txt, r.status_code, r.json())
