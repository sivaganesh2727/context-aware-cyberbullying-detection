import sys, os
# ensure repo root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)
print('health', client.get('/health').json())
r = client.post('/detect', json={'text':'hello world','include_explanation':True,'use_ensemble':False,'use_advanced_context':True})
print('detect code', r.status_code, r.text[:200])
r2 = client.post('/detect-batch', json={'texts':['hello','kill you'],'include_explanations':False,'use_ensemble':False})
print('batch code', r2.status_code, r2.text[:200])
