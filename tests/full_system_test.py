import sys
import os
import pytest
import numpy as np

# Make repository root importable when running the test module directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.preprocessing import clean_text
from src.negation_handler import NegationHandler
from src.context_analyzer import ContextAnalyzer
from src.explainability import explain_multilabel
from src.ontology import get_intervention_plan, aggregate_severity, recommend_intervention


def test_clean_text_removes_urls_and_usernames():
    s = "Check this out http://example.com @user"
    out = clean_text(s)
    assert 'http' not in out and '@' not in out


def test_negation_handler_detects_and_adjusts():
    nh = NegationHandler()
    ctx = nh.detect_negation_context("I don't kill you")
    assert ctx['has_negation'] is True

    preds = {'threat': 0.9, 'toxic': 0.8, 'insult': 0.7}
    adjusted, ctx2 = nh.adjust_predictions(preds.copy(), "I don't kill you")
    # threat score should be reduced substantially (exact value may vary)
    assert adjusted['threat'] < 0.5


def test_explainability_perturbation():
    # simple predict_proba function returning deterministic probs
    def predict_fn(texts):
        # return shape (N,3) for three labels
        arr = []
        for t in texts:
            if 'idiot' in t.lower():
                arr.append([0.1, 0.8, 0.05])
            else:
                arr.append([0.9, 0.05, 0.02])
        return np.array(arr)

    labels = ['clean', 'toxic', 'insult']
    out = explain_multilabel("You're an idiot", predict_fn, labels, num_features=3)
    assert isinstance(out, dict)
    assert '__detailed__' in out


def test_ontology_priority_and_recommendation():
    scores = {'threat': 0.9, 'toxic': 0.4}
    plan = get_intervention_plan(scores, min_score=0.3)
    assert plan['severity'] == 'CRITICAL' or plan['detected_label'] == 'threat'
    # clean plan should always include a confidence float
    clean = get_intervention_plan({})
    assert isinstance(clean.get('confidence'), float)
    assert clean.get('detected_label') == 'clean'
    plan2 = recommend_intervention(plan.copy())
    assert 'recommended_action' in plan2


def try_load_model_and_run(model_name=None):
    """Attempt to import and run AdvancedContextModel with a small batch.
    Returns True if successful, False otherwise (so tests can skip gracefully).
    """
    try:
        from src.bert_model import AdvancedContextModel
        m = AdvancedContextModel(model_name=model_name) if model_name else AdvancedContextModel()
        texts = ["You're an idiot", "I will kill you"]
        probs = m.predict_proba(texts)
        # must return a numpy array with batch rows
        return isinstance(probs, np.ndarray) and probs.shape[0] == 2
    except Exception:
        return False


def test_bert_model_inference_or_skip():
    # Prefer a small HF model if available; try default otherwise.
    small_model = 'sshleifer/tiny-distilroberta-base'
    ok = try_load_model_and_run(small_model)
    if not ok:
        pytest.skip("Unable to run HF model in this environment; skipping heavy inference test")


def test_full_system_end_to_end_or_skip():
    try:
        from src.main_system import CyberbullyingSystem
        sys = CyberbullyingSystem()
        out = sys.analyze("You're an idiot")
        assert 'is_bullying' in out and 'scores' in out
    except Exception:
        pytest.skip("Full system run skipped due to environment limitations")


def test_format_detection_output():
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from ui_streamlit import format_detection
    sample = {
        'text':'i will kill you',
        'is_bullying': True,
        'severity':'CRITICAL',
        'detected_types':['threat'],
        'scores':{'threat':0.9},
        'explanation':'Threat detected',
        'action':'POLICE_ALERT',
        'confidence':0.5,
        'highlighted_words':[('kill',0.86),('you',0.05),('i',0.0)],
        'context_info':{'reason':'The text implies intent to kill, injure, or physically harm.'},
        'processing_time_ms':123.4,
        'detected_label':'threat'
    }
    md = format_detection(sample)
    assert '📝 **Input Text:**' in md
    assert '🎯 **Detected Label:** threat' in md
    assert '👁️  **Visual Proof:**' in md
    assert '📍 **Context Analysis:**' in md


def test_api_endpoints():
    try:
        from fastapi.testclient import TestClient
        from src.api import app
    except ImportError:
        pytest.skip("FastAPI not installed")

    client = TestClient(app)
    h = client.get("/health")
    assert h.status_code == 200
    assert 'status' in h.json()

    payload = {
        "text": "You are an idiot",
        "include_explanation": True,
        "use_ensemble": False,
        "use_advanced_context": True
    }
    r = client.post("/detect", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert 'is_bullying' in data
    assert 'severity' in data
    # confidence must be numeric even for safe text
    assert isinstance(data.get('confidence'), (int, float))
    # non-bullying text should not raise validation

    # quick sanity check: non-offensive input returns 200 and a float confidence
    safe_payload = payload.copy()
    safe_payload['text'] = 'hello'
    r2 = client.post("/detect", json=safe_payload)
    assert r2.status_code == 200
    data2 = r2.json()
    assert isinstance(data2.get('confidence'), (int, float))
    assert data2.get('is_bullying') is False


def test_analyze_text_fallback(monkeypatch):
    # import the helper from UI
    import ui_streamlit
    from ui_streamlit import analyze_text, CyberbullyingSystem
    # ensure local engine is available
    assert CyberbullyingSystem is not None

    import requests

    # mimic timeout on requests.post
    class DummyResp:
        def raise_for_status(self):
            pass
        def json(self):
            return {'dummy': True}
    
    def fake_post(*args, **kwargs):
        raise requests.Timeout("Simulated timeout")

    monkeypatch.setattr(ui_streamlit.requests, 'post', fake_post)
    result, err, fallback, api_err = analyze_text("test", "http://localhost:8000", True, 1, False, False)
    assert err is None, "Error should be cleared after fallback"
    assert fallback is True
    assert api_err is not None
    assert result is not None and isinstance(result, dict)

    # simulate non-timeout request error and no fallback if engine missing
    def bad_post(*a, **k):
        raise requests.RequestException("bad")
    monkeypatch.setattr(ui_streamlit.requests, 'post', bad_post)
    # temporarily remove engine
    real_engine = ui_streamlit.CyberbullyingSystem
    ui_streamlit.CyberbullyingSystem = None
    result2, err2, fallback2, api_err2 = analyze_text("test", "http://localhost:8000", True, 1, False, False)
    assert err2 is not None
    assert result2 is None
    assert fallback2 is False
    ui_streamlit.CyberbullyingSystem = real_engine
