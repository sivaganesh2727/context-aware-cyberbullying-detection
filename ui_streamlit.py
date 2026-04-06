import streamlit as st
import requests
from typing import Any, Dict, List

# try to import the local inference engine so the UI can fall back
try:
    from src.main_system import CyberbullyingSystem
except Exception:
    CyberbullyingSystem = None


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------

def format_detection(result: Dict[str, Any]) -> str:
    """Convert a detection result dict into a markdown string."""
    try:
        # basic fields
        text = result.get('text', '')
        label = result.get('detected_label') or result.get('label', 'unknown')
        severity = result.get('severity', '')
        score = result.get('confidence', result.get('score', 0))
        explanation = result.get('explanation', '')

        md = f"📝 **Input Text:** {text}  \n"
        md += f"🎯 **Detected Label:** {label}  \n"
        if severity:
            md += f"⚠️ **Severity:** {severity}  \n"
        md += f"📊 **Confidence:** {score:.3f}  \n\n"

        if explanation:
            md += f"💡 **Explanation:**  \n{explanation}  \n\n"

        # visual proof/highlighted words
        proof = result.get('highlighted_words') or result.get('highlighted_tokens')
        if proof:
            # some results are list of tuples (word,score)
            if isinstance(proof, list) and proof and isinstance(proof[0], (list, tuple)):
                words = [w for w, _ in proof]
            else:
                words = proof
            md += "👁️  **Visual Proof:**  \n" + " ".join(words) + "  \n\n"

        # context info (negation, sarcasm, reason)
        ctx = result.get('context_info', {}) or {}
        reason = ctx.get('reason') or ctx.get('context_reason')
        ctx_lines: List[str] = []
        if ctx.get('negation_detected'):
            ctx_lines.append(f"Negation ({ctx.get('negation_type','')})")
        if ctx.get('has_sarcasm'):
            ctx_lines.append(f"Sarcasm ({ctx.get('sarcasm_confidence',0):.2f})")
        if reason:
            ctx_lines.append(reason)
        if ctx_lines:
            md += "📍 **Context Analysis:** " + "; ".join(ctx_lines) + "  \n"

        return md
    except Exception as e:
        # fallback when something unexpectedly changes
        return f"Error formatting result: {e}\nRaw data: {result}"


# ---------------------------------------------------------------------------
# Core logic helpers
# ---------------------------------------------------------------------------

def analyze_text(user_input: str,
                 api_url: str,
                 use_api: bool,
                 api_timeout: int,
                 ensemble_opt: bool,
                 advanced_ctx_opt: bool):
    """Run detection via API or local engine.

    Returns a tuple ``(result, error_msg, used_fallback, api_error)`` where
    * ``result`` is the detection dict if any,
    * ``error_msg`` is a user-facing error (usually displayed with ``st.error``),
    * ``used_fallback`` indicates the UI forced local inference after an API
      failure, and
    * ``api_error`` preserves the raw problem reported by the HTTP request
      (useful when ``used_fallback`` is True).
    """
    result = None
    error_msg = None
    used_fallback = False
    api_error = None

    if use_api:
        payload = {
            "text": user_input.strip(),
            "include_explanation": True,
            "use_ensemble": ensemble_opt,
            "use_advanced_context": advanced_ctx_opt
        }
        try:
            resp = requests.post(f"{api_url.rstrip('/')}/detect", json=payload, timeout=api_timeout)
            resp.raise_for_status()
            try:
                result = resp.json()
            except ValueError:
                api_error = f"Invalid JSON from API: {resp.text}"
                error_msg = api_error
        except requests.Timeout:
            api_error = f"timeout after {api_timeout} sec"
            error_msg = api_error
        except requests.RequestException as re:
            api_error = str(re)
            error_msg = api_error

        # if API error and local engine available, warn and fallback
        if api_error and CyberbullyingSystem is not None:
            st.warning(
                f"API error ({api_error}); falling back to local inference. "
                "Ensure the backend is running (`uvicorn src.api:app --port 8000`)")
            try:
                system = CyberbullyingSystem(model_name="unitary/toxic-bert",
                                            use_ensemble=ensemble_opt,
                                            use_advanced_context=advanced_ctx_opt)
                result = system.analyze(user_input.strip())
                used_fallback = True
                error_msg = None
            except Exception as e:
                error_msg = f"Local fallback failed: {e}"
    else:
        # direct engine (no HTTP involved)
        try:
            if CyberbullyingSystem is None:
                raise RuntimeError("Engine unavailable – install src package or use API mode")
            system = CyberbullyingSystem(model_name="unitary/toxic-bert",
                                        use_ensemble=ensemble_opt,
                                        use_advanced_context=advanced_ctx_opt)
            result = system.analyze(user_input.strip())
        except Exception as e:
            error_msg = str(e)

    return result, error_msg, used_fallback, api_error


# ---------------------------------------------------------------------------
# Layout & state
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Cyberbullying Detector", layout="wide")

# sidebar controls for options
with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API base URL", value="http://localhost:8000")
    use_api = st.checkbox("Send requests to API", value=True,
                         key="use_api",
                         help="Toggle to send payloads to the FastAPI service specified above")
    api_timeout = st.number_input("API timeout (seconds)", min_value=1, max_value=120, value=30,
                                  help="Increase if your backend is slow")
    ensemble_opt = st.checkbox("Use ensemble (slower, more accurate)", value=True)
    advanced_ctx_opt = st.checkbox("Advanced context analysis", value=True)
    st.markdown("---")
    st.markdown("**About**\n" \
                "This interface is part of a complete cyberbullying detection system. " \
                "It can talk to the FastAPI backend (`src/api.py`) or run the engine directly. " \
                "The model is context‑aware, severity‑based, explainable, and provides actionable recommendations.\n")
    st.markdown("---")
    st.markdown("**API endpoints**\n" \
                "`POST /detect` – analyze a single text\n" \
                "`POST /detect-batch` – analyze multiple texts\n" \
                "`GET /health`, `/models`, `/stats`\n")
    st.markdown("_If you choose API mode, start the backend in another terminal:_\n"  \
                "```\n"  \
                "uvicorn src.api:app --host 0.0.0.0 --port 8000\n"  \
                "```\n" )
    # health-check the configured URL so users get immediate feedback
    if use_api:
        try:
            h = requests.get(f"{api_url.rstrip('/')}/health", timeout=2)
            if h.status_code != 200:
                st.warning("API health check responded with non‑200 status; requests may fail.")
        except requests.RequestException:
            st.warning("Cannot reach API at the configured URL. Disabling API mode.")
            # automatically uncheck the box to prevent further errors
            st.session_state.use_api = False
            use_api = False

if "started" not in st.session_state:
    st.session_state.started = False

st.title("🛡️ Context-Aware, Severity-Based and Explainable Cyberbullying Detection with Actionable Interventions")

if not st.session_state.started:
    st.markdown(
        """
# Welcome

This application demonstrates a **complete, production‑ready cyberbullying detection system**.  
Built on state‑of‑the‑art NLP models, it provides:

* **Context awareness** – handles negation, sarcasm, and target type.
* **Severity scoring** – maps labels to CRITICAL/HIGH/MEDIUM/LOW and recommends actions.
* **Explainability** – highlights the words that triggered the detection using LIME.
* **Actionable outputs** – suggests suspension, hiding, warning, or monitoring decisions.

Use the button below to open the chat‑style analyzer.  
You can toggle whether the UI calls the FastAPI backend or runs the engine directly.
"""
    )
    if st.button("Open analyzer"):
        st.session_state.started = True
        st.session_state.history = []
else:
    # chat state
    if "history" not in st.session_state:
        st.session_state.history = []

    # input area appears after conversation so new responses are added at bottom
    with st.form(key="input_form", clear_on_submit=True):
        user_input = st.text_area("Enter a message", height=100)
        submitted = st.form_submit_button("Send")

    if submitted and user_input.strip():
        result, error_msg, used_fallback, api_error = analyze_text(
            user_input.strip(), api_url, use_api, api_timeout,
            ensemble_opt, advanced_ctx_opt
        )

        if error_msg:
            if not used_fallback:
                st.error(f"Error during analysis: {error_msg}")
        elif result is not None:
            if used_fallback:
                msg = "Used local engine due to API failure."
                if api_error:
                    msg += f" (reason: {api_error})"
                st.info(msg)
            st.session_state.history.append({"user": user_input, "response": result})
        else:
            st.error("Unknown error occurred during detection")

    # display conversation using chat-style messages
    for item in st.session_state.history:
        user_msg = item.get("user")
        resp = item.get("response")
        with st.chat_message("user"):
            st.write(user_msg)
        with st.chat_message("assistant"):
            st.markdown(format_detection(resp), unsafe_allow_html=True)

    if st.button("Clear conversation"):
        st.session_state.history = []
