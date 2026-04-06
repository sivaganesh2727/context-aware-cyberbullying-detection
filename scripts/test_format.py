import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ui_streamlit import format_detection

print(format_detection({'text':'hi','is_bullying':False,'detected_types':[], 'context_info':{}}))
