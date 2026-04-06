import streamlit as st
print('version', st.__version__)
print('has chat_message', hasattr(st, 'chat_message'))
print('has experimental_rerun', hasattr(st, 'experimental_rerun'))
