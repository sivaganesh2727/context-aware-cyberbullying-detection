import pathlib

p = pathlib.Path(r'c:\Users\vedak\Music\Cyberbullying_Project\ui_streamlit.py')

# read with utf-8 to handle any unicode characters in the file
lines = p.read_text(encoding='utf-8').splitlines()
if any('health-check the configured URL' in l for l in lines):
    print('already patched')
else:
    new_lines = []
    for line in lines:
        new_lines.append(line)
        if 'start the backend in another terminal' in line:
            new_lines.append('    # health-check the configured URL so users get immediate feedback')
            new_lines.append('    if use_api:')
            new_lines.append('        try:')
            new_lines.append('            h = requests.get(f"{api_url.rstrip(\"/\")}/health", timeout=2)')
            new_lines.append('            if h.status_code != 200:')
            new_lines.append('                st.warning("API health check responded with non‑200 status; requests may fail.")')
            new_lines.append('        except requests.RequestException:')
            new_lines.append("            st.warning(\"Cannot reach API at the configured URL. Either start the backend or uncheck 'Send requests to API'.\")")
    p.write_text("\n".join(new_lines), encoding='utf-8')
    print('patched')
