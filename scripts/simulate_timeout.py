import requests

def call_api(api_url, timeout):
    try:
        resp = requests.post(f"{api_url.rstrip('/')}/detect", json={"text":"test","include_explanation":True,"use_ensemble":False,"use_advanced_context":False}, timeout=timeout)
        resp.raise_for_status()
        print('json',resp.json())
    except requests.Timeout as t:
        print('timeout error', t)
    except requests.RequestException as e:
        print('request error', e)

if __name__ == '__main__':
    call_api('http://localhost:1', 1)
