# Simple container running both API and Streamlit UI
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8501

# start uvicorn on 8000 and streamlit on 8501
CMD ["bash","-c","uvicorn src.api:app --host 0.0.0.0 --port 8000 & \
                streamlit run ui_streamlit.py --server.port=8501 --server.address=0.0.0.0"]
