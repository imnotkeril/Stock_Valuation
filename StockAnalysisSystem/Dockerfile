# Build from the repository root:
#   docker build -f StockAnalysisSystem/Dockerfile -t stock-analysis .
#   docker run -p 8501:8501 --env-file .env stock-analysis
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY StockAnalysisSystem/requirements.txt StockAnalysisSystem/requirements.txt
RUN pip install -r StockAnalysisSystem/requirements.txt

COPY .streamlit .streamlit
COPY StockAnalysisSystem StockAnalysisSystem

RUN useradd --create-home --uid 1000 app \
    && mkdir -p StockAnalysisSystem/src/data/cache StockAnalysisSystem/src/data/sector_data \
    && chown -R app:app /app
USER app

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)" || exit 1

CMD ["streamlit", "run", "StockAnalysisSystem/src/main.py", "--server.address=0.0.0.0", "--server.port=8501"]
