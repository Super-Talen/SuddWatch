# SuddWatch flood-risk API
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

COPY src/__init__.py src/transformers.py src/pipeline_v2.py src/calibration.py src/
COPY models/suddwatch_model.joblib models/suddwatch_model.joblib
COPY api/ api/

# run as a non-root user
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
