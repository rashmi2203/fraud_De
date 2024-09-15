FROM python:3.9
WORKDIR /app
COPY . /app
RUN pip install scikit-learn joblib
CMD ["python", "score.py"]
