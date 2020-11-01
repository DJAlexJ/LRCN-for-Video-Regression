FROM python:3-slim

WORKDIR /usr/src/app

COPY app.py .
COPY model_weights.pt .
COPY model.py .
COPY preprocessing.py .
COPY loading_data.py .
COPY config.py .
COPY requirements.txt .
COPY Markup.xls .

RUN apt-get update && apt-get install libglib2.0-0 libgl1-mesa-dev -y && \
	pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "./app.py"]