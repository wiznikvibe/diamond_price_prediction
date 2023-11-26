FROM python:3.8-slim-buster
WORKDIR /service
COPY requirments.txt .
COPY . ./
RUN pip install -r requirments.txt
ENTRYPOINT ["python3", "app.py"]