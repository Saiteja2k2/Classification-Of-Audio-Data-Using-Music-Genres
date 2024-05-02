FROM python:3.11
ADD requirements.txt requirements.txt
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt

EXPOSE 8080
CMD ["python3", "app.py"]