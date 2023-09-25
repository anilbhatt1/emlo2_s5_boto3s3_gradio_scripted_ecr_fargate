FROM python:3.9.7-slim

WORKDIR src

COPY requirements.txt requirements.txt

Run pip3 install -r requirements.txt && rm -rf /root/.cache/pip

COPY . .

EXPOSE 8080

ENTRYPOINT ["./entrypoint.sh"]