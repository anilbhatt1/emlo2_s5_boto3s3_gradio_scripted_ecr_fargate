FROM python:3.9.14-slim-bullseye

WORKDIR /opt/src

COPY requirements.txt requirements.txt

Run pip3 install -r requirements.txt && rm -rf /root/.cache/pip

ENV model s3://emlo2s5/model_weights/model.script.pt
ENV flagged_dir s3://emlo2s5/inf_outputs

COPY . .

EXPOSE 8080

ENTRYPOINT ["python", "inference.py"]