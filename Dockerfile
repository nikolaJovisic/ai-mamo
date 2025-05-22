FROM python:3.10
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y #required for open-cv
RUN yes | apt install npm

COPY servers/requirements.txt /servers/requirements.txt
RUN pip install -r /servers/requirements.txt

COPY servers/mamo-front/package*.json /servers/mamo-front/
WORKDIR /servers/mamo-front
RUN npm install

COPY servers/mamo-front /servers/mamo-front
COPY servers/config.json /servers/mamo-front/src/config.json
RUN npm run build

COPY servers /servers

WORKDIR /servers
CMD ["python", "serve_rest.py"]
