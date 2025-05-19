FROM python:3.10
COPY servers/requirements.txt /servers/requirements.txt
RUN pip install -r /servers/requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y #required for open-cv
RUN yes | apt install npm
COPY . .
COPY servers/config.json /servers/mamo-front/src/config.json
WORKDIR /servers/mamo-front
RUN npm install
RUN npm run build
WORKDIR /servers
CMD ["python", "serve_rest.py"]
