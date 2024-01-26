FROM python:3.10

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y #required for open-cv
RUN pip install -r servers/requirements.txt
RUN yes | apt install npm
COPY . .
WORKDIR servers/mamo-front
RUN npm install
RUN npm run build
WORKDIR /
RUN chmod +x /servers/serve.sh
CMD ["/servers/serve.sh"]