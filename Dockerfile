FROM python:3.10

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y #required for open-cv
RUN pip install git+https://github.com/nikolaJovisic/transformers opencv-python scikit-image tensorflow==2.12.0 matplotlib flask
RUN pip install pynetdicom
COPY . .
RUN chmod +x /serve.sh
CMD ["/serve.sh"]