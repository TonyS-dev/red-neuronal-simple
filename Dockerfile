FROM tensorflow/tensorflow:2.15.0

WORKDIR /app

COPY neurona-simple /app

RUN pip install matplotlib numpy

CMD ["python", "neurona-simple.py"]
