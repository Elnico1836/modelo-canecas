FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY clasificador_canecas.h5 .
EXPOSE 5000
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]