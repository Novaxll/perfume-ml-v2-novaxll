FROM python:3.11-slim
WORKDIR /app
COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt
COPY . .
EXPOSE 80
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-80}"]
