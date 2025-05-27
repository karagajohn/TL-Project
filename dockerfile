FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libblas-dev liblapack-dev libatlas-base-dev gfortran \
    libfreetype6-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --user -r requirements.txt

ENV PATH=/root/.local/bin:$PATH

COPY . .

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "test.py", "--server.port=8501", "--server.address=0.0.0.0"]

