# Use an official Python runtime as a base image
FROM python:3.9-slim

# Define build-time arguments
ARG AIRFLOW_VERSION=2.5.1
ARG AIRFLOW_HOME=/opt/airflow

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AIRFLOW_HOME=${AIRFLOW_HOME} \
    PYTHONPATH=/app/src

# Install necessary system dependencies
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    wget \
    libssl-dev \
    git \
    libpq-dev \
    build-essential \
    libffi-dev \
    libsasl2-dev \
    libkrb5-dev \
    default-libmysqlclient-dev \
    libsndfile1 \
    libx11-dev \
    ffmpeg \
    vim \
    locales \
    libhdf5-dev \
    pkg-config && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Download Airflow constraints
RUN wget https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-3.9.txt -O /constraints-3.9.txt

# Set working directory and copy application code
WORKDIR /app
COPY . /app/

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir apache-airflow[postgres,mysql]==${AIRFLOW_VERSION} --constraint /constraints-3.9.txt

# Copy entrypoint script
# Copy entrypoint script and set it to be executable
COPY ./entrypoint.sh ${AIRFLOW_HOME}/entrypoint.sh

# Prepare Airflow user and directory structure
RUN useradd -ms /bin/bash -d ${AIRFLOW_HOME} airflow && \
    mkdir -p ${AIRFLOW_HOME}/dags && \
    chown -R airflow: ${AIRFLOW_HOME} && \
    chmod +x ${AIRFLOW_HOME}/entrypoint.sh

# Switch to non-root user
USER airflow

# Expose necessary ports
EXPOSE 8888 8050 8080

# Configure entrypoint and default command
ENTRYPOINT ["/opt/airflow/entrypoint.sh"]
CMD ["sh", "-c", "airflow webserver --port 8080 & airflow scheduler"]
