#!/usr/bin/env bash

# Initialize the database
airflow db init

# Start the scheduler in the background
airflow scheduler &> /dev/null &

# Create an admin user (this will error if the user already exists)
airflow users create \
    --username admin \
    --password admin \
    --role Admin \
    --email admin@admin.com \
    --firstname admin \
    --lastname admin || true

# Start the web server
exec airflow webserver
