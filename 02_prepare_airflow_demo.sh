#!/bin/bash

docker build airflow_dockerfile -t data-cafe-paul-airflow:0.0.1
docker-compose -f airflow-docker-compose.yaml up -d