#!/bin/bash

cd titanic_service
uvicorn api:app --host 0.0.0.0 --port 12000 --reload