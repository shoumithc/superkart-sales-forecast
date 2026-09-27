#!/bin/bash
# Build and run the SuperKart backend + frontend containers on a shared Docker network
set -e

docker network create superkart-net 2>/dev/null || true      # shared network (ignore if it exists)
docker rm -f superkart-backend superkart-frontend 2>/dev/null || true   # remove old containers

# Backend (Flask API) - reachable from the frontend as http://superkart-backend:7860
docker build -t superkart-backend ./backend_files
docker run -d --name superkart-backend --network superkart-net -p 7860:7860 superkart-backend

# Frontend (Streamlit) - talks to the backend through the Docker network
docker build -t superkart-frontend ./frontend_files
docker run -d --name superkart-frontend --network superkart-net -p 8501:8501 \
  -e BACKEND_URL=http://superkart-backend:7860 superkart-frontend

docker ps
echo "Backend  -> port 7860 | Frontend -> port 8501 (see the PORTS tab)"
