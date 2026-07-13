#!/bin/bash
# Stops the PharmAgent backend started by PharmAgent.app
pkill -f "uvicorn app.main:app" && echo "PharmAgent stopped." || echo "PharmAgent was not running."
sleep 1
