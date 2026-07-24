"""
WSGI entry point for production deployment with gunicorn.
"""
import os
import logging

# Set environment to production if not set
if not os.environ.get("FLASK_ENV"):
    os.environ["FLASK_ENV"] = "production"

from app import app
from src.logger import setup_logger

logger = setup_logger(__name__)

if __name__ == "__main__":
    logger.info("Starting Leaf Disease Detection WSGI application")
    app.run()
