from pymongo import MongoClient
from datetime import datetime
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
import time
logger = logging.getLogger(__name__)

# MongoDB connection with fallback to local instance
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 5

def get_db_connection():
    """Get MongoDB connection with retry logic"""
    for attempt in range(DEFAULT_RETRY_ATTEMPTS):
        try:
            client = MongoClient(MONGODB_URI, 
                               serverSelectionTimeoutMS=5000,
                               connectTimeoutMS=5000)
            client.admin.command('ping')
            db = client['chat_app']
            logger.info("Successfully connected to MongoDB")
            return db
        except Exception as e:
            logger.warning(f"Connection attempt {attempt + 1} failed: {str(e)}")
            if attempt < DEFAULT_RETRY_ATTEMPTS - 1:
                time.sleep(DEFAULT_RETRY_DELAY)
            else:
                logger.error("All connection attempts failed")
                return None

def save_message(message_content: str, role: str, model: str):
    """Save a message to MongoDB"""
    try:
        db = get_db_connection()
        if db is not None:  # Fixed boolean check
            messages = db['messages']
            message_doc = {
                'content': message_content,
                'role': role,
                'model': model,
                'timestamp': datetime.utcnow()
            }
            messages.insert_one(message_doc)
            logger.info(f"Message saved successfully: {role}")
            return True
    except Exception as e:
        logger.error(f"Failed to save message: {str(e)}")
    return False

def get_chat_history(limit: int = 10):
    """Retrieve chat history from MongoDB with optional limit"""
    try:
        db = get_db_connection()
        if db is not None:
            messages = db['messages']
            history = list(messages.find().sort('timestamp', -1).limit(limit))
            return [{'role': msg['role'], 'content': msg['content']} for msg in history][::-1]
        return []
    except Exception as e:
        logger.error(f"Failed to retrieve chat history: {str(e)}")
        return []

def cache_response(prompt: str, model: str, response: str):
    """Cache an Ollama response"""
    try:
        db = get_db_connection()
        if db is not None:  # Fixed boolean check
            cache = db['response_cache']
            cache_doc = {
                'prompt': prompt,
                'model': model,
                'response': response,
                'timestamp': datetime.utcnow()
            }
            cache.insert_one(cache_doc)
            logger.info("Response cached successfully")
    except Exception as e:
        logger.error(f"Failed to cache response: {str(e)}")

def get_cached_response(prompt: str, model: str):
    """Get cached response if available"""
    try:
        db = get_db_connection()
        if db is not None:  # Fixed boolean check
            cache = db['response_cache']
            cached = cache.find_one({
                'prompt': prompt,
                'model': model
            })
            if cached:
                logger.info("Cache hit: Found cached response")
                return cached['response']
    except Exception as e:
        logger.error(f"Failed to retrieve cached response: {str(e)}")
    return None

def clear_chat_history():
    """Clear all chat messages"""
    try:
        db = get_db_connection()
        if db is not None:  # Fixed boolean check
            messages = db['messages']
            messages.delete_many({})
            logger.info("Chat history cleared successfully")
            return True
    except Exception as e:
        logger.error(f"Failed to clear chat history: {str(e)}")
    return False