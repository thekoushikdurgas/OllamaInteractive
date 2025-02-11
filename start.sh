#!/bin/bash

# Start MongoDB
mongod --fork --logpath /var/log/mongodb.log

# Start Ollama in background
ollama serve &

# Wait for MongoDB to be ready
until mongosh --eval "print(\"MongoDB is ready\")" > /dev/null 2>&1; do
    echo "Waiting for MongoDB to start..."
    sleep 1
done

# Start Streamlit application
streamlit run main.py --server.address 0.0.0.0 --server.port 5000
