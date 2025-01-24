#!/bin/sh

echo "Pulling required models..."
/usr/bin/ollama pull llama3.2
/usr/bin/ollama pull deepseek-r1:7b
echo "Models pulled successfully!" 
