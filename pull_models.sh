#!/bin/sh

echo "Pulling required models..."
/usr/bin/ollama pull llama3.2:1b
/usr/bin/ollama pull deepseek-r1:1.5b
echo "Models pulled successfully!" 
