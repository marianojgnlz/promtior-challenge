#!/bin/sh

# Create certs directory if it doesn't exist
mkdir -p certs

# Check if certificates exist, if not create self-signed ones
if [ ! -f certs/fullchain.pem ] || [ ! -f certs/privkey.pem ]; then
    echo "Generating self-signed certificates..."
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout certs/privkey.pem \
        -out certs/fullchain.pem \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
    
    # Set proper permissions
    chmod 644 certs/fullchain.pem
    chmod 644 certs/privkey.pem
    
    echo "Self-signed certificates generated successfully"
else
    echo "Certificates already exist"
fi 