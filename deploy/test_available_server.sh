#!/usr/bin/env bash

# Extract upstream servers from nginx config
NGINX_CONF="server/nginx.conf"
ADDRS=$(grep -E "server\s+[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+:[0-9]+" "$NGINX_CONF" | \
        grep -oE "[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+:[0-9]+" )

echo "============= Active connections ================="

# Count connections for each port
for addr in $ADDRS; do
    # Count both incoming and outgoing connections to this address
    count=$(netstat -an | grep ESTABLISHED | grep -c "$addr ")
    if [ $count -gt 0 ]; then
        echo "Address $addr: $count connections"
    else
        echo "Address $addr: 0 connections"
    fi
done

echo "=================================================="
echo "============= Active server addresses ============"
addr_list=$(ls server/addr_*)
for addr in $ADDRS; do
    # Check if the address is working
    if ! curl -s "http://${addr}" > /dev/null --max-time 2; then
        echo "Address ${addr} is not working"
        continue
    fi
    # Write the address to the nginx config file
    echo "Address ${addr} is working"
done
echo "=================================================="