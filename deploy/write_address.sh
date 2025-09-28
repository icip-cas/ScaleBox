HOST=$1
PORT=$2
SERVER_DIR=${SERVER_DIR:-"server"}

# Wait for the server to be ready
while ! curl -s "http://${HOST}:${PORT}"
do
    echo "Waiting for server at ${HOST}:${PORT} to be ready..."
    sleep 1
done

# Write the server address to a file
echo "Server at ${HOST}:${PORT} is ready."
HOST_FILE=${SERVER_DIR}/addr_${HOST}_${PORT}.txt
echo "${HOST}:${PORT}" > ${HOST_FILE}
