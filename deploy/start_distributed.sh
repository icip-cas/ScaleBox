if [ $MASTER_IP == "$(hostname -I | awk '{print $1}')" ]; then
    bash deploy/start_distributed_nginx.sh
else
    make run-distributed
