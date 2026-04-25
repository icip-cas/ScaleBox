#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SERVER_DIR="${SERVER_DIR:-${PROJECT_ROOT}/server}"
NGINX_CONF_PATH="${SERVER_DIR}/nginx.conf"

NUM_NODES="${NUM_NODES:-1}"
NGINX_PORT="${NGINX_PORT:-8081}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-5}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-300}"

log() {
    echo "[start_distributed_nginx] $*"
}

die() {
    echo "Error: $*" >&2
    exit 1
}

require_positive_int() {
    local name="$1"
    local value="$2"

    [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be a positive integer, got: ${value}"
    (( value > 0 )) || die "${name} must be greater than 0, got: ${value}"
}

require_port() {
    local port="$1"

    [[ "${port}" =~ ^[0-9]+$ ]] || die "NGINX_PORT must be an integer, got: ${port}"
    (( port >= 1 && port <= 65535 )) || die "NGINX_PORT must be in [1, 65535], got: ${port}"
}

run_nginx() {
    if [[ "${EUID}" -eq 0 ]]; then
        nginx "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo nginx "$@"
    else
        die "nginx requires root privileges and sudo is not available"
    fi
}

#############################################################
# Validate Inputs
#############################################################

require_positive_int "NUM_NODES" "${NUM_NODES}"
require_positive_int "WAIT_INTERVAL_SECONDS" "${WAIT_INTERVAL_SECONDS}"
require_positive_int "MAX_WAIT_SECONDS" "${MAX_WAIT_SECONDS}"
require_port "${NGINX_PORT}"

#############################################################
# Wait For Upstream Nodes
#############################################################

mkdir -p "${SERVER_DIR}"

log "Waiting for ${NUM_NODES} node address files in ${SERVER_DIR} (timeout: ${MAX_WAIT_SECONDS}s)..."
elapsed=0
while true; do
    shopt -s nullglob
    addr_files=("${SERVER_DIR}"/addr_*)
    shopt -u nullglob

    if (( ${#addr_files[@]} >= NUM_NODES )); then
        break
    fi

    if (( elapsed >= MAX_WAIT_SECONDS )); then
        die "Timed out after ${MAX_WAIT_SECONDS}s while waiting for ${NUM_NODES} nodes"
    fi

    log "Detected ${#addr_files[@]}/${NUM_NODES} nodes; retry in ${WAIT_INTERVAL_SECONDS}s..."
    sleep "${WAIT_INTERVAL_SECONDS}"
    (( elapsed += WAIT_INTERVAL_SECONDS ))
done

#############################################################
# Generate Nginx Config
#############################################################

cat > "${NGINX_CONF_PATH}" <<EOF
events {
    worker_connections 1048576;
}

http {
    log_format upstream_log '\$remote_addr - \$remote_user [\$time_local] '
                            '"\$request" \$status \$body_bytes_sent '
                            '"\$http_referer" "\$http_user_agent" '
                            'upstream_addr=\$upstream_addr '
                            'upstream_status=\$upstream_status '
                            'upstream_response_time=\$upstream_response_time '
                            'upstream_connect_time=\$upstream_connect_time '
                            'request_time=\$request_time';

    access_log /var/log/nginx/access.log upstream_log;

    upstream myapp1 {
EOF

healthy_count=0
for addr_file in "${addr_files[@]}"; do
    addr="$(<"${addr_file}")"

    if [[ -z "${addr}" ]]; then
        log "Address file ${addr_file} is empty; removing it"
        rm -f "${addr_file}"
        continue
    fi

    if ! curl -fsS "http://${addr}" --max-time 2 >/dev/null; then
        log "Address ${addr} is unreachable; removing ${addr_file}"
        rm -f "${addr_file}"
        continue
    fi

    printf '        server %s max_fails=3 fail_timeout=30s;\n' "${addr}" >> "${NGINX_CONF_PATH}"
    (( healthy_count += 1 ))
    log "Address ${addr} is healthy and added"
done

(( healthy_count > 0 )) || die "No healthy upstream addresses found"

cat >> "${NGINX_CONF_PATH}" <<EOF
    }

    server {
        listen ${NGINX_PORT};
        listen [::]:${NGINX_PORT};
        server_name localhost;

        location / {
            proxy_pass http://myapp1;
            proxy_set_header X-Upstream-Addr \$upstream_addr;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header Host \$host;
        }
    }

    server {
        listen 81 default_server;
        listen [::]:81 default_server;
        root /var/www/html;
        index index.html index.htm index.nginx-debian.html;
        server_name _;

        location / {
            try_files / =404;
        }

        location /nginx_status {
            stub_status;
            # allow 127.0.0.1;
            # deny all;
        }
    }

    client_max_body_size 128M;
    fastcgi_read_timeout 600;
    proxy_read_timeout 600;
}
EOF

log "Nginx config generated at ${NGINX_CONF_PATH}"
cat "${NGINX_CONF_PATH}"

#############################################################
# Validate And Apply Nginx Config
#############################################################

log "Validating nginx config"
run_nginx -t -c "${NGINX_CONF_PATH}"

if [[ -f /var/run/nginx.pid ]]; then
    log "Nginx is running; reloading config"
    run_nginx -s reload -c "${NGINX_CONF_PATH}"
else
    log "Nginx is not running; starting"
    run_nginx -c "${NGINX_CONF_PATH}"
fi

log "Nginx is serving on port ${NGINX_PORT}"
log "Access URL: http://localhost:${NGINX_PORT}"
