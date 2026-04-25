#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

#############################################################
# Validate Inputs
#############################################################

if [[ -z "${MASTER_IP:-}" ]]; then
    echo "Error: MASTER_IP is not set." >&2
    echo "Please export MASTER_IP before running this script." >&2
    exit 1
fi

#############################################################
# Resolve Local Node IP
#############################################################

get_local_ip() {
    local ip_addr

    ip_addr="$(hostname -I 2>/dev/null | awk '{print $1}')"
    if [[ -n "${ip_addr}" ]]; then
        echo "${ip_addr}"
        return 0
    fi

    if command -v ip >/dev/null 2>&1; then
        ip_addr="$(ip route get 1.1.1.1 2>/dev/null | awk '{for (i = 1; i <= NF; i++) if ($i == "src") {print $(i + 1); exit}}')"
        if [[ -n "${ip_addr}" ]]; then
            echo "${ip_addr}"
            return 0
        fi
    fi

    return 1
}

#############################################################
# Start Distributed Services
#############################################################

LOCAL_IP="$(get_local_ip)"
if [[ -z "${LOCAL_IP}" ]]; then
    echo "Error: Unable to determine local IP address." >&2
    exit 1
fi

if [[ "${MASTER_IP}" == "${LOCAL_IP}" ]]; then
    echo "[start_distributed] Current node is master (${LOCAL_IP}); running distributed worker target and starting nginx entrypoint."
    make -C "${PROJECT_ROOT}" run-distributed
    bash "${PROJECT_ROOT}/deploy/start_distributed_nginx.sh"
else
    echo "[start_distributed] Current node is worker (${LOCAL_IP}); running distributed worker target."
    make -C "${PROJECT_ROOT}" run-distributed
fi
