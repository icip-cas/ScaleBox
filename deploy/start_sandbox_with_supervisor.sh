#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SUPERVISOR_DIR="${PROJECT_ROOT}/supervisor"
LOG_DIR="${SUPERVISOR_DIR}/logfile"
SUPERVISOR_CONF="${SUPERVISOR_DIR}/supervisord.conf"
SUPERVISOR_SOCK="${SUPERVISOR_DIR}/supervisor.sock"
SUPERVISOR_PID="${SUPERVISOR_DIR}/supervisord.pid"
SUPERVISOR_PORT="${SUPERVISOR_PORT:-9001}"

log() {
    echo "[start_sandbox_with_supervisor] $*"
}

die() {
    echo "Error: $*" >&2
    exit 1
}

run_python_pip_install() {
    if [[ "${EUID}" -eq 0 ]]; then
        python3 -m pip install supervisor
    elif command -v sudo >/dev/null 2>&1; then
        sudo python3 -m pip install supervisor
    else
        die "supervisor is not installed and sudo is not available"
    fi
}

#############################################################
# Validate Inputs And Dependencies
#############################################################

[[ "${SUPERVISOR_PORT}" =~ ^[0-9]+$ ]] || die "SUPERVISOR_PORT must be an integer, got: ${SUPERVISOR_PORT}"
(( SUPERVISOR_PORT >= 1 && SUPERVISOR_PORT <= 65535 )) || die "SUPERVISOR_PORT must be in [1, 65535], got: ${SUPERVISOR_PORT}"
command -v make >/dev/null 2>&1 || die "make command not found"

if ! command -v supervisord >/dev/null 2>&1 || ! command -v supervisorctl >/dev/null 2>&1; then
    log "Installing supervisor via pip"
    run_python_pip_install
fi

command -v supervisord >/dev/null 2>&1 || die "supervisord not found after installation"
command -v supervisorctl >/dev/null 2>&1 || die "supervisorctl not found after installation"

#############################################################
# Prepare Supervisor Workspace
#############################################################

mkdir -p "${SUPERVISOR_DIR}" "${LOG_DIR}"

cat > "${SUPERVISOR_CONF}" <<EOF
[unix_http_server]
file=${SUPERVISOR_SOCK}

[supervisord]
logfile=${LOG_DIR}/supervisord.log
pidfile=${SUPERVISOR_PID}
childlogdir=${LOG_DIR}

[rpcinterface:supervisor]
supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface

[supervisorctl]
serverurl=unix://${SUPERVISOR_SOCK}

[inet_http_server]
port=127.0.0.1:${SUPERVISOR_PORT}

[program:main]
directory=${PROJECT_ROOT}
command=make run-distributed
autostart=true
autorestart=true
startsecs=1
startretries=5
stderr_logfile=${LOG_DIR}/stderr.log
stdout_logfile=${LOG_DIR}/stdout.log
EOF

#############################################################
# Stop Existing Instance And Start New One
#############################################################

# Clear run logs for this launch while keeping supervisord daemon log history.
: > "${LOG_DIR}/stderr.log"
: > "${LOG_DIR}/stdout.log"

if [[ -S "${SUPERVISOR_SOCK}" ]]; then
    log "Stopping existing supervisor instance"
    supervisorctl -c "${SUPERVISOR_CONF}" shutdown >/dev/null 2>&1 || true
fi

log "Starting supervisor"
supervisord -c "${SUPERVISOR_CONF}"

#############################################################
# Report Process Status
#############################################################

log "Checking program status"
for _ in {1..5}; do
    supervisorctl -c "${SUPERVISOR_CONF}" status || true
    sleep 1
done
