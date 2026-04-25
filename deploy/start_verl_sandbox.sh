#!/usr/bin/env bash

set -euo pipefail

# Recommended runtime image: `scripts/Dockerfile.verl`
# Required environment variables: `RANK`, `WORLD_SIZE`, `MASTER_ADDR`.
# Optional: login to Weights & Biases before starting training.

export VLLM_ATTENTION_BACKEND=XFORMERS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCALEBOX_PATH="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_PATH="${PROJECT_PATH:-$HOME/verl}"

log() {
    echo "[start_verl_sandbox] $*"
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

require_non_negative_int() {
    local name="$1"
    local value="$2"

    [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be a non-negative integer, got: ${value}"
}

require_env() {
    local name="$1"
    [[ -n "${!name:-}" ]] || die "${name} is required but not set"
}


check_port() {
    (echo > /dev/tcp/${MASTER_ADDR}/${PORT}) >/dev/null 2>&1
    return $?
}

PORT=6379

#############################################################
# Validate Environment
#############################################################

require_env "RANK"
require_env "WORLD_SIZE"
require_non_negative_int "RANK" "${RANK}"
require_positive_int "WORLD_SIZE" "${WORLD_SIZE}"
require_env "MASTER_ADDR"
(( RANK < WORLD_SIZE )) || die "RANK (${RANK}) must be smaller than WORLD_SIZE (${WORLD_SIZE})"
command -v make >/dev/null 2>&1 || die "make command not found"
command -v ray >/dev/null 2>&1 || die "ray command not found"
[[ -f "${SCALEBOX_PATH}/deploy/start_distributed_nginx.sh" ]] || die "missing deploy/start_distributed_nginx.sh"

#############################################################
# Start ScaleBox service
#############################################################

current_date="$(date +"%m%d")"

# Sandbox service state and node address files are stored under SERVER_DIR.
export SERVER_DIR="${PROJECT_PATH}/server/new_multi_node_sandbox_codemath_16k_7B_test_${current_date}"

cd "${SCALEBOX_PATH}"

mkdir -p "${SERVER_DIR}"
log "Using SERVER_DIR=${SERVER_DIR}"

if [[ "${WORLD_SIZE}" -eq 1 || "${RANK}" -ne 0 ]]; then
    log "Starting sandbox server on rank ${RANK}"
    [[ -f "${HOME}/miniconda3/bin/activate" ]] || die "conda activate script not found at ${HOME}/miniconda3/bin/activate"
    source ~/miniconda3/bin/activate
    source activate sandbox
    make run-distributed > "${SERVER_DIR}/sandbox_${RANK}.log" 2>&1 &
    conda deactivate
fi

# Give workers time to register addr_* files before rank0 starts nginx bootstrap.
sleep 60s

#############################################################
# Start Nginx service
#############################################################

log "Current rank is: ${RANK}"
EFFECTIVE_NGINX_PORT="${NGINX_PORT:-8082}"
if [[ "${RANK}" -eq 0 ]]; then
    NUM_NODES="$(( WORLD_SIZE - 1 ))"
    NGINX_PORT="${EFFECTIVE_NGINX_PORT}"
    WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-5}"
    MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-600}"

    if (( NUM_NODES > 0 )); then
        log "Starting nginx via deploy/start_distributed_nginx.sh"
        SERVER_DIR="${SERVER_DIR}" \
        NUM_NODES="${NUM_NODES}" \
        NGINX_PORT="${NGINX_PORT}" \
        WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS}" \
        MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS}" \
        bash "${SCALEBOX_PATH}/deploy/start_distributed_nginx.sh"
    else
        log "WORLD_SIZE=1, skipping nginx upstream bootstrap"
    fi
fi

#############################################################
# Periodic Nginx Telemetry
#############################################################

PERIODIC_TASK_PID=""
if [[ "${RANK}" -eq 0 ]]; then
    NGINX_LOG="${SERVER_DIR}/nginx.log"

    cleanup() {
        if [[ -n "${PERIODIC_TASK_PID}" ]]; then
            kill "${PERIODIC_TASK_PID}" >/dev/null 2>&1 || true
        fi
    }
    trap cleanup EXIT

    # Periodically dump cumulative, active, and health status metrics.
    run_periodic_task() {
        while true; do
            {
                echo "Running at $(date)"

                echo "============= Cumulative connections ================="
                awk -F'upstream_addr=' '{print $2}' /var/log/nginx/access.log | awk '{print $1}' | sort | uniq -c | sort -rn

                shopt -s nullglob
                addr_list=("${SERVER_DIR}"/addr_*)
                shopt -u nullglob

                echo "============= Active connections ================="
                # Count active established connections per upstream address.
                for addr_file in "${addr_list[@]}"; do
                    addr="$(<"${addr_file}")"
                    # Count both incoming and outgoing established connections.
                    count=$(netstat -an | grep ESTABLISHED | grep -c "$addr ")
                    if [[ ${count} -gt 0 ]]; then
                        echo "Address ${addr}: ${count} connections"
                    else
                        echo "Address ${addr}: 0 connections"
                    fi
                done

                echo "============= Working servers ================="
                
                for addr_file in "${addr_list[@]}"; do
                    addr="$(<"${addr_file}")"
                    if ! curl -s "http://${addr}" > /dev/null --max-time 2; then
                        echo "Address ${addr} is not working"
                        continue
                    fi
                    echo "Address ${addr} is working"
                done
            } >> "$NGINX_LOG" 2>&1

            # Run this telemetry cycle every 100 seconds.
            sleep 100
        done
    }

    # Run telemetry in background and track pid for EXIT cleanup.
    run_periodic_task &
    PERIODIC_TASK_PID="$!"
fi

#############################################################
# Start ray
#############################################################

if [[ "${RANK}" -eq 0 ]]; then
    ray start --head --port "${PORT}"
else
    while ! check_port; do
        echo "Port ${PORT} on ${MASTER_ADDR} is not open yet. Retrying in 30 seconds..."
        sleep 30s # wait for head node to start
    done
    ray start --address="${MASTER_ADDR}:${PORT}"
fi

echo "Ray started on rank ${RANK}"

#############################################################
# RL Training (rank 0 only)
#############################################################

cd "${PROJECT_PATH}"
mkdir -p logs

current_time="$(date +"%m%d%H%M")"
export SANDBOX_ENDPOINT="http://localhost:${EFFECTIVE_NGINX_PORT}"

if [[ "${RANK}" -eq 0 ]]; then
    mini_batch_size=256
    temperature=0.9
    clip_ratio=0.2

    max_prompt_length=$((1024 * 2))
    max_response_length=$((1024 * 8))
    max_num_batched_tokens=$((1024 * 10))
    enable_overlong_buffer=True
    overlong_buffer_len=$((1024 * 4))

    export MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    export OUTPUT_DIR="${PROJECT_PATH}/checkpoints/fusion_prime_7b_single_distill-mb32-t0.9-cr0.2-${current_time}"
    export TRAIN_FILES="[YOUR_TRAIN_FILE_PATH]"
    export VAL_FILES="[YOUR_VAL_FILE_PATH]"

    PYTHONUNBUFFERED=1 /usr/bin/python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VAL_FILES} \
    data.train_batch_size=1024 \
    data.val_batch_size=32 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_batch_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.clip_ratio=${clip_ratio} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    reward_model.reward_manager="prime" \
    custom_reward_function.path=scalebox.py \
    custom_reward_function.name=compute_score \
    +custom_reward_function.reward_kwargs.sandbox_fusion_url=${SANDBOX_ENDPOINT}/common_evaluate_batch \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='code_rl' \
    trainer.experiment_name="fusion_prime_7b-mb${mini_batch_size}-t${temperature}-cr${clip_ratio}-${current_time}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${WORLD_SIZE} \
    trainer.save_freq=20 \
    trainer.test_freq=160 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=10 \
    trainer.default_local_dir=${OUTPUT_DIR} \
    data.filter_overlong_prompts=True \
    2>&1 | tee logs/fusion_prime_7b-mb${mini_batch_size}-t${temperature}-cr${clip_ratio}-${current_time}.log

    echo "Training is done on rank 0, stopping Ray..."
    ray stop --force

else
    #############################################################
    # rank != 0 processes, wait for main process to stop
    #############################################################
    echo "Worker rank ${RANK} is waiting for Ray to stop..."

    # (optional) if your Ray version is new, you can use ray status to detect
    while true; do
        if ! ray status 1>/dev/null 2>&1; then
            echo "Ray cluster no longer available. Exiting worker..."
            break
        fi
        sleep 5m
    done
fi

echo "Rank ${RANK} script ended."
