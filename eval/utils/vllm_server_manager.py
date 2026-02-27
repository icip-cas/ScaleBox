"""
VLLMServerManager - manage startup and shutdown of multiple vLLM servers.

Features:
1. Automatically calculate how many model instances can be deployed based on total GPU/NPU count and per-model usage
2. Automatically allocate GPU/NPU devices and ports
3. Launch multiple vLLM server processes
4. Provide health checks and wait-for-ready utilities
5. Provide a method to stop all servers
6. Support Huawei Ascend NPUs
"""

import os
import subprocess
import time
import socket
import signal
import requests
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import threading


@dataclass
class ServerInstance:
    """Represents one vLLM server instance."""
    process: subprocess.Popen
    port: int
    device_ids: List[int]  # Renamed: list of GPU or NPU device IDs
    model_path: str
    api_url: str
    
    def is_alive(self) -> bool:
        return self.process.poll() is None


class VLLMServerManager:
    """
    Manage multiple vLLM server instances (supports GPU and NPU).
    
    Example:
    ```python
    # GPU mode
    manager = VLLMServerManager(
        model_path="/path/to/model",
        num_gpus_total=8,
        num_gpus_per_model=2,
        base_port=8000,
    )
    
    # NPU mode
    manager = VLLMServerManager(
        model_path="/path/to/model",
        num_gpus_total=8,
        num_gpus_per_model=2,
        base_port=8000,
        use_npu=True,
    )
    
    # Start all servers
    endpoints = manager.start_servers()
    # endpoints = ["http://localhost:8000/v1", "http://localhost:8001/v1", ...]
    
    # Use endpoints for sampling...
    
    # Stop all servers
    manager.stop_servers()
    ```
    """
    
    def __init__(
        self,
        model_path: str,
        num_gpus_total: int = 1,
        num_gpus_per_model: int = 1,
        base_port: int = 8000,
        host: str = "0.0.0.0",
        max_model_len: Optional[int] = None,
        dtype: str = "auto",
        trust_remote_code: bool = True,
        api_key: str = "EMPTY",
        extra_args: Optional[List[str]] = None,
        served_model_name: Optional[str] = None,
        use_npu: bool = False,
        mem_fraction: float = 0.9,
        wait_timeout: int = 600,
        health_check_interval: int = 5,
    ):
        """
        Initialize VLLMServerManager.
        
        Args:
            model_path: Model path
            num_gpus_total: Total number of GPUs/NPUs
            num_gpus_per_model: Number of GPUs/NPUs used per model
            base_port: Starting port
            host: Host address to bind the server
            max_model_len: Max model length (optional)
            dtype: Data type (auto/float16/bfloat16/float32)
            trust_remote_code: Whether to trust remote code
            api_key: API key
            extra_args: Extra vLLM startup arguments
            served_model_name: Served model name (used in API calls)
            use_npu: Whether to use Huawei Ascend NPUs
            mem_fraction: GPU/NPU memory utilization fraction (0.0-1.0)
            wait_timeout: Timeout waiting for service readiness (seconds)
            health_check_interval: Health check interval (seconds)
        """
        self.model_path = model_path
        self.num_gpus_total = num_gpus_total
        self.num_gpus_per_model = num_gpus_per_model
        self.base_port = base_port
        self.host = host
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.api_key = api_key
        self.extra_args = extra_args or []
        self.served_model_name = served_model_name or os.path.basename(model_path)
        self.use_npu = use_npu
        self.mem_fraction = mem_fraction
        self.wait_timeout = wait_timeout
        self.health_check_interval = health_check_interval
        
        # Device type name (for logging)
        self.device_name = "NPU" if use_npu else "GPU"
        
        # Calculate deployable instance count
        self.num_instances = num_gpus_total // num_gpus_per_model
        if self.num_instances == 0:
            raise ValueError(
                f"Cannot deploy model: total {self.device_name} count ({num_gpus_total}) < required {self.device_name} per model ({num_gpus_per_model})"
            )
        
        # Store server instances
        self.server_instances: List[ServerInstance] = []
        self._started = False
        
        # ========== Added: track allocated ports ==========
        self._allocated_ports = set()
        # ==========================================
        
        # Log lock
        self._log_lock = threading.Lock()
        
        # ========== Added: create log directory ==========
        self.log_dir = os.path.join(os.getcwd(), "log")
        os.makedirs(self.log_dir, exist_ok=True)
        self._log_files = []  # Track opened log file handles
        # ========================================
        
    def _log(self, message: str):
        """Thread-safe logging output."""
        with self._log_lock:
            print(f"[VLLMServerManager] {message}")
    
    def _find_free_port(self, start_port: int) -> int:
        """Find an available port starting from the specified port."""
        port = start_port
        while port < start_port + 1000:
            # ========== Fix: skip already allocated ports ==========
            if port in self._allocated_ports:
                port += 1
                continue
            # ==========================================
            
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", port))
                    # ========== Fix: record allocated ports ==========
                    self._allocated_ports.add(port)
                    # ==========================================
                    return port
            except OSError:
                port += 1
        raise RuntimeError(f"Unable to find an available port (starting from {start_port})")
    
    def _allocate_devices(self, instance_idx: int) -> List[int]:
        """Allocate GPU/NPU devices for a specific instance."""
        start_device = instance_idx * self.num_gpus_per_model
        return list(range(start_device, start_device + self.num_gpus_per_model))
    
    def _build_server_command(self, port: int, device_ids: List[int]) -> List[str]:
        """Build the vLLM server startup command."""
        # Use the `vllm serve` command (Huawei-recommended)
        cmd = [
            "vllm", "serve", self.model_path,
            "--port", str(port),
            "--host", self.host,
            "--tensor-parallel-size", str(self.num_gpus_per_model),
            "--dtype", self.dtype,
            "--served-model-name", self.served_model_name,
        ]
        
        if self.trust_remote_code:
            cmd.append("--trust-remote-code")
        
        if self.max_model_len:
            cmd.extend(["--max-model-len", str(self.max_model_len)])
        
        if self.api_key and self.api_key != "EMPTY":
            cmd.extend(["--api-key", self.api_key])
        
        # GPU/NPU memory utilization fraction (effective for both GPU and NPU)
        cmd.extend(["--gpu-memory-utilization", str(self.mem_fraction)])
        
        # NPU-specific arguments
        if self.use_npu:
            # Note: --device npu is not needed; vLLM Ascend auto-detects NPU via env vars
            # Disable CUDA graph (not supported by NPU)
            cmd.append("--enforce-eager")
        
        # Append extra arguments
        cmd.extend(self.extra_args)
        
        return cmd
    
    def _setup_npu_environment(self, env: dict, device_ids: List[int]) -> dict:
        """Set NPU-related environment variables."""
        device_str = ",".join(map(str, device_ids))
        
        # Huawei Ascend NPU environment variables
        env["ASCEND_VISIBLE_DEVICES"] = device_str
        env["ASCEND_RT_VISIBLE_DEVICES"] = device_str
        
        # Set NPU runtime configurations
        env["HCCL_BUFFSIZE"] = "120"
        env["HCCL_OP_BASE_FFTS_MODE_ENABLE"] = "TRUE"
        env["HCCL_ALGO"] = "level0:NA;level1:ring"
        
        # Disable CUDA (ensure NPU is used)
        env["CUDA_VISIBLE_DEVICES"] = ""
        
        # vLLM NPU backend setting
        env["VLLM_USE_ASCEND"] = "1"
        
        # Optional: set log level
        if "ASCEND_GLOBAL_LOG_LEVEL" not in env:
            env["ASCEND_GLOBAL_LOG_LEVEL"] = "3"  # ERROR level
        
        return env
    
    def _setup_gpu_environment(self, env: dict, device_ids: List[int]) -> dict:
        """Set GPU-related environment variables."""
        # If CUDA_VISIBLE_DEVICES is already set, use it as the available GPU list
        if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"]:
            available_gpus = [int(x.strip()) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip()]
            # Map device_ids to actual GPU IDs
            actual_device_ids = [available_gpus[i] for i in device_ids if i < len(available_gpus)]
            device_str = ",".join(map(str, actual_device_ids))
        else:
            device_str = ",".join(map(str, device_ids))
        env["CUDA_VISIBLE_DEVICES"] = device_str
        return env
    
    def _wait_for_server(self, port: int, timeout: int) -> bool:
        """Wait for the server to become ready."""
        health_url = f"http://localhost:{port}/health"
        models_url = f"http://localhost:{port}/v1/models"
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Check health endpoint first
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    # Then check models endpoint to ensure the model is loaded
                    response = requests.get(models_url, timeout=5)
                    if response.status_code == 200:
                        return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(self.health_check_interval)
        
        return False
    
    def _start_single_server(self, instance_idx: int) -> Optional[ServerInstance]:
        """Start a single vLLM server."""
        # Allocate devices
        device_ids = self._allocate_devices(instance_idx)
        
        # Find an available port
        port = self._find_free_port(self.base_port + instance_idx)
        
        # Set environment variables
        env = os.environ.copy()
        if self.use_npu:
            env = self._setup_npu_environment(env, device_ids)
        else:
            env = self._setup_gpu_environment(env, device_ids)
        
        # Build command
        cmd = self._build_server_command(port, device_ids)
        
        self._log(f"Starting instance {instance_idx}: port={port}, {self.device_name}={device_ids}")
        self._log(f"Command: {' '.join(cmd)}")
        
        # Print key environment variables (for debugging)
        if self.use_npu:
            self._log(f"Environment: ASCEND_VISIBLE_DEVICES={env.get('ASCEND_VISIBLE_DEVICES')}, ASCEND_RT_VISIBLE_DEVICES={env.get('ASCEND_RT_VISIBLE_DEVICES')}")
        else:
            self._log(f"Environment: CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}")
        
        try:
            # ========== Changed: write logs into the log directory ==========
            log_file_path = os.path.join(self.log_dir, f"vllm_server_{port}.log")
            log_file = open(log_file_path, "w")
            self._log_files.append(log_file)
            self._log(f"Log file: {log_file_path}")
            # =============================================
            
            # Launch process
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,  # Create a new process group for easier cleanup
            )
            
            api_url = f"http://localhost:{port}/v1"
            
            instance = ServerInstance(
                process=process,
                port=port,
                device_ids=device_ids,
                model_path=self.model_path,
                api_url=api_url,
            )
            
            return instance
            
        except Exception as e:
            self._log(f"Failed to start instance {instance_idx}: {e}")
            return None
    
    def start_servers(self, wait_ready: bool = True) -> List[str]:
        """
        Start all vLLM server instances.
        
        Args:
            wait_ready: Whether to wait until all services are ready
            
        Returns:
            List of API endpoints for all services
        """
        if self._started:
            self._log("Servers are already running, returning existing endpoints")
            return self.get_endpoints()
        
        self._log(f"Preparing to start {self.num_instances} vLLM server instances")
        self._log(f"Model: {self.model_path}")
        self._log(f"Device type: {self.device_name}")
        self._log(f"Each instance uses {self.num_gpus_per_model} {self.device_name}(s)")
        self._log(f"{self.device_name} memory utilization fraction: {self.mem_fraction}")
        
        # Start all instances
        for i in range(self.num_instances):
            instance = self._start_single_server(i)
            if instance:
                self.server_instances.append(instance)
                # Wait before starting the next instance to avoid resource contention
                if i < self.num_instances - 1:
                    self._log(f"Waiting 5 seconds before starting the next instance...")
                    time.sleep(5)
            else:
                self._log(f"Warning: failed to start instance {i}")
        
        if not self.server_instances:
            raise RuntimeError("No vLLM server instance was started successfully")
        
        # Wait for all services to be ready
        if wait_ready:
            self._log("Waiting for all services to become ready...")
            ready_instances = []
            
            for instance in self.server_instances:
                self._log(f"Checking service on port {instance.port}...")
                if self._wait_for_server(instance.port, self.wait_timeout):
                    self._log(f"Service on port {instance.port} is ready")
                    ready_instances.append(instance)
                else:
                    self._log(f"Warning: service on port {instance.port} timed out or failed to start")
                    # Check whether the process has already exited
                    if instance.process.poll() is not None:
                        self._log(f"Process exited with return code: {instance.process.returncode}")
                        # Try to read process output
                        try:
                            output, _ = instance.process.communicate(timeout=5)
                            if output:
                                output_str = output.decode('utf-8', errors='ignore')
                                # Print only the last 2000 characters
                                if len(output_str) > 2000:
                                    output_str = "...(truncated)...\n" + output_str[-2000:]
                                self._log(f"Process output:\n{output_str}")
                        except Exception as e:
                            self._log(f"Failed to read process output: {e}")
                    # Try to terminate processes that are not ready
                    try:
                        os.killpg(os.getpgid(instance.process.pid), signal.SIGTERM)
                    except:
                        pass
            
            self.server_instances = ready_instances
            
            if not self.server_instances:
                raise RuntimeError("No vLLM server instance became ready successfully")
        
        self._started = True
        endpoints = self.get_endpoints()
        self._log(f"Successfully started {len(endpoints)} vLLM server instance(s)")
        for i, ep in enumerate(endpoints):
            self._log(f"  Instance {i}: {ep}")
        
        return endpoints
    
    def get_endpoints(self) -> List[str]:
        """Get API endpoint list for all services."""
        return [instance.api_url for instance in self.server_instances]
    
    def get_chat_endpoints(self) -> List[str]:
        """Get Chat API endpoint list for all services."""
        return [f"{instance.api_url}/chat/completions" for instance in self.server_instances]
    
    def get_completions_endpoints(self) -> List[str]:
        """Get Completions API endpoint list for all services."""
        return [f"{instance.api_url}/completions" for instance in self.server_instances]
    
    def health_check(self) -> Dict[str, bool]:
        """Check health status of all services."""
        results = {}
        for instance in self.server_instances:
            try:
                response = requests.get(f"http://localhost:{instance.port}/health", timeout=5)
                results[instance.api_url] = response.status_code == 200
            except:
                results[instance.api_url] = False
        return results
    
    def stop_servers(self):
        """Stop all vLLM server instances."""
        if not self.server_instances:
            self._log("No running service instances to stop")
            return
        
        self._log("Stopping all vLLM server instances...")
        
        for instance in self.server_instances:
            try:
                # Send SIGTERM to the whole process group
                pgid = os.getpgid(instance.process.pid)
                os.killpg(pgid, signal.SIGTERM)
                self._log(f"Sent termination signal to service on port {instance.port} (PID={instance.process.pid}, PGID={pgid})")
            except ProcessLookupError:
                self._log(f"Service on port {instance.port} is already stopped")
            except Exception as e:
                self._log(f"Error while stopping service on port {instance.port}: {e}")
        
        # Wait for processes to exit
        for instance in self.server_instances:
            try:
                instance.process.wait(timeout=10)
                self._log(f"Service on port {instance.port} exited normally")
            except subprocess.TimeoutExpired:
                # Force kill
                self._log(f"Service on port {instance.port} is unresponsive, forcing termination...")
                try:
                    os.killpg(os.getpgid(instance.process.pid), signal.SIGKILL)
                except:
                    pass
        
        self.server_instances = []
        self._started = False
        # ========== Added: clear allocated port records ==========
        self._allocated_ports.clear()
        # ==========================================
        
        # ========== Added: close log file handles ==========
        for log_file in self._log_files:
            try:
                log_file.close()
            except:
                pass
        self._log_files = []
        self._log(f"Log files saved to: {self.log_dir}")
        # ==========================================
        
        self._log("All vLLM server instances have been stopped")
    
    def __enter__(self):
        """Context manager entry."""
        self.start_servers()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_servers()
        return False
    
    def get_model_name(self) -> str:
        """Get the model name served by the service."""
        return self.served_model_name
    
    def get_device_type(self) -> str:
        """Get device type."""
        return self.device_name
