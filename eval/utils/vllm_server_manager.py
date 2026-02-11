"""
VLLMServerManager - 管理多个vLLM Server的启动和停止

功能：
1. 根据总GPU/NPU数量和每个模型使用的数量，自动计算可以部署多少个模型实例
2. 自动分配GPU/NPU和端口
3. 启动多个vLLM server进程
4. 提供健康检查和等待服务就绪的功能
5. 提供停止所有server的方法
6. 支持华为昇腾NPU
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
    """表示一个vLLM server实例"""
    process: subprocess.Popen
    port: int
    device_ids: List[int]  # 改名：GPU或NPU的ID列表
    model_path: str
    api_url: str
    
    def is_alive(self) -> bool:
        return self.process.poll() is None


class VLLMServerManager:
    """
    管理多个vLLM Server实例（支持GPU和NPU）
    
    使用示例:
    ```python
    # GPU模式
    manager = VLLMServerManager(
        model_path="/path/to/model",
        num_gpus_total=8,
        num_gpus_per_model=2,
        base_port=8000,
    )
    
    # NPU模式
    manager = VLLMServerManager(
        model_path="/path/to/model",
        num_gpus_total=8,
        num_gpus_per_model=2,
        base_port=8000,
        use_npu=True,
    )
    
    # 启动所有server
    endpoints = manager.start_servers()
    # endpoints = ["http://localhost:8000/v1", "http://localhost:8001/v1", ...]
    
    # 使用endpoints进行采样...
    
    # 停止所有server
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
        初始化VLLMServerManager
        
        Args:
            model_path: 模型路径
            num_gpus_total: 总GPU/NPU数量
            num_gpus_per_model: 每个模型使用的GPU/NPU数量
            base_port: 起始端口号
            host: 服务器绑定的主机地址
            max_model_len: 最大模型长度（可选）
            dtype: 数据类型 (auto/float16/bfloat16/float32)
            trust_remote_code: 是否信任远程代码
            api_key: API密钥
            extra_args: 额外的vLLM启动参数
            served_model_name: 服务的模型名称（用于API调用）
            use_npu: 是否使用华为昇腾NPU
            mem_fraction: GPU/NPU显存使用比例 (0.0-1.0)
            wait_timeout: 等待服务就绪的超时时间（秒）
            health_check_interval: 健康检查间隔（秒）
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
        
        # 设备类型名称（用于日志）
        self.device_name = "NPU" if use_npu else "GPU"
        
        # 计算可以部署的模型实例数量
        self.num_instances = num_gpus_total // num_gpus_per_model
        if self.num_instances == 0:
            raise ValueError(
                f"无法部署模型：总{self.device_name}数量({num_gpus_total}) < 每个模型需要的{self.device_name}数量({num_gpus_per_model})"
            )
        
        # 存储服务器实例
        self.server_instances: List[ServerInstance] = []
        self._started = False
        
        # ========== 新增：记录已分配的端口 ==========
        self._allocated_ports = set()
        # ==========================================
        
        # 日志锁
        self._log_lock = threading.Lock()
        
        # ========== 新增：创建 log 目录 ==========
        self.log_dir = os.path.join(os.getcwd(), "log")
        os.makedirs(self.log_dir, exist_ok=True)
        self._log_files = []  # 记录打开的日志文件句柄
        # ========================================
        
    def _log(self, message: str):
        """线程安全的日志输出"""
        with self._log_lock:
            print(f"[VLLMServerManager] {message}")
    
    def _find_free_port(self, start_port: int) -> int:
        """从指定端口开始查找可用端口"""
        port = start_port
        while port < start_port + 1000:
            # ========== 修复：跳过已分配的端口 ==========
            if port in self._allocated_ports:
                port += 1
                continue
            # ==========================================
            
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", port))
                    # ========== 修复：记录已分配的端口 ==========
                    self._allocated_ports.add(port)
                    # ==========================================
                    return port
            except OSError:
                port += 1
        raise RuntimeError(f"无法找到可用端口（从{start_port}开始）")
    
    def _allocate_devices(self, instance_idx: int) -> List[int]:
        """为指定实例分配GPU/NPU"""
        start_device = instance_idx * self.num_gpus_per_model
        return list(range(start_device, start_device + self.num_gpus_per_model))
    
    def _build_server_command(self, port: int, device_ids: List[int]) -> List[str]:
        """构建vLLM server启动命令"""
        # 使用 vllm serve 命令（华为官方推荐）
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
        
        # GPU/NPU 显存使用比例（对GPU和NPU都生效）
        cmd.extend(["--gpu-memory-utilization", str(self.mem_fraction)])
        
        # NPU 特定参数
        if self.use_npu:
            # 注意：不需要 --device npu，vLLM Ascend 会通过环境变量自动检测 NPU
            # 禁用 CUDA graph（NPU 不支持）
            cmd.append("--enforce-eager")
        
        # 添加额外参数
        cmd.extend(self.extra_args)
        
        return cmd
    
    def _setup_npu_environment(self, env: dict, device_ids: List[int]) -> dict:
        """设置NPU相关环境变量"""
        device_str = ",".join(map(str, device_ids))
        
        # 华为昇腾 NPU 环境变量
        env["ASCEND_VISIBLE_DEVICES"] = device_str
        env["ASCEND_RT_VISIBLE_DEVICES"] = device_str
        
        # 设置 NPU 相关的运行时配置
        env["HCCL_BUFFSIZE"] = "120"
        env["HCCL_OP_BASE_FFTS_MODE_ENABLE"] = "TRUE"
        env["HCCL_ALGO"] = "level0:NA;level1:ring"
        
        # 禁用 CUDA（确保使用 NPU）
        env["CUDA_VISIBLE_DEVICES"] = ""
        
        # vLLM NPU 后端设置
        env["VLLM_USE_ASCEND"] = "1"
        
        # 可选：设置日志级别
        if "ASCEND_GLOBAL_LOG_LEVEL" not in env:
            env["ASCEND_GLOBAL_LOG_LEVEL"] = "3"  # ERROR level
        
        return env
    
    def _setup_gpu_environment(self, env: dict, device_ids: List[int]) -> dict:
        """设置GPU相关环境变量"""
        # 如果环境变量中已经设置了 CUDA_VISIBLE_DEVICES，使用它作为可用 GPU 列表
        if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"]:
            available_gpus = [int(x.strip()) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip()]
            # 将 device_ids 映射到实际的 GPU ID
            actual_device_ids = [available_gpus[i] for i in device_ids if i < len(available_gpus)]
            device_str = ",".join(map(str, actual_device_ids))
        else:
            device_str = ",".join(map(str, device_ids))
        env["CUDA_VISIBLE_DEVICES"] = device_str
        return env
    
    def _wait_for_server(self, port: int, timeout: int) -> bool:
        """等待服务器就绪"""
        health_url = f"http://localhost:{port}/health"
        models_url = f"http://localhost:{port}/v1/models"
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # 先检查health endpoint
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    # 再检查models endpoint确保模型已加载
                    response = requests.get(models_url, timeout=5)
                    if response.status_code == 200:
                        return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(self.health_check_interval)
        
        return False
    
    def _start_single_server(self, instance_idx: int) -> Optional[ServerInstance]:
        """启动单个vLLM server"""
        # 分配设备
        device_ids = self._allocate_devices(instance_idx)
        
        # 查找可用端口
        port = self._find_free_port(self.base_port + instance_idx)
        
        # 设置环境变量
        env = os.environ.copy()
        if self.use_npu:
            env = self._setup_npu_environment(env, device_ids)
        else:
            env = self._setup_gpu_environment(env, device_ids)
        
        # 构建命令
        cmd = self._build_server_command(port, device_ids)
        
        self._log(f"启动实例 {instance_idx}: 端口={port}, {self.device_name}={device_ids}")
        self._log(f"命令: {' '.join(cmd)}")
        
        # 打印关键环境变量（调试用）
        if self.use_npu:
            self._log(f"环境变量: ASCEND_VISIBLE_DEVICES={env.get('ASCEND_VISIBLE_DEVICES')}, ASCEND_RT_VISIBLE_DEVICES={env.get('ASCEND_RT_VISIBLE_DEVICES')}")
        else:
            self._log(f"环境变量: CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}")
        
        try:
            # ========== 修改：日志输出到 log 目录 ==========
            log_file_path = os.path.join(self.log_dir, f"vllm_server_{port}.log")
            log_file = open(log_file_path, "w")
            self._log_files.append(log_file)
            self._log(f"日志文件: {log_file_path}")
            # =============================================
            
            # 启动进程
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,  # 创建新的进程组，方便后续杀死
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
            self._log(f"启动实例 {instance_idx} 失败: {e}")
            return None
    
    def start_servers(self, wait_ready: bool = True) -> List[str]:
        """
        启动所有vLLM server实例
        
        Args:
            wait_ready: 是否等待所有服务就绪
            
        Returns:
            所有服务的API endpoint列表
        """
        if self._started:
            self._log("服务器已经启动，返回现有endpoints")
            return self.get_endpoints()
        
        self._log(f"准备启动 {self.num_instances} 个vLLM server实例")
        self._log(f"模型: {self.model_path}")
        self._log(f"设备类型: {self.device_name}")
        self._log(f"每个实例使用 {self.num_gpus_per_model} 个{self.device_name}")
        self._log(f"{self.device_name}显存使用比例: {self.mem_fraction}")
        
        # 启动所有实例
        for i in range(self.num_instances):
            instance = self._start_single_server(i)
            if instance:
                self.server_instances.append(instance)
                # 在启动下一个实例前等待一段时间，避免资源竞争
                if i < self.num_instances - 1:
                    self._log(f"等待 5 秒后启动下一个实例...")
                    time.sleep(5)
            else:
                self._log(f"警告: 实例 {i} 启动失败")
        
        if not self.server_instances:
            raise RuntimeError("没有成功启动任何vLLM server实例")
        
        # 等待所有服务就绪
        if wait_ready:
            self._log("等待所有服务就绪...")
            ready_instances = []
            
            for instance in self.server_instances:
                self._log(f"检查端口 {instance.port} 的服务...")
                if self._wait_for_server(instance.port, self.wait_timeout):
                    self._log(f"端口 {instance.port} 服务就绪")
                    ready_instances.append(instance)
                else:
                    self._log(f"警告: 端口 {instance.port} 服务启动超时或失败")
                    # 检查进程是否已经死掉
                    if instance.process.poll() is not None:
                        self._log(f"进程已退出，返回码: {instance.process.returncode}")
                        # 尝试读取进程输出
                        try:
                            output, _ = instance.process.communicate(timeout=5)
                            if output:
                                output_str = output.decode('utf-8', errors='ignore')
                                # 只打印最后 2000 个字符
                                if len(output_str) > 2000:
                                    output_str = "...(截断)...\n" + output_str[-2000:]
                                self._log(f"进程输出:\n{output_str}")
                        except Exception as e:
                            self._log(f"读取进程输出失败: {e}")
                    # 尝试终止未就绪的进程
                    try:
                        os.killpg(os.getpgid(instance.process.pid), signal.SIGTERM)
                    except:
                        pass
            
            self.server_instances = ready_instances
            
            if not self.server_instances:
                raise RuntimeError("没有任何vLLM server实例成功就绪")
        
        self._started = True
        endpoints = self.get_endpoints()
        self._log(f"成功启动 {len(endpoints)} 个vLLM server实例")
        for i, ep in enumerate(endpoints):
            self._log(f"  实例 {i}: {ep}")
        
        return endpoints
    
    def get_endpoints(self) -> List[str]:
        """获取所有服务的API endpoint列表"""
        return [instance.api_url for instance in self.server_instances]
    
    def get_chat_endpoints(self) -> List[str]:
        """获取所有服务的Chat API endpoint列表"""
        return [f"{instance.api_url}/chat/completions" for instance in self.server_instances]
    
    def get_completions_endpoints(self) -> List[str]:
        """获取所有服务的Completions API endpoint列表"""
        return [f"{instance.api_url}/completions" for instance in self.server_instances]
    
    def health_check(self) -> Dict[str, bool]:
        """检查所有服务的健康状态"""
        results = {}
        for instance in self.server_instances:
            try:
                response = requests.get(f"http://localhost:{instance.port}/health", timeout=5)
                results[instance.api_url] = response.status_code == 200
            except:
                results[instance.api_url] = False
        return results
    
    def stop_servers(self):
        """停止所有vLLM server实例"""
        if not self.server_instances:
            self._log("没有需要停止的服务实例")
            return
        
        self._log("正在停止所有vLLM server实例...")
        
        for instance in self.server_instances:
            try:
                # 发送SIGTERM信号给整个进程组
                pgid = os.getpgid(instance.process.pid)
                os.killpg(pgid, signal.SIGTERM)
                self._log(f"已发送终止信号到端口 {instance.port} 的服务 (PID={instance.process.pid}, PGID={pgid})")
            except ProcessLookupError:
                self._log(f"端口 {instance.port} 的服务已经停止")
            except Exception as e:
                self._log(f"停止端口 {instance.port} 的服务时出错: {e}")
        
        # 等待进程结束
        for instance in self.server_instances:
            try:
                instance.process.wait(timeout=10)
                self._log(f"端口 {instance.port} 的服务已正常退出")
            except subprocess.TimeoutExpired:
                # 强制杀死
                self._log(f"端口 {instance.port} 的服务未响应，强制终止...")
                try:
                    os.killpg(os.getpgid(instance.process.pid), signal.SIGKILL)
                except:
                    pass
        
        self.server_instances = []
        self._started = False
        # ========== 新增：清理已分配端口记录 ==========
        self._allocated_ports.clear()
        # ==========================================
        
        # ========== 新增：关闭日志文件句柄 ==========
        for log_file in self._log_files:
            try:
                log_file.close()
            except:
                pass
        self._log_files = []
        self._log(f"日志文件已保存到: {self.log_dir}")
        # ==========================================
        
        self._log("所有vLLM server实例已停止")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start_servers()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.stop_servers()
        return False
    
    def get_model_name(self) -> str:
        """获取服务的模型名称"""
        return self.served_model_name
    
    def get_device_type(self) -> str:
        """获取设备类型"""
        return self.device_name