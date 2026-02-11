"""
MultiAPIRunner - 支持多个API端口的动态负载均衡采样

功能：
1. 将每个prompt重复n_sample次，作为独立任务
2. 多个端口共享一个任务队列
3. 每个端口有最大并发限制
4. 任务完成后立即从队列取下一个（动态调度）
"""

import asyncio
import json
from typing import List, Callable, Optional, Dict, Any, Tuple
from abc import ABC
import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm
import traceback
import random
import openai
from datetime import datetime

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

class MultiAPIRunner(ABC):
    """
    支持多个API endpoint的动态负载均衡采样Runner
    
    工作流程：
    1. 将prompts展开：每个prompt重复n_sample次，带有(orig_idx, sample_idx)标记
    2. 所有展开的任务放入共享队列
    3. 每个端口是一个worker，有并发限制(batch_size)
    4. worker不断从队列取任务，完成一个立即取下一个
    5. 结果按(orig_idx, sample_idx)聚合
    
    使用示例:
    ```python
    runner = MultiAPIRunner(
        args=args,  # args.batch_size控制每个server的最大并发
        model=model_name,
        api_endpoints=["http://localhost:8000/v1", "http://localhost:8001/v1"],
    )
    
    results = runner.run_batch(prompts)
    # results[i] = [sample_0, sample_1, ..., sample_{n_sample-1}]
    ```
    """
    
    def __init__(
        self,
        args,
        model: str,
        api_endpoints: List[str],
        api_key: str = "EMPTY",
        debug: bool = True,  # 是否开启调试日志
    ):
        """
        初始化MultiAPIRunner
        
        Args:
            args: 包含采样参数的对象（n_sample, temperature, top_p, batch_size等）
            model: 模型名称（用于API调用）
            api_endpoints: API endpoint列表（如 ["http://localhost:8000/v1", ...]）
            api_key: API密钥
            debug: 是否开启调试日志
        """
        self.args = args
        self.model = model
        self.api_endpoints = api_endpoints
        self.api_key = api_key
        self.batch_size = getattr(args, 'batch_size', 16) or 16  # 使用batch_size作为每个server的最大并发数
        self.timeout = getattr(args, 'timeout', 60000)
        self.debug = debug  # 调试开关
        
        # 模型名称
        self.model_name = getattr(args, 'model_name', model) or model
        # 预留: 计算输入token数用（可选）
        self._tokenizer = None
        self._max_context_len = (
            getattr(args, 'max_model_len', None)
            or getattr(args, 'max_context_len', None)
            or getattr(args, 'max_completion_tokens', None)
        )
        
        # 构建stop tokens列表
        stop_attr = getattr(self.args, 'stop_token', None)
        if isinstance(stop_attr, str):
            self.stop_tokens = [token.strip() for token in stop_attr.split(',') if token.strip()]
        elif stop_attr:
            self.stop_tokens = stop_attr
        else:
            self.stop_tokens = []
        
        # create client pool (与图片中的逻辑一致)
        # 使用 AsyncOpenAI 异步客户端
        self.client_pool = []
        self.api_bases = []  # 保存处理后的base_url用于日志
        api_bases = api_endpoints if isinstance(api_endpoints, list) else [api_endpoints]
        for base_url in api_bases:
            # 确保base_url格式正确
            if not base_url.endswith('/v1'):
                if base_url.endswith('/'):
                    base_url = base_url + 'v1'
                else:
                    base_url = base_url + '/v1' if '/v1' not in base_url else base_url
            
            self.api_bases.append(base_url)
            self.client_pool.append(openai.AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=self.timeout
            ))
        
        print(f"[MultiAPIRunner] ========== 初始化完成 ==========")
        print(f"[MultiAPIRunner] 创建了 {len(self.client_pool)} 个 AsyncOpenAI client:")
        for i, ep in enumerate(self.api_bases):
            print(f"  [Client {i}] {ep} (最大并发: {self.batch_size})")
        print(f"[MultiAPIRunner] 调试模式: {'开启' if self.debug else '关闭'}")
        print(f"[MultiAPIRunner] 采样策略: 异步动态负载均衡")
        print(f"[MultiAPIRunner] ================================")
    
    def _get_tokenizer(self):
        if self._tokenizer is not None or AutoTokenizer is None:
            return self._tokenizer
        model_path = getattr(self.args, 'model_path', None) or getattr(self.args, 'model_name', None)
        if not model_path:
            return None
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception as e:
            self._debug_log(f"[MultiAPIRunner] 无法加载tokenizer: {e}")
            self._tokenizer = None
        return self._tokenizer

    def _count_prompt_tokens(self, prompt_or_messages) -> Optional[int]:
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            return None
        try:
            if isinstance(prompt_or_messages, list):
                input_ids = tokenizer.apply_chat_template(
                    prompt_or_messages, tokenize=True, add_generation_prompt=True
                )
                return len(input_ids)
            return len(tokenizer.encode(prompt_or_messages))
        except Exception:
            return None

    def _adjust_max_tokens(self, prompt_or_messages, max_tokens: int) -> int:
        if not self._max_context_len:
            return max_tokens
        input_tokens = self._count_prompt_tokens(prompt_or_messages)
        if input_tokens is None:
            return max_tokens
        available = self._max_context_len - input_tokens
        if available < 1:
            available = 1
        if available < max_tokens:
            self._debug_log(
                f"[MultiAPIRunner] 动态max_tokens: {max_tokens} -> {available} "
                f"(input={input_tokens}, context={self._max_context_len})"
            )
        return min(max_tokens, available)

    def _debug_log(self, message: str):
        """调试日志，带时间戳"""
        if self.debug:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"[{timestamp}] {message}")
    
    def _truncate(self, s: str, n: int = 2000) -> str:
        if not isinstance(s, str):
            return s
        return s if len(s) <= n else s[:n] + f"...(剩余{len(s)-n}字节已截断)"

    def _sanitize_payload(self, payload: dict) -> dict:
        safe = dict(payload)
        if "messages" in safe:
            safe["messages"] = "[messages truncated]"
        return safe

    def _log_error(self, title: str, *, url=None, payload=None, status=None,
                   reason=None, headers=None, body_text=None, exc=None):
        req_id = None
        if headers:
            try:
                req_id = headers.get("x-request-id") or headers.get("X-Request-Id") \
                         or headers.get("openai-request-id") or headers.get("OpenAI-Request-Id")
            except Exception:
                pass
        meta = {
            "url": str(url) if url else None,
            "status": status,
            "reason": reason,
            "request_id": req_id,
            "payload": self._sanitize_payload(payload) if payload else None,
            "body_preview": self._truncate(body_text, 2000) if body_text else None,
        }
        if headers:
            h = dict(headers)
            if "Authorization" in h:
                h["Authorization"] = "***REDACTED***"
            meta["headers"] = h
        if exc is not None:
            meta["exception_type"] = type(exc).__name__
            meta["exception_repr"] = repr(exc)
            meta["traceback"] = traceback.format_exc()

        print(f"[MultiAPIRunner] API调用异常 - {title}:\n{json.dumps(meta, ensure_ascii=False, indent=2)}")

    async def get_openai_response(
        self,
        client: openai.AsyncOpenAI,
        model: str,
        user_prompt: str,
        max_tokens: int = 8192,
        temperature: float = 0.6,
        top_p: float = 0.95,
        n: int = 1,
        stream: bool = False,
    ) -> str:
        """
        使用AsyncOpenAI client异步调用API (与图片中的逻辑一致)
        
        Args:
            client: AsyncOpenAI客户端实例
            model: 模型名称
            user_prompt: 用户输入
            max_tokens: 最大token数
            temperature: 温度参数
            top_p: top_p参数
            n: 生成数量
            stream: 是否流式输出
            
        Returns:
            生成的文本内容
        """
        try:
            prompt_text = user_prompt if user_prompt is not None else ""
            
            # 根据输入长度动态调整max_tokens，避免超出上下文
            max_tokens = self._adjust_max_tokens(prompt_text, max_tokens)

            # 构建请求参数（标准OpenAI参数）
            kwargs = {
                "model": model,
                "prompt": prompt_text,
                "max_tokens": max_tokens,
                "n": n,
                "stream": stream,
            }
            
            # 添加自定义参数
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p
            
            # 添加stop tokens
            if self.stop_tokens:
                kwargs["stop"] = self.stop_tokens
            
            # 非标准OpenAI参数通过 extra_body 传递（如 top_k, min_p 等 vLLM 支持的参数）
            extra_body = {}
            if hasattr(self.args, 'top_k') and self.args.top_k > 0:
                extra_body["top_k"] = self.args.top_k
            
            if hasattr(self.args, 'min_p') and self.args.min_p > 0:
                extra_body["min_p"] = self.args.min_p
            
            if extra_body:
                kwargs["extra_body"] = extra_body
            
            # 异步调用API
            response = await client.completions.create(**kwargs)
            
            # 提取响应内容
            if response.choices:
                return response.choices[0].text or ""
            return ""
            
        except Exception as e:
            self._log_error("AsyncOpenAI API调用失败", exc=e)
            return ""

    async def _call_api_single(
        self, 
        prompt: str, 
        client: openai.AsyncOpenAI,
    ) -> str:
        """
        异步调用API一次，使用指定的client
        (与图片中的逻辑一致，但由endpoint_worker指定client)
        
        Returns:
            生成的文本，如果失败返回空字符串
        """
        # 调用 get_openai_response (与图片中的逻辑一致)
        res = await self.get_openai_response(
            client=client,
            model=self.model_name,
            user_prompt=prompt,
            max_tokens=self.args.max_completion_tokens,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            n=1,
            stream=False,
        )
        
        return res

    async def _run_batch_async(
        self, 
        prompts: List[str], 
        save_callback: Optional[Callable] = None
    ) -> List[List[str]]:
        """
        批量异步采样，使用动态负载均衡
        
        工作流程：
        1. 将每个prompt重复n_sample次，创建(orig_idx, sample_idx, prompt)任务
        2. 所有任务放入共享队列
        3. 每个端口worker从队列取任务，完成后立即取下一个
        4. 结果按原始索引聚合
        """
        n_sample = self.args.n_sample
        num_prompts = len(prompts)
        total_tasks = num_prompts * n_sample
        
        print(f"\n[MultiAPIRunner] ========== 开始批量采样 ==========")
        print(f"[MultiAPIRunner] Prompts 数量: {num_prompts}")
        print(f"[MultiAPIRunner] 每个 Prompt 采样次数: {n_sample}")
        print(f"[MultiAPIRunner] 总任务数: {total_tasks}")
        print(f"[MultiAPIRunner] Client 数量: {len(self.client_pool)}")
        print(f"[MultiAPIRunner] 每个 Client 最大并发: {self.batch_size}")
        print(f"[MultiAPIRunner] 理论最大并发: {len(self.client_pool) * self.batch_size}")
        print(f"[MultiAPIRunner] =====================================\n")
        
        # 1. 展开prompts：每个prompt重复n_sample次
        # 格式: (orig_idx, sample_idx, prompt)
        task_list = []
        for orig_idx, prompt in enumerate(prompts):
            for sample_idx in range(n_sample):
                task_list.append((orig_idx, sample_idx, prompt))

        # 打乱任务顺序，让难易任务均匀分布
        random.shuffle(task_list)
        self._debug_log(f"[队列] 已创建并打乱 {len(task_list)} 个任务")

        task_queue = asyncio.Queue()
        for task in task_list:
            await task_queue.put(task)
        
        self._debug_log(f"[队列] 所有任务已放入队列，队列大小: {task_queue.qsize()}")
        
        # 2. 结果存储: results[orig_idx][sample_idx] = sample_text
        results: Dict[int, Dict[int, str]] = {i: {} for i in range(num_prompts)}
        results_lock = asyncio.Lock()
        
        # 3. 进度条
        pbar = tqdm(total=total_tasks, desc="Sampling", ncols=120)
        pbar_lock = asyncio.Lock()
        
        # 4. 端口统计
        endpoint_stats = {i: {"completed": 0, "active": 0, "failed": 0} for i in range(len(self.client_pool))}
        stats_lock = asyncio.Lock()
        
        # 5. 全局计数器
        completed_count = [0]  # 使用列表以便在闭包中修改
        completed_lock = asyncio.Lock()
        
        # 6. 定义单个端口的worker（每个端口使用专属的client）
        async def endpoint_worker(endpoint_idx: int, client: openai.AsyncOpenAI):
            """
            单个端口的worker协程
            - 维护最多batch_size个并发任务
            - 任务完成后立即从队列取下一个
            - 使用专属的AsyncOpenAI client
            """
            active_tasks: set = set()
            max_concurrent = self.batch_size
            
            self._debug_log(f"[Worker {endpoint_idx}] 启动，endpoint: {self.api_bases[endpoint_idx]}")
            
            async def process_single_task(orig_idx: int, sample_idx: int, prompt: str):
                """处理单个采样任务"""
                task_id = f"prompt_{orig_idx}_sample_{sample_idx}"
                
                # 更新统计
                async with stats_lock:
                    endpoint_stats[endpoint_idx]["active"] += 1
                    current_active = endpoint_stats[endpoint_idx]["active"]
                
                # 获取当前队列剩余数量
                queue_remaining = task_queue.qsize()
                
                self._debug_log(
                    f"[Worker {endpoint_idx}] ▶ 开始任务 {task_id} | "
                    f"当前活跃: {current_active}/{max_concurrent} | "
                    f"队列剩余: {queue_remaining}"
                )
                
                try:
                    # 使用该worker专属的client调用API
                    sample = await self._call_api_single(prompt, client)
                    
                    # 存储结果
                    async with results_lock:
                        results[orig_idx][sample_idx] = sample
                    
                    # 更新进度条
                    async with pbar_lock:
                        pbar.update(1)
                    
                    # 更新全局完成计数
                    async with completed_lock:
                        completed_count[0] += 1
                        current_completed = completed_count[0]
                    
                    # 更新统计
                    async with stats_lock:
                        endpoint_stats[endpoint_idx]["completed"] += 1
                        endpoint_stats[endpoint_idx]["active"] -= 1
                        worker_completed = endpoint_stats[endpoint_idx]["completed"]
                    
                    # 截断sample用于显示
                    sample_preview = sample[:50] + "..." if len(sample) > 50 else sample
                    sample_preview = sample_preview.replace('\n', '\\n')
                    
                    self._debug_log(
                        f"[Worker {endpoint_idx}] ✓ 完成任务 {task_id} | "
                        f"总进度: {current_completed}/{total_tasks} | "
                        f"本Worker完成: {worker_completed} | "
                        f"响应预览: {sample_preview}"
                    )
                        
                except Exception as e:
                    async with stats_lock:
                        endpoint_stats[endpoint_idx]["active"] -= 1
                        endpoint_stats[endpoint_idx]["failed"] += 1
                    
                    self._debug_log(
                        f"[Worker {endpoint_idx}] ✗ 任务失败 {task_id} | "
                        f"错误: {str(e)[:100]}"
                    )
            
            while True:
                # 尝试填满并发槽
                slots_available = max_concurrent - len(active_tasks)
                
                tasks_fetched = 0
                for _ in range(slots_available):
                    try:
                        orig_idx, sample_idx, prompt = task_queue.get_nowait()
                        task = asyncio.create_task(
                            process_single_task(orig_idx, sample_idx, prompt)
                        )
                        active_tasks.add(task)
                        tasks_fetched += 1
                    except asyncio.QueueEmpty:
                        break
                
                if tasks_fetched > 0:
                    self._debug_log(
                        f"[Worker {endpoint_idx}] 从队列取出 {tasks_fetched} 个任务 | "
                        f"当前活跃任务数: {len(active_tasks)}"
                    )
                
                if not active_tasks:
                    # 队列空且没有活动任务，worker退出
                    self._debug_log(f"[Worker {endpoint_idx}] 队列已空，Worker 退出")
                    break
                
                # 等待任意一个任务完成
                done, active_tasks = await asyncio.wait(
                    active_tasks, 
                    return_when=asyncio.FIRST_COMPLETED
                )
                # done中的任务已完成，active_tasks更新为pending任务
        
        # 7. 创建所有端口的worker，每个worker使用对应的client
        print(f"[MultiAPIRunner] 启动 {len(self.client_pool)} 个 Worker...")
        workers = [
            endpoint_worker(i, client) 
            for i, client in enumerate(self.client_pool)
        ]
        
        # 并行运行所有worker
        await asyncio.gather(*workers)
        
        pbar.close()
        
        # 8. 打印统计信息
        print(f"\n[MultiAPIRunner] ========== 采样完成统计 ==========")
        print(f"[MultiAPIRunner] 总任务数: {total_tasks}")
        print(f"[MultiAPIRunner] 各 Client 统计:")
        total_completed = 0
        total_failed = 0
        for i in range(len(self.client_pool)):
            completed = endpoint_stats[i]['completed']
            failed = endpoint_stats[i]['failed']
            total_completed += completed
            total_failed += failed
            percentage = (completed / total_tasks * 100) if total_tasks > 0 else 0
            print(f"  [Client {i}] {self.api_bases[i]}")
            print(f"           完成: {completed} ({percentage:.1f}%) | 失败: {failed}")
        print(f"[MultiAPIRunner] 总完成: {total_completed} | 总失败: {total_failed}")
        print(f"[MultiAPIRunner] =====================================\n")
        
        # 9. 转换结果格式: Dict[int, Dict[int, str]] -> List[List[str]]
        final_results: List[List[str]] = []
        for orig_idx in range(num_prompts):
            samples = []
            for sample_idx in range(n_sample):
                sample = results[orig_idx].get(sample_idx, "")
                samples.append(sample)
            final_results.append(samples)
            
            # 调用回调
            if save_callback:
                save_callback(orig_idx, samples)
        
        return final_results

    def run_batch(
        self, 
        prompts: List[str], 
        save_callback: Optional[Callable] = None
    ) -> List[List[str]]:
        """
        批量运行推理，返回格式与VLLMRunner一致
        
        Args:
            prompts: prompt列表
            save_callback: 保存回调函数，签名为 callback(idx, samples)
            
        Returns:
            采样结果列表，results[i] = [sample_0, sample_1, ..., sample_{n_sample-1}]
        """
        def run_async_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._run_batch_async(prompts, save_callback))
            finally:
                loop.close()

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_async_in_thread)
            results = future.result()

        return results


class MultiAPIRunnerWithRetry(MultiAPIRunner):
    """
    带重试机制的MultiAPIRunner
    """
    
    def __init__(
        self,
        args,
        model: str,
        api_endpoints: List[str],
        api_key: str = "EMPTY",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        debug: bool = True,
    ):
        super().__init__(
            args, model, api_endpoints, api_key, debug
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    async def _call_api_single(
        self, 
        prompt: str, 
        client: openai.AsyncOpenAI,
    ) -> str:
        """调用API，带重试机制"""
        for attempt in range(self.max_retries):
            result = await super()._call_api_single(prompt, client)
            if result:  # 成功获取结果
                return result
            
            if attempt < self.max_retries - 1:
                self._debug_log(f"[重试] 第 {attempt + 1} 次失败，{self.retry_delay * (attempt + 1)}秒后重试...")
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        return ""
