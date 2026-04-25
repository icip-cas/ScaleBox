"""
MultiAPIRunner - dynamic load-balanced sampling with multiple server endpoints

Features:
1. Repeat each prompt n_sample times as independent tasks
2. Multiple endpoints share one task queue
3. Each endpoint has a max concurrency limit
4. Fetch the next task immediately after one completes (dynamic scheduling)
"""

import asyncio
import json
from typing import List, Callable, Optional, Dict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import traceback
import random
import openai
from datetime import datetime

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

class MultiAPIRunner:
    """
    Dynamic load-balanced sampling runner for multiple server endpoints.
    
    Workflow:
    1. Expand prompts: each prompt is repeated n_sample times with (orig_idx, sample_idx)
    2. Put all expanded tasks into a shared queue
    3. Each endpoint runs a worker with concurrency limit (batch_size)
    4. Workers continuously pull tasks; each finished task immediately pulls the next
    5. Aggregate results by (orig_idx, sample_idx)
    
    Example:
    ```python
    runner = MultiAPIRunner(
        args=args,  # args.batch_size controls max concurrency per server
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
        debug: bool = True,  # Whether to enable debug logs
    ):
        """
        Initialize MultiAPIRunner.
        
        Args:
            args: Object with sampling params (n_sample, temperature, top_p, batch_size, etc.)
            model: Model name used by the sampling service
            api_endpoints: Server endpoint list (e.g. ["http://localhost:8000/v1", ...])
            debug: Whether to enable debug logs
        """
        self.args = args
        self.batch_size = getattr(args, 'batch_size', 16) or 16  # Use batch_size as max concurrency per server
        self.timeout = getattr(args, 'timeout', 60000)
        self.debug = debug  # Debug switch
        
        # Model name
        self.model_name = getattr(args, 'model_name', model) or model
        # Reserved: used for counting input tokens (optional)
        self._tokenizer = None
        self._max_context_len = (
            getattr(args, 'max_model_len', None)
            or getattr(args, 'max_context_len', None)
            or getattr(args, 'max_completion_tokens', None)
        )
        
        # Build stop token list
        stop_attr = getattr(self.args, 'stop_token', None)
        if isinstance(stop_attr, str):
            self.stop_tokens = [token.strip() for token in stop_attr.split(',') if token.strip()]
        elif stop_attr:
            self.stop_tokens = stop_attr
        else:
            self.stop_tokens = []
        
        # Create client pool for local server endpoints
        # Use AsyncOpenAI async clients
        self.client_pool = []
        self.api_bases = []  # Keep normalized base_url values for logging
        api_bases = api_endpoints if isinstance(api_endpoints, list) else [api_endpoints]
        api_bases = [base_url for base_url in api_bases if base_url]
        if not api_bases:
            raise ValueError("`api_endpoints` must not be empty.")
        for base_url in api_bases:
            # Ensure base_url format is correct
            if not base_url.endswith('/v1'):
                if base_url.endswith('/'):
                    base_url = base_url + 'v1'
                else:
                    base_url = base_url + '/v1' if '/v1' not in base_url else base_url
            
            self.api_bases.append(base_url)
            self.client_pool.append(openai.AsyncOpenAI(
                base_url=base_url,
                api_key="EMPTY",
                timeout=self.timeout
            ))
        
        print(f"[MultiAPIRunner] Initialized {len(self.client_pool)} clients | debug={self.debug} | concurrency/client={self.batch_size}")
        for i, ep in enumerate(self.api_bases):
            print(f"  [Client {i}] {ep}")
    
    def _get_tokenizer(self):
        if self._tokenizer is not None or AutoTokenizer is None:
            return self._tokenizer
        model_path = getattr(self.args, 'model_path', None) or getattr(self.args, 'model_name', None)
        if not model_path:
            return None
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception as e:
            self._debug_log(f"[MultiAPIRunner] Failed to load tokenizer: {e}")
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
                f"[MultiAPIRunner] Dynamic max_tokens: {max_tokens} -> {available} "
                f"(input={input_tokens}, context={self._max_context_len})"
            )
        return min(max_tokens, available)

    def _debug_log(self, message: str):
        """Debug log with timestamp."""
        if self.debug:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"[{timestamp}] {message}")
    
    def _truncate(self, s: str, n: int = 2000) -> str:
        if not isinstance(s, str):
            return s
        return s if len(s) <= n else s[:n] + f"...({len(s)-n} bytes truncated)"

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

        print(f"[MultiAPIRunner] API call error - {title}:\n{json.dumps(meta, ensure_ascii=False, indent=2)}")

    def _resolve_stop_tokens(self, stop_tokens=None):
        return self.stop_tokens if stop_tokens is None else stop_tokens

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
        stop_tokens=None,
    ) -> str:
        """
        Call the API asynchronously with an AsyncOpenAI client (aligned with the reference logic).
        
        Args:
            client: AsyncOpenAI client instance
            model: Model name
            user_prompt: User input
            max_tokens: Max token count
            temperature: Temperature
            top_p: Top-p
            n: Number of generations
            stream: Whether to stream output
            
        Returns:
            Generated text content
        """
        try:
            prompt_text = user_prompt if user_prompt is not None else ""
            
            # Dynamically adjust max_tokens by input length to avoid context overflow
            max_tokens = self._adjust_max_tokens(prompt_text, max_tokens)

            # Build request params (standard OpenAI fields)
            kwargs = {
                "model": model,
                "prompt": prompt_text,
                "max_tokens": max_tokens,
                "n": n,
                "stream": stream,
            }
            
            # Add custom params
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p
            
            # Add stop tokens
            resolved_stop_tokens = self._resolve_stop_tokens(stop_tokens)
            if resolved_stop_tokens:
                kwargs["stop"] = resolved_stop_tokens
            
            # Pass non-standard OpenAI params through extra_body (e.g. top_k, min_p supported by vLLM)
            extra_body = {}
            if hasattr(self.args, 'top_k') and self.args.top_k > 0:
                extra_body["top_k"] = self.args.top_k
            
            if hasattr(self.args, 'min_p') and self.args.min_p > 0:
                extra_body["min_p"] = self.args.min_p
            
            if extra_body:
                kwargs["extra_body"] = extra_body
            
            # Call API asynchronously
            response = await client.completions.create(**kwargs)
            
            # Extract response text
            if response.choices:
                return response.choices[0].text or ""
            return ""
            
        except Exception as e:
            self._log_error("AsyncOpenAI API call failed", exc=e)
            return ""

    async def _call_api_single(
        self, 
        prompt: str, 
        client: openai.AsyncOpenAI,
        stop_tokens=None,
    ) -> str:
        """
        Call the API once asynchronously with the specified client.
        (Aligned with the reference logic, but the client is chosen by endpoint_worker.)
        
        Returns:
            Generated text, or empty string on failure
        """
        # Call get_openai_response (aligned with the reference logic)
        res = await self.get_openai_response(
            client=client,
            model=self.model_name,
            user_prompt=prompt,
            max_tokens=self.args.max_completion_tokens,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            n=1,
            stream=False,
            stop_tokens=stop_tokens,
        )
        
        return res

    async def _run_batch_async(
        self, 
        prompts: List[str], 
        save_callback: Optional[Callable] = None,
        stop_tokens_by_prompt: Optional[List] = None,
    ) -> List[List[str]]:
        """
        Batched async sampling with dynamic load balancing.
        
        Workflow:
        1. Repeat each prompt n_sample times and create (orig_idx, sample_idx, prompt) tasks
        2. Put all tasks into a shared queue
        3. Each endpoint worker pulls tasks and immediately fetches the next after completion
        4. Aggregate results by original prompt index
        """
        n_sample = self.args.n_sample
        num_prompts = len(prompts)
        total_tasks = num_prompts * n_sample
        
        print(
            f"[MultiAPIRunner] Start batch | prompts={num_prompts} | "
            f"n_sample={n_sample} | total_tasks={total_tasks} | "
            f"clients={len(self.client_pool)} | concurrency/client={self.batch_size} | "
            f"max_total={len(self.client_pool) * self.batch_size}"
        )
        
        if stop_tokens_by_prompt is not None and len(stop_tokens_by_prompt) != num_prompts:
            raise RuntimeError(
                f"Received {len(stop_tokens_by_prompt)} stop token entries, expected {num_prompts}."
            )

        # 1. Expand prompts: repeat each prompt n_sample times
        # Format: (orig_idx, sample_idx, prompt, stop_tokens)
        task_list = [
            (
                orig_idx,
                sample_idx,
                prompt,
                stop_tokens_by_prompt[orig_idx] if stop_tokens_by_prompt is not None else None,
            )
            for orig_idx, prompt in enumerate(prompts)
            for sample_idx in range(n_sample)
        ]

        # Shuffle task order to distribute easy/hard tasks more evenly
        random.shuffle(task_list)
        self._debug_log(f"[Queue] Created and shuffled {len(task_list)} tasks")

        task_queue = asyncio.Queue()
        for task in task_list:
            await task_queue.put(task)
        
        self._debug_log(f"[Queue] All tasks enqueued, queue size: {task_queue.qsize()}")
        
        # 2. Result storage: results[orig_idx][sample_idx] = sample_text
        results: Dict[int, Dict[int, str]] = {i: {} for i in range(num_prompts)}
        results_lock = asyncio.Lock()
        
        # 3. Progress bar
        pbar = tqdm(total=total_tasks, desc="Sampling", ncols=120)
        pbar_lock = asyncio.Lock()
        
        # 4. Endpoint stats
        endpoint_stats = {i: {"completed": 0, "active": 0, "failed": 0} for i in range(len(self.client_pool))}
        stats_lock = asyncio.Lock()
        
        # 5. Global counter
        completed_count = [0]  # Use a list so it can be modified inside closures
        completed_lock = asyncio.Lock()
        
        # 6. Define endpoint worker (each endpoint uses a dedicated client)
        async def endpoint_worker(endpoint_idx: int, client: openai.AsyncOpenAI):
            """
            Worker coroutine for a single endpoint.
            - Keeps up to batch_size concurrent tasks
            - Pulls a new task immediately after one finishes
            - Uses a dedicated AsyncOpenAI client
            """
            active_tasks: set = set()
            max_concurrent = self.batch_size
            
            self._debug_log(f"[Worker {endpoint_idx}] Started, endpoint: {self.api_bases[endpoint_idx]}")
            
            async def process_single_task(orig_idx: int, sample_idx: int, prompt: str, stop_tokens):
                """Process a single sampling task."""
                task_id = f"prompt_{orig_idx}_sample_{sample_idx}"
                
                # Update stats
                async with stats_lock:
                    endpoint_stats[endpoint_idx]["active"] += 1
                    current_active = endpoint_stats[endpoint_idx]["active"]
                
                # Get current remaining queue size
                queue_remaining = task_queue.qsize()
                
                self._debug_log(
                    f"[Worker {endpoint_idx}] ▶ Start task {task_id} | "
                    f"active: {current_active}/{max_concurrent} | "
                    f"queue remaining: {queue_remaining}"
                )
                
                try:
                    # Call API with this worker's dedicated client
                    sample = await self._call_api_single(prompt, client, stop_tokens=stop_tokens)
                    
                    # Store result
                    async with results_lock:
                        results[orig_idx][sample_idx] = sample
                    
                    # Update progress bar
                    async with pbar_lock:
                        pbar.update(1)
                    
                    # Update global completion counter
                    async with completed_lock:
                        completed_count[0] += 1
                        current_completed = completed_count[0]
                    
                    # Update stats
                    async with stats_lock:
                        endpoint_stats[endpoint_idx]["completed"] += 1
                        endpoint_stats[endpoint_idx]["active"] -= 1
                        worker_completed = endpoint_stats[endpoint_idx]["completed"]
                    
                    # Truncate sample for display
                    sample_preview = sample[:50] + "..." if len(sample) > 50 else sample
                    sample_preview = sample_preview.replace('\n', '\\n')
                    
                    self._debug_log(
                        f"[Worker {endpoint_idx}] ✓ Completed task {task_id} | "
                        f"overall progress: {current_completed}/{total_tasks} | "
                        f"worker completed: {worker_completed} | "
                        f"response preview: {sample_preview}"
                    )
                        
                except Exception as e:
                    async with stats_lock:
                        endpoint_stats[endpoint_idx]["active"] -= 1
                        endpoint_stats[endpoint_idx]["failed"] += 1
                    
                    self._debug_log(
                        f"[Worker {endpoint_idx}] ✗ Task failed {task_id} | "
                        f"error: {str(e)[:100]}"
                    )
            
            while True:
                # Try to fill available concurrency slots
                slots_available = max_concurrent - len(active_tasks)
                
                tasks_fetched = 0
                for _ in range(slots_available):
                    try:
                        orig_idx, sample_idx, prompt, stop_tokens = task_queue.get_nowait()
                        task = asyncio.create_task(
                            process_single_task(orig_idx, sample_idx, prompt, stop_tokens)
                        )
                        active_tasks.add(task)
                        tasks_fetched += 1
                    except asyncio.QueueEmpty:
                        break
                
                if tasks_fetched > 0:
                    self._debug_log(
                        f"[Worker {endpoint_idx}] Fetched {tasks_fetched} tasks from queue | "
                        f"current active tasks: {len(active_tasks)}"
                    )
                
                if not active_tasks:
                    # Queue empty and no active tasks, worker exits
                    self._debug_log(f"[Worker {endpoint_idx}] Queue empty, worker exits")
                    break
                
                # Wait until any active task completes
                _, active_tasks = await asyncio.wait(
                    active_tasks, 
                    return_when=asyncio.FIRST_COMPLETED
                )
                # Tasks in done are finished; active_tasks now contains pending tasks
        
        # 7. Create workers for all endpoints, each with its corresponding client
        workers = [
            endpoint_worker(i, client) 
            for i, client in enumerate(self.client_pool)
        ]
        
        # Run all workers in parallel
        await asyncio.gather(*workers)
        
        pbar.close()
        
        # 8. Print summary stats
        total_completed = 0
        total_failed = 0
        summary_lines = []
        for i in range(len(self.client_pool)):
            completed = endpoint_stats[i]['completed']
            failed = endpoint_stats[i]['failed']
            total_completed += completed
            total_failed += failed
            percentage = (completed / total_tasks * 100) if total_tasks > 0 else 0
            summary_lines.append(
                f"  [Client {i}] {self.api_bases[i]} | completed={completed} ({percentage:.1f}%) | failed={failed}"
            )
        print(
            f"[MultiAPIRunner] Sampling summary | total_tasks={total_tasks} | "
            f"completed={total_completed} | failed={total_failed}"
        )
        for line in summary_lines:
            print(line)
        # 9. Convert result format: Dict[int, Dict[int, str]] -> List[List[str]]
        final_results: List[List[str]] = []
        for orig_idx in range(num_prompts):
            samples = []
            for sample_idx in range(n_sample):
                sample = results[orig_idx].get(sample_idx, "")
                samples.append(sample)
            final_results.append(samples)

            # Call callback
            if save_callback:
                save_callback(orig_idx, samples)

        if len(final_results) != num_prompts:
            raise RuntimeError(
                f"MultiAPIRunner returned {len(final_results)} prompt results, expected {num_prompts}."
            )
        for orig_idx, samples in enumerate(final_results):
            if not isinstance(samples, list):
                raise RuntimeError(
                    f"MultiAPIRunner result at index {orig_idx} must be a list, got {type(samples).__name__}."
                )
            if not all(isinstance(sample, str) for sample in samples):
                raise RuntimeError(
                    f"MultiAPIRunner samples at index {orig_idx} must all be strings."
                )

        return final_results

    def run_batch(
        self, 
        prompts: List[str], 
        stop_tokens_by_prompt: Optional[List] = None,
        save_callback: Optional[Callable] = None
    ) -> List[List[str]]:
        """
        Run batched inference with output format aligned to VLLMRay.
        
        Args:
            prompts: Prompt list
            save_callback: Save callback function, signature: callback(idx, samples)
            
        Returns:
            Sampling result list, results[i] = [sample_0, sample_1, ..., sample_{n_sample-1}]
        """
        def run_async_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self._run_batch_async(prompts, save_callback, stop_tokens_by_prompt)
                )
            finally:
                loop.close()

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_async_in_thread)
            results = future.result()

        return results


class MultiAPIRunnerWithRetry(MultiAPIRunner):
    """
    MultiAPIRunner with retry support.
    """
    
    def __init__(
        self,
        args,
        model: str,
        api_endpoints: List[str],
        max_retries: int = 3,
        retry_delay: float = 1.0,
        debug: bool = True,
    ):
        super().__init__(
            args, model, api_endpoints, debug
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    async def _call_api_single(
        self, 
        prompt: str, 
        client: openai.AsyncOpenAI,
        stop_tokens=None,
    ) -> str:
        """Call API with retry support."""
        for attempt in range(self.max_retries):
            result = await super()._call_api_single(prompt, client)
            if result:  # Successfully got a result
                return result
            
            if attempt < self.max_retries - 1:
                self._debug_log(f"[Retry] Attempt {attempt + 1} failed, retrying in {self.retry_delay * (attempt + 1)}s...")
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        return ""
