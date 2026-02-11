import asyncio
import aiohttp
import json
import time
from collections import deque
from typing import List, Optional
from abc import ABC
import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm.asyncio import tqdm
import traceback
try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


class APIRunner(ABC):
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.api_url = args.api_url if hasattr(args, 'api_url') else "http://localhost:8000/v1/completions"
        self.api_key = args.api_key if hasattr(args, 'api_key') else "EMPTY"
        self.model_name = args.model_name if hasattr(args, 'model_name') else "model"
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

        self._rpm = getattr(self.args, "rpm", 0) or 0
        self._rate_limiter = _AsyncRateLimiter(self._rpm) if self._rpm > 0 else None

    def _get_tokenizer(self):
        if self._tokenizer is not None or AutoTokenizer is None:
            return self._tokenizer
        model_path = getattr(self.args, 'model_path', None) or getattr(self.args, 'model_name', None)
        if not model_path:
            return None
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            self._tokenizer = None
        return self._tokenizer

    def _count_prompt_tokens(self, messages_or_text):
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            return None
        try:
            if isinstance(messages_or_text, list):
                input_ids = tokenizer.apply_chat_template(
                    messages_or_text, tokenize=True, add_generation_prompt=True
                )
                return len(input_ids)
            return len(tokenizer.encode(messages_or_text))
        except Exception:
            return None

    def _adjust_max_tokens(self, messages_or_text, max_tokens: int) -> int:
        if not self._max_context_len:
            return max_tokens
        input_tokens = self._count_prompt_tokens(messages_or_text)
        if input_tokens is None:
            return max_tokens
        available = self._max_context_len - input_tokens
        if available < 1:
            available = 1
        return min(max_tokens, available)

    def _truncate(self, s: str, n: int = 2000):
        if not isinstance(s, str):
            return s
        return s if len(s) <= n else s[:n] + f"...(剩余{len(s)-n}字节已截断)"

    def _sanitize_payload(self, payload: dict):
        safe = dict(payload)
        if "prompt" in safe and isinstance(safe["prompt"], str):
            safe["prompt"] = self._truncate(safe["prompt"], 500)
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

        print(f"API调用异常 - {title}:\n{json.dumps(meta, ensure_ascii=False, indent=2)}")

    def _extract_segment(self, prompt: str, start_token: str, end_token: Optional[str]) -> str:
        if not start_token:
            return ""
        start_idx = prompt.find(start_token)
        if start_idx == -1:
            return ""
        start_idx += len(start_token)
        if end_token:
            end_idx = prompt.find(end_token, start_idx)
            if end_idx != -1:
                return prompt[start_idx:end_idx].strip()
        return prompt[start_idx:].strip()

    def _parse_prompt_to_messages(self, prompt: str) -> List[dict]:
        """将包含角色标记的prompt解析为chat API的messages格式"""
        prompt_type = getattr(self.args, 'prompt_type', None)
        if prompt_type:
            try:
                from utils.template import TEMPLATES, Role
            except Exception:
                TEMPLATES = None
                Role = None

            if TEMPLATES and Role and prompt_type in TEMPLATES:
                template = TEMPLATES[prompt_type]
                role_starts = template.role_starts or {}
                role_ends = template.role_ends or {}

                system_start = role_starts.get(Role.SYSTEM)
                human_start = role_starts.get(Role.HUMAN)
                assistant_start = role_starts.get(Role.ASSISTANT)
                system_end = role_ends.get(Role.SYSTEM) if role_ends else None
                human_end = role_ends.get(Role.HUMAN) if role_ends else None

                messages: List[dict] = []

                if system_start and system_start in prompt:
                    end_token = system_end if system_end else human_start
                    system_content = self._extract_segment(prompt, system_start, end_token)
                    if system_content:
                        messages.append({"role": "system", "content": system_content})

                if human_start and human_start in prompt:
                    end_token = human_end if human_end else assistant_start
                    human_content = self._extract_segment(prompt, human_start, end_token)
                    if human_content:
                        messages.append({"role": "user", "content": human_content})

                return messages

        return []

    async def _call_api_once(self, prompt: str, session: aiohttp.ClientSession) -> List[str]:
        """调用一次API获取响应"""
        is_chat_api = "/chat/completions" in self.api_url
        
        if is_chat_api:
            messages = self._parse_prompt_to_messages(prompt)
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self._adjust_max_tokens(messages, self.args.max_completion_tokens),
                "temperature": self.args.temperature,
                "top_p": self.args.top_p,
                "top_k": self.args.top_k if hasattr(self.args, 'top_k') else 20,
                "n": self.args.n_sample,
                "stop": None if self.args.n_sample > 1 else self.stop_tokens,
                "stream": False
            }
            if (
                getattr(self.args, "prompt_type", None) == "chatml_qwen3"
                and getattr(self.args, "reasoning_model", False)
            ):
                payload["enable_thinking"] = True
        else:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": self._adjust_max_tokens(prompt, self.args.max_completion_tokens),
                "temperature": self.args.temperature,
                "top_p": self.args.top_p,
                "top_k": self.args.top_k if hasattr(self.args, 'top_k') else 20,
                "n": self.args.n_sample,
                "stop": None if self.args.n_sample > 1 else self.stop_tokens,
                "stream": False
            }
        
        if hasattr(self.args, 'min_p') and self.args.min_p > 0:
            payload["min_p"] = self.args.min_p

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        try:
            async with session.post(self.api_url, json=payload, headers=headers) as response:
                text = await response.text()
                if response.status == 200:
                    try:
                        result = json.loads(text)
                        if is_chat_api:
                            return [choice.get("message", {}).get("content", "") for choice in result.get("choices", [])]
                        else:
                            return [choice.get("text", "") for choice in result.get("choices", [])]
                    except json.JSONDecodeError as je:
                        self._log_error(
                            "响应JSON解析失败",
                            url=response.url,
                            payload=payload,
                            status=response.status,
                            reason=response.reason,
                            headers=response.headers,
                            body_text=text,
                            exc=je
                        )
                        return []
                else:
                    err_msg = None
                    try:
                        err_json = json.loads(text)
                        if isinstance(err_json, dict):
                            err_msg = err_json.get("error") or err_json.get("message")
                    except Exception:
                        pass

                    self._log_error(
                        "HTTP非200响应",
                        url=response.url,
                        payload=payload,
                        status=response.status,
                        reason=response.reason,
                        headers=response.headers,
                        body_text=err_msg or text
                    )
                    return []

        except asyncio.TimeoutError as e:
            self._log_error("请求超时", url=self.api_url, payload=payload, exc=e)
            return []

        except aiohttp.ClientError as e:
            self._log_error("aiohttp客户端错误", url=self.api_url, payload=payload, exc=e)
            return []

        except Exception as e:
            self._log_error("未知异常", url=self.api_url, payload=payload, exc=e)
            return []

    async def _run_batch_async(self, prompts: List[str], save_callback=None) -> List[List[str]]:
        """批量异步采样，每个prompt采样n_sample次"""
        timeout = aiohttp.ClientTimeout(total=getattr(self.args, 'timeout', 3600))
        max_concurrency = getattr(self.args, 'max_concurrency', 2)
        connector = aiohttp.TCPConnector(limit=max_concurrency)

        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            sem = asyncio.Semaphore(max_concurrency)
            
            async def sample_once(idx: int) -> tuple:
                async with sem:
                    if self._rate_limiter is not None:
                        await self._rate_limiter.acquire()
                    result = await self._call_api_once(prompts[idx], session)
                    return idx, result
            
            tasks = [sample_once(idx) for idx in range(len(prompts))]
            
            batch_results = await tqdm.gather(
                *tasks,
                desc="Sampling",
                ncols=120
            )
            
            # 按索引排序并构建结果
            results = [None] * len(prompts)
            for idx, samples in batch_results:
                results[idx] = samples
                if save_callback:
                    save_callback(idx, samples)
            
            return results

    def run_batch(self, prompts: List[str], save_callback=None) -> List[List[str]]:
        """批量运行推理，返回格式与VLLMRunner一致"""
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


class _AsyncRateLimiter:
    def __init__(self, rpm: int):
        self._rpm = max(int(rpm), 1)
        self._lock = None
        self._timestamps = deque()

    async def acquire(self) -> None:
        if self._lock is None:
            self._lock = asyncio.Lock()
        while True:
            async with self._lock:
                now = time.monotonic()
                window_start = now - 60.0
                while self._timestamps and self._timestamps[0] <= window_start:
                    self._timestamps.popleft()
                if len(self._timestamps) < self._rpm:
                    self._timestamps.append(now)
                    return
                wait_for = self._timestamps[0] + 60.0 - now
            if wait_for > 0:
                await asyncio.sleep(wait_for)
            else:
                await asyncio.sleep(0)
