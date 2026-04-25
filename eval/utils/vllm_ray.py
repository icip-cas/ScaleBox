import math
import os
import socket
from abc import ABC

from tqdm import tqdm

try:
    from vllm import LLM, SamplingParams
except ImportError:
    pass

try:
    import ray
except ImportError:
    ray = None


class VLLMRay(ABC):
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.model_tokenizer_path = self.args.model_path
        self.batch_size = self.args.batch_size
        self.sampling_params_kwargs = {
            "n": self.args.n_sample,
            "max_tokens": self.args.max_completion_tokens,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "top_k": self.args.top_k,
            "min_p": self.args.min_p,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": self.args.stop_token,
        }
        self.sampling_params = SamplingParams(**self.sampling_params_kwargs)

        self.num_gpus_total = args.num_gpus_total
        self.num_gpus_per_model = args.num_gpus_per_model
        self.use_npu = getattr(args, "npu", False)

        if self.num_gpus_per_model <= 0:
            raise ValueError("`num_gpus_per_model` must be greater than 0.")

        self.num_instances = self.num_gpus_total // self.num_gpus_per_model
        if self.num_instances < 1:
            raise ValueError(
                "Cannot deploy any model instance; "
                f"got num_gpus_total={self.num_gpus_total}, "
                f"num_gpus_per_model={self.num_gpus_per_model}, "
                f"num_instances={self.num_instances}."
            )

        self.use_ray = bool(getattr(args, "use_ray", False) and self.num_instances > 1)
        if self.use_ray:
            if ray is None:
                raise ImportError("`ray` is required when using `--use_ray`.")
            if self.use_npu:
                ray.init(ignore_reinit_error=True, resources={"NPU": self.num_gpus_total})
            else:
                ray.init(ignore_reinit_error=True)

    def find_n_free_ports(self, n):
        ports = []
        sockets = []
        current_port = 5000

        while len(ports) < n and current_port <= 8000:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(("", current_port))
                ports.append(current_port)
                sockets.append(sock)
            except OSError:
                sock.close()
            current_port += 1

        if len(ports) < n:
            raise RuntimeError(f"Could only find {len(ports)} free ports within the specified range.")
        return ports, sockets

    def _run_single(self, prompt: str) -> list[str]:
        pass

    @staticmethod
    def _freeze_stop_tokens(stop_tokens):
        if isinstance(stop_tokens, list):
            return tuple(stop_tokens)
        return stop_tokens

    @staticmethod
    def _build_sampling_params(sampling_params_kwargs, stop_tokens):
        return SamplingParams(**{**sampling_params_kwargs, "stop": stop_tokens})

    @staticmethod
    def _generate_grouped(model, prompts, stop_tokens_by_prompt, sampling_params_kwargs, batch_size):
        grouped_prompts = {}
        default_stop_tokens = sampling_params_kwargs["stop"]

        for index, prompt in enumerate(prompts):
            stop_tokens = default_stop_tokens if stop_tokens_by_prompt is None else stop_tokens_by_prompt[index]
            stop_tokens = default_stop_tokens if stop_tokens is None else stop_tokens
            signature = VLLMRay._freeze_stop_tokens(stop_tokens)
            if signature not in grouped_prompts:
                grouped_prompts[signature] = {
                    "stop_tokens": stop_tokens,
                    "items": [],
                }
            grouped_prompts[signature]["items"].append((index, prompt))

        outputs = [None] * len(prompts)
        for group in grouped_prompts.values():
            group_prompts = [prompt for _, prompt in group["items"]]
            sampling_params = VLLMRay._build_sampling_params(
                sampling_params_kwargs,
                group["stop_tokens"],
            )
            if batch_size == 0:
                group_outputs = model.generate(group_prompts, sampling_params)
            else:
                group_outputs = []
                for start in range(0, len(group_prompts), batch_size):
                    group_outputs.extend(
                        model.generate(group_prompts[start : start + batch_size], sampling_params)
                    )
            if len(group_outputs) != len(group["items"]):
                raise RuntimeError(
                    f"VLLMRay returned {len(group_outputs)} outputs, expected {len(group['items'])}."
                )
            for (index, _), output in zip(group["items"], group_outputs):
                outputs[index] = output
        return outputs

    @staticmethod
    def single_process_inference(
        model_path,
        num_gpus_per_model,
        vllm_port,
        prompts,
        sampling_params_kwargs,
        batch_size,
        stop_tokens_by_prompt=None,
    ):
        os.environ["VLLM_PORT"] = str(vllm_port)
        print(f"Using VLLM_PORT: {vllm_port}")

        model = LLM(
            model=model_path,
            tensor_parallel_size=num_gpus_per_model,
            trust_remote_code=True,
        )

        return VLLMRay._generate_grouped(
            model,
            prompts,
            stop_tokens_by_prompt,
            sampling_params_kwargs,
            batch_size,
        )

    def run_batch(self, prompts: list[str], stop_tokens_by_prompt=None, save_callback=None) -> list[list[str]]:
        if stop_tokens_by_prompt is not None and len(stop_tokens_by_prompt) != len(prompts):
            raise RuntimeError(
                f"Received {len(stop_tokens_by_prompt)} stop token entries, expected {len(prompts)}."
            )

        outputs = [[] for _ in prompts]
        remaining_prompts = []
        remaining_indices = []
        remaining_stop_tokens = [] if stop_tokens_by_prompt is not None else None
        pbar = tqdm(total=len(prompts), desc="Sampling", ncols=120) if prompts else None

        try:
            for prompt_index, prompt in enumerate(prompts):
                remaining_prompts.append(prompt)
                remaining_indices.append(prompt_index)
                if remaining_stop_tokens is not None:
                    remaining_stop_tokens.append(stop_tokens_by_prompt[prompt_index])

            if self.use_ray:
                if self.use_npu:
                    get_answers_func = ray.remote(resources={"NPU": self.num_gpus_per_model})(
                        VLLMRay.single_process_inference
                    ).remote
                else:
                    get_answers_func = ray.remote(num_gpus=self.num_gpus_per_model)(
                        VLLMRay.single_process_inference
                    ).remote

                num_processes = min(len(remaining_prompts), self.num_instances)
                chunk_size = math.ceil(len(remaining_prompts) / num_processes)
                ports, sockets = self.find_n_free_ports(num_processes)

                gathered_responses = []
                for idx, start in enumerate(range(0, len(remaining_prompts), chunk_size)):
                    gathered_responses.append(
                        get_answers_func(
                            self.model_tokenizer_path,
                            self.num_gpus_per_model,
                            ports[idx],
                            remaining_prompts[start : start + chunk_size],
                            self.sampling_params_kwargs,
                            self.batch_size,
                            None if remaining_stop_tokens is None else remaining_stop_tokens[start : start + chunk_size],
                        )
                    )

                for sock in sockets:
                    sock.close()

                gathered_responses = ray.get(gathered_responses)
                gathered_responses = [item for sublist in gathered_responses for item in sublist]

                if len(gathered_responses) != len(remaining_indices):
                    raise RuntimeError(
                        f"VLLMRay returned {len(gathered_responses)} outputs, expected {len(remaining_indices)}."
                    )

                for index, vllm_output in zip(remaining_indices, gathered_responses):
                    outputs[index] = [output.text for output in vllm_output.outputs]
                    if save_callback:
                        save_callback(index, outputs[index])
                    if pbar is not None:
                        pbar.update(1)
                return outputs

            if remaining_prompts:
                llm = LLM(
                    model=self.model_tokenizer_path,
                    tokenizer=self.model_tokenizer_path,
                    tensor_parallel_size=self.num_gpus_per_model,
                    trust_remote_code=True,
                )
                vllm_outputs = self._generate_grouped(
                    llm,
                    remaining_prompts,
                    remaining_stop_tokens,
                    self.sampling_params_kwargs,
                    self.batch_size,
                )
                if len(vllm_outputs) != len(remaining_indices):
                    raise RuntimeError(
                        f"VLLMRay returned {len(vllm_outputs)} outputs, expected {len(remaining_indices)}."
                    )
                for index, vllm_output in zip(remaining_indices, vllm_outputs):
                    outputs[index] = [output.text for output in vllm_output.outputs]
                    if save_callback:
                        save_callback(index, outputs[index])
                    if pbar is not None:
                        pbar.update(1)
            return outputs
        finally:
            if pbar is not None:
                pbar.close()

