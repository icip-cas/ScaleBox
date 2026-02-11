import os
import math
try:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
except ImportError as e:
    # print("Cannot import vllm")
    pass

from abc import ABC
import ray
import socket

class VLLMRunner(ABC):
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.model_tokenizer_path = self.args.model_path
        self.batch_size = self.args.batch_size
        # self.llm = LLM(
        #     model=model_tokenizer_path,
        #     tokenizer=model_tokenizer_path,
        #     tensor_parallel_size=args.tensor_parallel_size,
        #     dtype=args.dtype,
        #     enforce_eager=True,
        #     disable_custom_all_reduce=True,
        #     enable_prefix_caching=args.enable_prefix_caching,
        #     trust_remote_code=args.trust_remote_code,
        # )
        self.sampling_params = SamplingParams(
            n=self.args.n_sample,
            max_tokens=self.args.max_completion_tokens,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            min_p=self.args.min_p,
            frequency_penalty=0,
            presence_penalty=0,
            stop=self.args.stop_token,
        )

#############dp#############
        self.num_gpus_total = args.num_gpus_total
        self.num_gpus_per_model = args.num_gpus_per_model
        self.use_npu = getattr(args, 'npu', False)  # 是否使用NPU

        self.use_ray = False
        if self.num_gpus_total // self.num_gpus_per_model > 1:
            self.use_ray = True
            if self.use_npu:
                # NPU模式：注册NPU资源
                ray.init(ignore_reinit_error=True, resources={"NPU": self.num_gpus_total})
            else:
                # GPU模式：默认行为
                ray.init(ignore_reinit_error=True)


#############dp#############
    def find_n_free_ports(self, n):
        ports = []
        sockets = []
        port_range_start = 5000
        port_range_end = 8000
        current_port = port_range_start

        while len(ports) < n and current_port <= port_range_end:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.bind(("", current_port))
                ports.append(current_port)
                sockets.append(s)
            except OSError:
                # 如果端口已经被占用，继续尝试下一个端口
                pass
            current_port += 1

        if len(ports) < n:
            raise RuntimeError(
                f"Could only find {len(ports)} free ports within the specified range."
            )

        return ports, sockets

    def _run_single(self, prompt: str) -> list[str]:
        pass

#############dp#############
    @staticmethod
    def single_process_inference(
        model_path, num_gpus_per_model, vllm_port, *args, **kwargs
    ):

        # Set an available port
        os.environ["VLLM_PORT"] = str(vllm_port)
        print(f"Using VLLM_PORT: {vllm_port}")

        model = LLM(
            model=model_path,
            tensor_parallel_size=num_gpus_per_model,
            trust_remote_code=True,
        )
        
        if args[2] == 0:
            return model.generate(args[0], args[1], **kwargs)
        else:
            # args[0]就是分块数据
            response = []
            for i in range(0, len(args[0]), args[2]):
                response.extend(model.generate(args[0][i:i+args[2]],args[1], **kwargs))
            return response

    def run_batch(self, prompts: list[str], save_callback=None) -> list[list[str]]: 
        outputs = [None for _ in prompts]
        remaining_prompts = []
        remaining_indices = []
        for prompt_index, prompt in enumerate(prompts):
            remaining_prompts.append(prompt)
            remaining_indices.append(prompt_index)

#############dp#############
        if self.use_ray:
            if self.use_npu:
                # NPU模式：使用NPU资源
                get_answers_func = ray.remote(resources={"NPU": self.num_gpus_per_model})(
                    VLLMRunner.single_process_inference
                ).remote
            else:
                # GPU模式：使用GPU资源
                get_answers_func = ray.remote(num_gpus=self.num_gpus_per_model)(
                    VLLMRunner.single_process_inference
                ).remote

            num_processes = min(
                len(remaining_prompts), max(1, self.num_gpus_total // self.num_gpus_per_model)
            )
            chunk_size = math.ceil(len(remaining_prompts) / num_processes)

            ports, sockets = self.find_n_free_ports(num_processes)

            gathered_responses = []
            for idx, i in enumerate(range(0, len(remaining_prompts), chunk_size)):
                gathered_responses.append(
                    get_answers_func(
                        self.model_tokenizer_path,
                        self.num_gpus_per_model,
                        ports[idx],
                        remaining_prompts[i : i + chunk_size],
                        self.sampling_params,
                        self.batch_size,
                    )
                )
            
            for s in sockets:
                s.close()

            gathered_responses = ray.get(gathered_responses)

            gathered_responses = [
                item for sublist in gathered_responses for item in sublist
            ]

            for index, vllm_output in zip(remaining_indices, gathered_responses):
                outputs[index] = [o.text for o in vllm_output.outputs]
                if save_callback:
                    save_callback(index, outputs[index])
            
            return outputs

        else:
            if remaining_prompts:
                llm = LLM(
                    model=self.model_tokenizer_path,
                    tokenizer=self.model_tokenizer_path,
                    tensor_parallel_size=self.num_gpus_per_model,
                    trust_remote_code=True,
                )
                vllm_outputs = llm.generate(remaining_prompts, self.sampling_params)
                for index, vllm_output in zip(remaining_indices, vllm_outputs):
                    outputs[index] = [o.text for o in vllm_output.outputs] # 保存的是n次采样的结果，一共n个结果
                    if save_callback:
                        save_callback(index, outputs[index])
            return outputs
