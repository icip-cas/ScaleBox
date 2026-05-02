import argparse

def build_arg_parser():
    # Build the CLI argument parser.
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--benchmark_data_path", type=str, default=None, help="Path to the benchmark source data; required for normal runs and also required in eval_only mode")
    argparser.add_argument("--eval_path", type=str, default=None, help="Result file path used only in eval_only mode; must be a JSONL with id/prompt/response")
    argparser.add_argument("--config", type=str, required=True, help="Single-benchmark YAML config file")
    argparser.add_argument("--model_path", type=str, default=None, help="Model path for generation; required for use_server/use_ray")
    argparser.add_argument("--endpoint", type=str, default="http://0.0.0.0:8080", help="ScaleBox sandbox endpoint")
    argparser.add_argument("--prompt_type", type=str, default="qwen2.5-instruct", help="Prompt template type")
    argparser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    argparser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    argparser.add_argument("--top_k", type=int, default=20, help="Top-k sampling parameter")
    argparser.add_argument("--min_p", type=float, default=0.0, help="Min-p sampling parameter")
    argparser.add_argument("--max_completion_tokens", type=int, default=8192, help="Maximum generated tokens per response")
    argparser.add_argument("--max_model_len", type=int, default=None, help="Maximum context length for vLLM server")
    argparser.add_argument("--n_sample", type=int, default=1, help="Number of generations for each sample")
    argparser.add_argument("--stop_token", type=str, default=None, help="Stop token for the model; defaults to the prompt template stop_str when omitted")
    argparser.add_argument("--num_gpus_total", type=int, default=1, help="Total number of GPUs/NPUs")
    argparser.add_argument("--num_gpus_per_model", type=int, default=1, help="Number of GPUs/NPUs per model instance")
    argparser.add_argument("--npu", action="store_true", default=False, help="Use NPU instead of GPU")
    argparser.add_argument("--thinking", action="store_true", default=False, help="Only supported for qwen3; enables Qwen3 think mode")
    argparser.add_argument("--output_dir", type=str, default="res/multi_language", help="Base output directory; results are saved under a timestamped subdirectory")
    argparser.add_argument("--batch_size", type=int, default=0, help="Batch size for the model")
    argparser.add_argument("--model_name", type=str, default="model", help="Model name used by the sampling service")
    argparser.add_argument("--sample_only", action="store_true", default=False, help="Only sample without evaluation")
    argparser.add_argument("--eval_only", action="store_true", default=False, help="Only evaluate existing responses")
    argparser.add_argument("--use_server", action="store_true", default=False, help="Start local vLLM server instances for sampling")
    argparser.add_argument("--use_ray", action="store_true", default=False, help="Use ray with local vLLM for parallel sampling")
    argparser.add_argument("--vllm_server_base_port", type=int, default=8000, help="Base port for vLLM servers")
    argparser.add_argument("--vllm_server_host", type=str, default="0.0.0.0", help="Host for vLLM servers")
    argparser.add_argument("--vllm_server_dtype", type=str, default="auto", help="Data type for vLLM server")
    argparser.add_argument("--vllm_server_wait_timeout", type=int, default=600, help="Timeout for waiting vLLM server")
    argparser.add_argument("--mem_fraction", type=float, default=0.9, help="GPU/NPU memory utilization fraction")
    argparser.add_argument("--resume_sample", action="store_true", default=False, help="Resume sampling from an existing samples file")
    argparser.add_argument("--resume_sample_path", type=str, default=None, help="Path to the samples.jsonl file used for resume sampling")
    argparser.add_argument("--save_full_scalebox_result", action="store_true", default=False, help="Save the full ScaleBox response body in results instead of the summarized score")
    return argparser

def parse_args():
    # Parse CLI arguments and apply basic validation rules.
    argparser = build_arg_parser()
    args = argparser.parse_args()
    if args.sample_only and args.eval_only:
        argparser.error("--sample_only and --eval_only cannot be used together")
    args._defaults = {action.dest: action.default for action in argparser._actions if action.dest != "help"}
    return args
