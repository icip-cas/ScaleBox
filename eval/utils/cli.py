import argparse


def build_arg_parser():
    """中文：构建命令行参数解析器。
    English: Build the CLI argument parser.
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--benchmark_data_path", type=str, default=None, help="Benchmark 原始数据路径；非 eval_only 时必填，eval_only 时也必须提供 / Path to the benchmark source data; required for normal runs and also required in eval_only mode")
    argparser.add_argument("--eval_path", type=str, default=None, help="仅在 eval_only 时使用的结果文件路径；必须是包含 id/prompt/response 的 JSONL / Result file path used only in eval_only mode; must be a JSONL with id/prompt/response")
    argparser.add_argument("--config", type=str, required=True, help="单 benchmark 的 YAML 配置文件 / Single-benchmark YAML config file")
    argparser.add_argument("--model_path", type=str, default=None, help="采样模型路径；use_server/use_ray 时必填 / Model path for generation; required for use_server/use_ray")
    argparser.add_argument("--endpoint", type=str, default="http://0.0.0.0:8080", help="ScaleBox sandbox 接口地址 / ScaleBox sandbox endpoint")
    argparser.add_argument("--prompt_type", type=str, default="qwen2.5-instruct", help="Prompt 模版类型 / Prompt template type")
    argparser.add_argument("--temperature", type=float, default=0.6, help="采样温度 / Sampling temperature")
    argparser.add_argument("--top_p", type=float, default=0.95, help="Top-p 采样参数 / Top-p sampling parameter")
    argparser.add_argument("--top_k", type=int, default=20, help="Top-k 采样参数 / Top-k sampling parameter")
    argparser.add_argument("--min_p", type=float, default=0.0, help="Min-p 采样参数 / Min-p sampling parameter")
    argparser.add_argument("--max_completion_tokens", type=int, default=8192, help="单次生成的最大 token 数 / Maximum generated tokens per response")
    argparser.add_argument("--max_model_len", type=int, default=None, help="vLLM Server 的最大上下文长度 / Maximum context length for vLLM server")
    argparser.add_argument("--n_sample", type=int, default=1, help="每条样本生成次数 / Number of generations for each sample")
    argparser.add_argument("--stop_token", type=str, default=None, help="模型停止词；为空时自动使用 prompt 模版自带的 stop_str / Stop token for the model; defaults to the prompt template stop_str when omitted")
    argparser.add_argument("--num_gpus_total", type=int, default=1, help="GPU/NPU 总数 / Total number of GPUs/NPUs")
    argparser.add_argument("--num_gpus_per_model", type=int, default=1, help="每个模型实例使用的 GPU/NPU 数 / Number of GPUs/NPUs per model instance")
    argparser.add_argument("--npu", action="store_true", default=False, help="使用 NPU 而不是 GPU / Use NPU instead of GPU")
    argparser.add_argument("--thinking", action="store_true", default=False, help="仅 qwen3 支持；开启后使用 Qwen3 think 模式 / Only supported for qwen3; enables Qwen3 think mode")
    argparser.add_argument("--output_dir", type=str, default="res/multi_language", help="基础输出目录；结果会写入时间戳子目录 / Base output directory; results are saved under a timestamped subdirectory")
    argparser.add_argument("--batch_size", type=int, default=0, help="模型批大小 / Batch size for the model")
    argparser.add_argument("--model_name", type=str, default="model", help="采样服务使用的模型名 / Model name used by the sampling service")
    argparser.add_argument("--sample_only", action="store_true", default=False, help="只采样，不做评测 / Only sample without evaluation")
    argparser.add_argument("--eval_only", action="store_true", default=False, help="只评测已有 response / Only evaluate existing responses")
    argparser.add_argument("--use_server", action="store_true", default=False, help="启动本地 vLLM Server 并采样 / Start local vLLM server instances for sampling")
    argparser.add_argument("--use_ray", action="store_true", default=False, help="使用 ray + 本地 vLLM 并行采样 / Use ray with local vLLM for parallel sampling")
    argparser.add_argument("--vllm_server_base_port", type=int, default=8000, help="vLLM Server 起始端口 / Base port for vLLM servers")
    argparser.add_argument("--vllm_server_host", type=str, default="0.0.0.0", help="vLLM Server 监听地址 / Host for vLLM servers")
    argparser.add_argument("--vllm_server_dtype", type=str, default="auto", help="vLLM Server 数据类型 / Data type for vLLM server")
    argparser.add_argument("--vllm_server_wait_timeout", type=int, default=600, help="等待 vLLM Server 就绪的超时时间 / Timeout for waiting vLLM server")
    argparser.add_argument("--mem_fraction", type=float, default=0.9, help="GPU/NPU 显存占用比例 / GPU/NPU memory utilization fraction")
    argparser.add_argument("--resume_sample", action="store_true", default=False, help="从已有采样文件继续补采 / Resume sampling from an existing samples file")
    argparser.add_argument("--resume_sample_path", type=str, default=None, help="用于断点续采的 samples.jsonl 路径 / Path to the samples.jsonl file used for resume sampling")
    argparser.add_argument("--save_full_scalebox_result", action="store_true", default=False, help="在 results.jsonl 中将 scalebox 保存为 ScaleBox 返回体完整内容；默认保存汇总分数 / Save the full ScaleBox response body in results.jsonl instead of the summarized score")
    return argparser



def parse_args():
    """中文：解析命令行参数并做基础校验。
    English: Parse CLI arguments and apply basic validation rules.
    """
    argparser = build_arg_parser()
    args = argparser.parse_args()
    if args.sample_only and args.eval_only:
        argparser.error("--sample_only and --eval_only cannot be used together")
    args._defaults = {action.dest: action.default for action in argparser._actions if action.dest != "help"}
    return args
