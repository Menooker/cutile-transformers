from optimum_benchmark import Benchmark, BenchmarkConfig, TorchrunConfig, InferenceConfig, PyTorchConfig
from optimum_benchmark.logging_utils import setup_logging
import torch
import transformers.models.qwen2.modeling_qwen2 as qwen2_mod
from cutile.modules.Qwen2MLP import MyQwen2MLP


setup_logging(level="INFO")
cutile_enabled = True

if cutile_enabled:
    qwen2_mod.Qwen2MLP = MyQwen2MLP

model_name = "/home/menooker/models/Qwen2.5-1.5B"

if __name__ == "__main__":
    launcher_config = TorchrunConfig(nproc_per_node=1)
    scenario_config = InferenceConfig(latency=True)
    scenario_config.input_shapes['sequence_length']=128
    backend_config = PyTorchConfig(model=model_name, device="cuda", device_ids="0",  task='text-generation', torch_dtype="float16")
    name_postfix = "_mlp_optimized" if cutile_enabled else ""
    benchmark_config = BenchmarkConfig(
        name=f"qwen2.5-1.5B_inference{name_postfix}",
        scenario=scenario_config,
        launcher=launcher_config,
        backend=backend_config,
    )
    benchmark_report = Benchmark.launch(benchmark_config)
    # push artifacts to the hub
    benchmark_config.save_json(f"benchreports/qwen2.5-1.5B_inference{name_postfix}_benchmark_config.json")
    benchmark_report.save_json(f"benchreports/qwen2.5-1.5B_inference{name_postfix}_benchmark_report.json")