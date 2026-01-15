
import torch.nn as nn
import torch.nn.functional as F
import torch
from cutile.ops.matmul_silu_mul import launch_matmul_silu_mul, launch_gemv_silu_mul
from cutile.ops.matmul import launch_matmul, launch_gemv


class MyQwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, dtype=torch.float16)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, dtype=torch.float16)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False, dtype=torch.float16)
        assert config.hidden_act in ["silu"], "Unsupported activation function"
        # self.act_fn = ACT2FN[config.hidden_act]

    def init_weights(self):
        self.gate_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.up_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.down_proj.weight.data.normal_(mean=0.0, std=0.02)


    def forward(self, x):
        batch_size = x.size(0)
        xv = x.view(-1, x.size(-1))
        M = xv.size(0)
        v0 = torch.empty((M, self.intermediate_size), device=x.device, dtype=torch.float16)
        # if M == 1:
        #     launch_gemv_silu_mul(xv, self.gate_proj.weight, self.up_proj.weight, v0, approx=False)
        # else:
        launch_matmul_silu_mul(xv, self.gate_proj.weight, self.up_proj.weight, v0, approx=False)
        finalout = torch.empty((M, self.hidden_size), device=x.device, dtype=torch.float16)
        # if M == 1:
        #     launch_gemv(v0, self.down_proj.weight, finalout)
        # else:
        launch_matmul(v0, self.down_proj.weight, finalout, transb=True, act=0)
        return finalout.view(batch_size, -1, self.hidden_size)


if __name__ == "__main__":
    import transformers.models.qwen2.modeling_qwen2 as qwen2_mod
    # Simple test
    class Config:
        hidden_size = 128
        intermediate_size = 64
        hidden_act = "silu"

    config = Config()
    mlp = MyQwen2MLP(config).cuda()
    x = torch.rand((8, 16, 128), device='cuda', dtype=torch.float16)
    output = mlp(x)
    mlp2 = qwen2_mod.Qwen2MLP(config).cuda()
    mlp2.gate_proj = mlp.gate_proj
    mlp2.up_proj = mlp.up_proj
    mlp2.down_proj = mlp.down_proj
    output2 = mlp2(x)
    torch.testing.assert_close(output, output2, rtol=1e-5, atol=1e-4)
    print("✓ MyQwen2MLP test passed!")