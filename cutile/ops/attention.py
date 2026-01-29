import torch
import math
from typing import Optional
from cutile.ops.attention_prefill import fmha_prefill
from cutile.ops.attention_decode import fmha_decode

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
old = ALL_ATTENTION_FUNCTIONS["sdpa"]

def fmha(
    module: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    has_backward: Optional[bool] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Core FMHA implementation with minimal required parameters.
    """
    if scaling is None:
        scaling = 1.0 / math.sqrt(q.size(-1))

    if q.size(-2) == 1:
        return old(module, q, k, v, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal, has_backward=has_backward, **kwargs)
        # return fmha_decode(q, k, v, sm_scale=scaling), None

    # Set default values
    is_causal = True if is_causal is None else is_causal
    has_backward = False if has_backward is None else has_backward
    # Call fmha_interface with the given arguments
    o = fmha_prefill(
        q,
        k,
        v,
        is_causal=is_causal,
        scaling=scaling,
    )
    # return o.transpose(1, 2).contiguous(), None
    return o, None