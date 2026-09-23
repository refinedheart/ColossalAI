from collections import OrderedDict

import torch
from torch import nn

from colossalai.shardformer.modeling.mixtral import _register_fused_expert_checkpoint_hook


class _FusedExperts(nn.Module):
    def __init__(self, num_experts: int, gate_up_width: int = 6, down_input_width: int = 3):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, gate_up_width, 4))
        self.down_proj = nn.Parameter(torch.empty(num_experts, 4, down_input_width))


def test_load_full_fused_expert_checkpoint_for_local_ep_shard():
    full_gate_up = torch.arange(8 * 6 * 4, dtype=torch.float32).reshape(8, 6, 4)
    full_down = torch.arange(8 * 4 * 3, dtype=torch.float32).reshape(8, 4, 3)
    experts = _FusedExperts(num_experts=2)
    _register_fused_expert_checkpoint_hook(experts, expert_start_idx=2, num_experts_per_ep=2, num_experts=8)

    state_dict = OrderedDict(gate_up_proj=full_gate_up, down_proj=full_down)
    experts.load_state_dict(state_dict)

    assert torch.equal(experts.gate_up_proj, full_gate_up[2:4])
    assert torch.equal(experts.down_proj, full_down[2:4])


def test_load_local_fused_expert_checkpoint_without_reslicing():
    local_gate_up = torch.randn(2, 6, 4)
    local_down = torch.randn(2, 4, 3)
    experts = _FusedExperts(num_experts=2)
    _register_fused_expert_checkpoint_hook(experts, expert_start_idx=2, num_experts_per_ep=2, num_experts=8)

    experts.load_state_dict(OrderedDict(gate_up_proj=local_gate_up, down_proj=local_down))

    assert torch.equal(experts.gate_up_proj, local_gate_up)
    assert torch.equal(experts.down_proj, local_down)


def test_load_full_fused_expert_checkpoint_with_tp_shards():
    full_gate_up = torch.arange(8 * 8 * 4, dtype=torch.float32).reshape(8, 8, 4)
    full_down = torch.arange(8 * 4 * 6, dtype=torch.float32).reshape(8, 4, 6)
    experts = _FusedExperts(num_experts=2, gate_up_width=4, down_input_width=3)
    _register_fused_expert_checkpoint_hook(
        experts, expert_start_idx=2, num_experts_per_ep=2, num_experts=8, tp_rank=1, tp_size=2
    )

    experts.load_state_dict(OrderedDict(gate_up_proj=full_gate_up, down_proj=full_down))

    expected_gate_up = torch.cat((full_gate_up[2:4, 2:4], full_gate_up[2:4, 6:8]), dim=1)
    assert torch.equal(experts.gate_up_proj, expected_gate_up)
    assert torch.equal(experts.down_proj, full_down[2:4, :, 3:6])
