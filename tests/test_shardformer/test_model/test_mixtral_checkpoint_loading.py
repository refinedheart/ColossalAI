from collections import OrderedDict
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn

from colossalai.checkpoint_io.moe_checkpoint import MoECheckpointIO
from colossalai.shardformer.modeling.mixtral import _mark_fused_expert_tp_shard, _register_fused_expert_checkpoint_hook
from colossalai.tensor.moe_tensor.api import is_moe_tensor, set_moe_tensor_ep_group
from colossalai.testing.utils import spawn


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


def _check_tp_expert_checkpoint_gather(rank: int, world_size: int, port: int):
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        full_gate_up = torch.arange(8 * 8 * 4, dtype=torch.float32).reshape(8, 8, 4)
        full_down = torch.arange(8 * 4 * 6, dtype=torch.float32).reshape(8, 4, 6)
        local_gate_up = torch.cat(
            (full_gate_up[2:4, rank * 2 : (rank + 1) * 2], full_gate_up[2:4, 4 + rank * 2 : 6 + rank * 2]),
            dim=1,
        )
        local_down = full_down[2:4, :, rank * 3 : (rank + 1) * 3]
        experts = _FusedExperts(num_experts=2, gate_up_width=4, down_input_width=3)
        experts.gate_up_proj = nn.Parameter(local_gate_up.clone())
        experts.down_proj = nn.Parameter(local_down.clone())
        _mark_fused_expert_tp_shard(experts.gate_up_proj, "gate_up_proj", dist.group.WORLD, 2, 2, 8)
        _mark_fused_expert_tp_shard(experts.down_proj, "down_proj", dist.group.WORLD, 2, 2, 8)
        set_moe_tensor_ep_group(experts.gate_up_proj, dist.group.WORLD)
        set_moe_tensor_ep_group(experts.down_proj, dist.group.WORLD)
        assert is_moe_tensor(experts.gate_up_proj) and is_moe_tensor(experts.down_proj)

        saved_state = {}
        for shard, _ in MoECheckpointIO._model_sharder(experts):
            saved_state.update(shard)
        assert torch.equal(saved_state["gate_up_proj"], full_gate_up[2:4])
        assert torch.equal(saved_state["down_proj"], full_down[2:4])
        assert torch.equal(experts.gate_up_proj.shard_fn(full_gate_up), local_gate_up)
        assert torch.equal(experts.down_proj.shard_fn(full_down), local_down)

        reloaded = _FusedExperts(num_experts=2, gate_up_width=4, down_input_width=3)
        _register_fused_expert_checkpoint_hook(
            reloaded, expert_start_idx=2, num_experts_per_ep=2, num_experts=8, tp_rank=rank, tp_size=world_size
        )
        reloaded.load_state_dict(saved_state)
        assert torch.equal(reloaded.gate_up_proj, local_gate_up)
        assert torch.equal(reloaded.down_proj, local_down)
    finally:
        dist.destroy_process_group()


def test_gather_and_load_tp_sharded_fused_expert_checkpoint():
    spawn(_check_tp_expert_checkpoint_gather, nprocs=2)


def _check_tp_ep_expert_checkpoint_gather(rank: int, world_size: int, port: int):
    assert world_size == 4
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        # Use a 2x2 EP/TP mesh: ranks in each TP group have different EP groups.
        tp_groups = [dist.new_group(ranks=[0, 1]), dist.new_group(ranks=[2, 3])]
        ep_groups = [dist.new_group(ranks=[0, 2]), dist.new_group(ranks=[1, 3])]
        tp_rank = rank % 2
        ep_rank = rank // 2
        tp_group = tp_groups[ep_rank]
        ep_group = ep_groups[tp_rank]
        expert_start_idx = ep_rank * 4

        full_gate_up = torch.arange(8 * 8 * 4, dtype=torch.float32).reshape(8, 8, 4)
        full_down = torch.arange(8 * 4 * 6, dtype=torch.float32).reshape(8, 4, 6)
        ep_gate_up = full_gate_up[expert_start_idx : expert_start_idx + 4]
        ep_down = full_down[expert_start_idx : expert_start_idx + 4]
        local_gate_up = torch.cat(
            (
                ep_gate_up[:, tp_rank * 2 : (tp_rank + 1) * 2],
                ep_gate_up[:, 4 + tp_rank * 2 : 6 + tp_rank * 2],
            ),
            dim=1,
        )
        local_down = ep_down[:, :, tp_rank * 3 : (tp_rank + 1) * 3]

        experts = _FusedExperts(num_experts=4, gate_up_width=4, down_input_width=3)
        experts.gate_up_proj = nn.Parameter(local_gate_up.clone())
        experts.down_proj = nn.Parameter(local_down.clone())
        _mark_fused_expert_tp_shard(experts.gate_up_proj, "gate_up_proj", tp_group, expert_start_idx, 4, 8)
        _mark_fused_expert_tp_shard(experts.down_proj, "down_proj", tp_group, expert_start_idx, 4, 8)
        set_moe_tensor_ep_group(experts.gate_up_proj, ep_group)
        set_moe_tensor_ep_group(experts.down_proj, ep_group)
        assert is_moe_tensor(experts.gate_up_proj) and is_moe_tensor(experts.down_proj)
        assert dist.get_process_group_ranks(experts.gate_up_proj.ep_group) == dist.get_process_group_ranks(ep_group)

        saved_state = {}
        for shard, _ in MoECheckpointIO._model_sharder(experts):
            saved_state.update(shard)
        assert torch.equal(saved_state["gate_up_proj"], ep_gate_up)
        assert torch.equal(saved_state["down_proj"], ep_down)

        reloaded = _FusedExperts(num_experts=4, gate_up_width=4, down_input_width=3)
        _register_fused_expert_checkpoint_hook(
            reloaded,
            expert_start_idx=expert_start_idx,
            num_experts_per_ep=4,
            num_experts=8,
            tp_rank=tp_rank,
            tp_size=2,
        )
        reloaded.load_state_dict(saved_state)
        assert torch.equal(reloaded.gate_up_proj, local_gate_up)
        assert torch.equal(reloaded.down_proj, local_down)
    finally:
        dist.destroy_process_group()


def test_gather_and_load_tp_ep_sharded_fused_expert_checkpoint():
    spawn(_check_tp_ep_expert_checkpoint_gather, nprocs=4)
