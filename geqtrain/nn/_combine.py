from typing import List, Tuple
import torch
from e3nn import o3

from e3nn.util.jit import compile_mode

from geqtrain.data import AtomicDataDict
from geqtrain.nn import GraphModuleMixin


@compile_mode("script")
class CombineModule(GraphModuleMixin, torch.nn.Module):
    """
    Combine multiple fields by additive accumulation with irrep-aware channel matching.

    The first field defines the output irreps/layout. Each subsequent field can either:
    - have exactly the same irreps, or
    - be an irrep subset of the first field (e.g., `1x0e` added into `1x0e+1x2e`).
    """

    def __init__(
        self,
        fields: List[str],
        out_field: str,
        # Other:
        irreps_in = None,
    ):
        super().__init__()
        assert len(fields) > 1

        self.fields = fields
        self.out_field = out_field

        base_field = fields[0]
        base_irreps = irreps_in[base_field]

        # check and init irreps
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=fields,
            irreps_out={self.out_field: base_irreps},
        )

        self.add_fields: List[str] = list(fields[1:])
        self.src_indices: List[torch.Tensor] = []
        self.dst_indices: List[torch.Tensor] = []

        for field in self.add_fields:
            field_irreps = irreps_in[field]
            src_idx, dst_idx = self._build_index_map(base_irreps, field_irreps)
            self.src_indices.append(src_idx)
            self.dst_indices.append(dst_idx)

    @staticmethod
    def _build_index_map(
        base_irreps: o3.Irreps,
        add_irreps: o3.Irreps,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if base_irreps == add_irreps:
            dim = base_irreps.dim
            idx = torch.arange(dim, dtype=torch.long)
            return idx, idx

        src_indices: List[int] = []
        dst_indices: List[int] = []

        # Build destination slots with per-irrep multiplicity bookkeeping.
        base_slots = []
        base_offset = 0
        for mul, ir in base_irreps:
            base_slots.append(
                {
                    "ir": ir,
                    "mul": int(mul),
                    "used": 0,
                    "offset": int(base_offset),
                    "ir_dim": int(ir.dim),
                }
            )
            base_offset += mul * ir.dim

        src_offset = 0
        for add_mul, add_ir in add_irreps:
            remaining = int(add_mul)
            consumed = 0
            add_ir_dim = int(add_ir.dim)

            for slot in base_slots:
                if slot["ir"] != add_ir:
                    continue
                available = slot["mul"] - slot["used"]
                if available <= 0:
                    continue

                take = min(remaining, available)
                if take <= 0:
                    continue

                src_start = src_offset + consumed * add_ir_dim
                dst_start = slot["offset"] + slot["used"] * add_ir_dim

                for j in range(take * add_ir_dim):
                    src_indices.append(src_start + j)
                    dst_indices.append(dst_start + j)

                slot["used"] += take
                consumed += take
                remaining -= take
                if remaining == 0:
                    break

            if remaining > 0:
                raise ValueError(
                    f"Cannot add irreps {add_irreps} into {base_irreps}: "
                    f"missing capacity for component {add_mul}x{add_ir}."
                )

            src_offset += add_mul * add_ir_dim

        if len(src_indices) != add_irreps.dim:
            raise ValueError(
                f"Internal index mapping error while combining {add_irreps} into {base_irreps}."
            )

        return (
            torch.tensor(src_indices, dtype=torch.long),
            torch.tensor(dst_indices, dtype=torch.long),
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        combined = data[self.fields[0]].clone()
        for idx, field in enumerate(self.add_fields):
            source = data[field]
            src_idx = self.src_indices[idx].to(device=source.device)
            dst_idx = self.dst_indices[idx].to(device=combined.device)

            if source.shape[:-1] != combined.shape[:-1]:
                raise ValueError(
                    f"Cannot combine field '{field}' with shape {source.shape} into "
                    f"'{self.fields[0]}' with shape {combined.shape}: leading dimensions differ."
                )

            update = source.index_select(dim=-1, index=src_idx)
            combined = combined.index_add(dim=-1, index=dst_idx, source=update)

        data[self.out_field] = combined
        return data
