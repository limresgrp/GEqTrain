import logging
from typing import Optional, List

import torch
from e3nn import o3

from geqtrain.data import AtomicDataDict
from geqtrain.nn import GraphModuleMixin, SequentialGraphNetwork
from geqtrain.nn._embedding_time import PositionalEmbedding
from geqtrain.utils import Config


def RecycleModel(
    config: Config, model: Optional[SequentialGraphNetwork]
) -> SequentialGraphNetwork:
    """
    Builder for a model that performs recycling.
    """
    logging.info("--- Building Recycle Model Wrapper ---")
    if model is None:
        raise ValueError("RecycleModel requires a non-None model to wrap.")

    return RecycleModelWrapper(config=config, model=model)


class RecycleBlock(GraphModuleMixin, torch.nn.Module):
    """
    A block that adds recycle step embeddings to the data dictionary.
    """

    def __init__(
        self,
        recycling_steps: int,
        conditioning_fields: List[str],
        time_encoding_d_model: int,
        irreps_in: dict = None,
    ):
        super().__init__()
        self.recycling_steps = recycling_steps
        self.time_encoding = None
        
        # Check if recycle_step is a conditioning field
        if AtomicDataDict.RECYCLE_STEP_KEY in conditioning_fields:
            self.time_encoding = PositionalEmbedding(
                d_model=time_encoding_d_model, max_len=recycling_steps
            )
            # Remove it from the main list to avoid processing it as a standard field
            self.conditioning_fields = [f for f in conditioning_fields if f != AtomicDataDict.RECYCLE_STEP_KEY]
        else:
            self.conditioning_fields = conditioning_fields

        # Define irreps for outputs
        new_irreps_out = {}
        if self.time_encoding is not None:
            new_irreps_out[AtomicDataDict.RECYCLE_STEP_KEY] = f"{time_encoding_d_model}x0e"

        for field in self.conditioning_fields:
            if field not in irreps_in:
                raise ValueError(f"Conditioning field '{field}' not found in irreps_in: {list(irreps_in.keys())}")
            new_irreps_out[field] = irreps_in[field]

        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out=new_irreps_out,
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        """
        Prepares the data for the current recycle step.
        - Adds the time step embedding.
        - Updates conditioning fields based on the previous step's output.
        """
        recycle_step = data[AtomicDataDict.RECYCLE_STEP_KEY]

        # 1. Add time embedding if it's configured
        if self.time_encoding is not None:
            data[AtomicDataDict.RECYCLE_STEP_KEY] = self.time_encoding(recycle_step, num_nodes=data[AtomicDataDict.POSITIONS_KEY].shape[0])

        # 2. Update conditioning fields
        for field in self.conditioning_fields:
            # On the first cycle (k=0), initialize conditioning features to zero.
            if recycle_step[0] == 0:
                # Use a field that is guaranteed to exist to get shape and device
                ref_tensor = data[AtomicDataDict.POSITIONS_KEY]
                data[field] = torch.zeros(
                    (ref_tensor.shape[0], self.irreps_in[field].dim),
                    device=ref_tensor.device,
                    dtype=ref_tensor.dtype,
                )
            else:
                # On subsequent cycles, perform a ResNet-style update.
                # The previous output is expected to be in `data[f"{field}_prev_out"]`
                prev_out_key = f"{field}_prev_out"
                if prev_out_key not in data:
                    raise KeyError(f"Expected previous output '{prev_out_key}' in data dictionary for recycle step > 0.")

                # Use the output of the previous step as the conditioning for the current step.
                data[field] = data[prev_out_key]
                
                # Clean up the previous output key
                del data[prev_out_key]

        return data

class RecycleModelWrapper(GraphModuleMixin, torch.nn.Module):
    """
    A wrapper model that implements recycling logic.
    It iterates over a wrapped model, conditioning each step on the previous one.
    """
    _is_wrapper: bool = True  # Prevent flattening by SequentialGraphNetwork

    def __init__(self, config: Config, model: SequentialGraphNetwork):
        super().__init__()
        self.recycling_steps = config.get("recycling_steps", 1)
        initial_irreps_in = config.get("irreps_in", {})
        self.output_field = config.get("output_field")
        if self.output_field is None:
            raise ValueError("RecycleModel requires 'output_field' to be specified in the config.")

        conditioning_fields = config.get("conditioning_fields", [])
        alpha_init = config.get("alpha_init", 1.0)

        # This alpha is for the additive output update
        self.alphas = torch.nn.ParameterList([
            torch.nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32)) for _ in range(self.recycling_steps)])

        self.recycle_block = RecycleBlock(
            recycling_steps=self.recycling_steps,
            time_encoding_d_model=config.get("time_encoding_d_model", 64),
            conditioning_fields=conditioning_fields,
            irreps_in=model.irreps_out,
        )
        self.wrapped_model = model

        # Final irreps of the wrapper
        final_irreps_out = self.wrapped_model.irreps_out.copy()

        # The main output is the accumulated sum, which has the same irreps
        final_irreps_out.update(self.recycle_block.irreps_out)
        # The deep supervision output will be stacked
        deep_supervision_key = self.output_field + AtomicDataDict.DEEP_SUPERVISION_SUFFIX
        final_irreps_out[deep_supervision_key] = o3.Irreps(self.wrapped_model.irreps_out[self.output_field])

        self._init_irreps(
            irreps_in=initial_irreps_in,
            irreps_out=final_irreps_out,
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        deep_supervision_predictions = []
        final_prediction = None
        pos = data[AtomicDataDict.POSITIONS_KEY]
        num_atoms = pos.shape[0]
        device = pos.device

        for k in range(self.recycling_steps):
            data[AtomicDataDict.RECYCLE_STEP_KEY] = torch.tensor([k], device=device)

            # 1. Prepare inputs for this cycle
            # This updates conditioning fields based on the previous step's output
            data = self.recycle_block(data)

            # 2. Run the core model
            current_data = self.wrapped_model(data)
            current_prediction = current_data[self.output_field]

            # 3. Update the final output prediction with a weighted additive correction
            weighted_correction = self.alphas[k] * current_prediction

            if k == 0:
                # Initialize the prediction. This is not in-place and starts the graph.
                final_prediction = weighted_correction
            else:
                # Use out-of-place addition for subsequent updates.
                final_prediction = final_prediction + weighted_correction

            # 4. Store outputs for deep supervision and for the next cycle's conditioning
            deep_supervision_predictions.append(current_prediction.clone())
            for field in self.recycle_block.conditioning_fields:
                # Pass the output of the conditioning field to the next step, detached
                data[f"{field}_prev_out"] = current_data[field].detach()

        # Set the final outputs in the data dictionary
        data[self.output_field] = final_prediction
        data[self.output_field + AtomicDataDict.DEEP_SUPERVISION_SUFFIX] = torch.stack(deep_supervision_predictions, dim=-1)
        # Update data with key-values in current_data which data is missing
        for key, value in current_data.items():
            if key not in data:
                data[key] = value
        
        return data
