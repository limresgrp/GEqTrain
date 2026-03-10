import logging
from typing import Optional, List

import torch
from e3nn import o3

from geqtrain.data import AtomicDataDict, _NODE_FIELDS, _EDGE_FIELDS
from geqtrain.nn import GraphModuleMixin, SequentialGraphNetwork
from geqtrain.nn._embedding_time import PositionalEmbedding
from geqtrain.utils import Config

from geqtrain.data.AtomicData import register_fields


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
        recycled_fields: List[str],
        time_encoding_d_model: int,
        irreps_in: dict = None,
    ):
        super().__init__()
        self.recycling_steps = recycling_steps
        self.time_encoding = None

        # Check if recycle_step is a conditioning field
        if AtomicDataDict.RECYCLE_STEP_KEY in recycled_fields:
            self.time_encoding = PositionalEmbedding(
                d_model=time_encoding_d_model, max_len=recycling_steps
            )
            # Remove it from the main list to avoid processing it as a standard field
            self.recycled_fields = [f for f in recycled_fields if f != AtomicDataDict.RECYCLE_STEP_KEY]
        else:
            self.recycled_fields = recycled_fields

        # Define irreps for outputs
        new_irreps_out = {}
        if self.time_encoding is not None:
            new_irreps_out[AtomicDataDict.RECYCLE_STEP_KEY] = f"{time_encoding_d_model}x0e"

        for field in self.recycled_fields:
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
        - Updates recycled fields based on the previous step's output.
        """
        recycle_step = data[AtomicDataDict.RECYCLE_STEP_KEY]

        # 1. Add time embedding if it's configured
        if self.time_encoding is not None: # Assuming time embedding is node-level
            data[AtomicDataDict.RECYCLE_STEP_KEY] = self.time_encoding(recycle_step, num_nodes=data[AtomicDataDict.POSITIONS_KEY].shape[0]) 

        # 2. Update recycled fields
        for field in self.recycled_fields:
            # On the first cycle (k=0), initialize recycled features to zero.
            # For subsequent cycles (k>0), the RecycleModelWrapper is responsible
            # for setting these fields based on the previous prediction_state.
            if recycle_step[0] == 0:
                field_dim = self.irreps_in[field].dim
                
                # Determine batch size based on field level (node or edge)
                if field in _NODE_FIELDS:
                    field_batch_size = data[AtomicDataDict.POSITIONS_KEY].shape[0]
                elif field in _EDGE_FIELDS:
                    field_batch_size = data[AtomicDataDict.EDGE_INDEX_KEY].shape[1]
                else:
                    raise ValueError(f"Recycled field '{field}' is not a recognized node or edge field. Cannot infer zero-initialization shape.")

                data[field] = torch.zeros(
                    (field_batch_size, field_dim),
                    device=data[AtomicDataDict.POSITIONS_KEY].device,
                    dtype=data[AtomicDataDict.POSITIONS_KEY].dtype,
                )

        return data

class RecycleModelWrapper(GraphModuleMixin, torch.nn.Module):
    """
    A wrapper model that implements recycling logic.
    It iterates over a wrapped model, conditioning each step on the previous one.
    """
    _is_wrapper: bool = True  # Prevent flattening by SequentialGraphNetwork

    def __init__(self, config: Config, model: SequentialGraphNetwork):
        super().__init__()
        self.recycling_steps = int(config.get("recycling_steps", 1))
        initial_irreps_in = config.get("irreps_in", {})
        self.output_field = config.get("output_field")
        if self.output_field is None:
            raise ValueError("RecycleModel requires 'output_field' to be specified in the config.")

        recycled_fields = config.get("recycled_fields", [])
        # Single alpha parameter for the additive output update, initialized to 0.0 so sigmoid(0.0) = 0.5
        self.alpha_param = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        register_fields(node_fields=[AtomicDataDict.RECYCLE_STEP_KEY])
        self.recycle_block = RecycleBlock(
            recycling_steps=self.recycling_steps,
            time_encoding_d_model=config.get("time_encoding_d_model", 64),
            recycled_fields=recycled_fields,
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
        
        # Determine the shape of the output field for initialization
        output_irreps = self.wrapped_model.irreps_out[self.output_field]
        output_dim = output_irreps.dim
        
        # Infer if it's node-level or edge-level to get the batch dimension
        if self.output_field in _NODE_FIELDS:
            batch_size = data[AtomicDataDict.POSITIONS_KEY].shape[0]
        elif self.output_field in _EDGE_FIELDS:
            batch_size = data[AtomicDataDict.EDGE_INDEX_KEY].shape[1]
        else:
            raise ValueError(f"Output field '{self.output_field}' is not a recognized node or edge field. Cannot infer initial prediction_state shape.")

        device = data[AtomicDataDict.POSITIONS_KEY].device
        dtype = data[AtomicDataDict.POSITIONS_KEY].dtype
        
        # Initialize prediction_state to zeros. This is the initial guess.
        prediction_state = torch.zeros(batch_size, output_dim, device=device, dtype=dtype)

        # Apply sigmoid to the single alpha parameter to get a step size between 0 and 1
        alpha_sigmoid = torch.sigmoid(self.alpha_param)

        for k in range(self.recycling_steps):
            data[AtomicDataDict.RECYCLE_STEP_KEY] = torch.tensor([k], dtype=torch.long, device=device)

            # For k > 0, the recycled field (which is `self.output_field`)
            # is set to the `prediction_state` from the previous iteration.
            # For k = 0, `RecycleBlock` will initialize `data[self.output_field]` to zeros.
            if k > 0:
                if self.output_field not in self.recycle_block.recycled_fields:
                    raise ValueError(f"Output field '{self.output_field}' must be in 'recycled_fields' for iterative refinement.")
                data[self.output_field] = prediction_state.detach() # Detach to prevent gradients flowing through previous steps

            # 1. Prepare inputs for this cycle (adds time embedding, initializes recycled fields to zero for k=0)
            data = self.recycle_block(data)

            # 2. Run the core model to predict a correction/delta
            current_data = self.wrapped_model(data)
            delta_prediction = current_data[self.output_field] # The model now always predicts a delta

            # 3. Update the prediction state (ResNet-style: current_prediction = previous_prediction + alpha * delta)
            prediction_state = (1 - alpha_sigmoid) * prediction_state + alpha_sigmoid * delta_prediction

            # 4. Store intermediate prediction for deep supervision
            deep_supervision_predictions.append(prediction_state.clone())

        # Set the final outputs in the data dictionary
        data[self.output_field] = prediction_state
        data[self.output_field + AtomicDataDict.DEEP_SUPERVISION_SUFFIX] = torch.stack(deep_supervision_predictions, dim=-1)
        # Update data with key-values in current_data which data is missing
        for key, value in current_data.items():
            if key not in data:
                data[key] = value
        
        return data
