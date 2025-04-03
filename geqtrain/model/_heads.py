import logging
from typing import List, Optional

from geqtrain.data import AtomicDataDict
from geqtrain.nn import (
    SequentialGraphNetwork,
    ReadoutModule,
    # GVPGeqTrain,
    WeightedTP,
    TransformerBlock,
)

from geqtrain.utils import Config
from geqtrain.data import AtomicDataDict
from torch.utils.data import ConcatDataset





def Heads(model, config: Config, initialize: bool, dataset: Optional[ConcatDataset] = None) -> SequentialGraphNetwork:
    '''
    instanciates a layer with multiple ReadoutModules
    '''

    logging.info("--- Building Heads Module ---")

    layers = {
        "wrapped_model": model,
    }

    for head_tuple in config.get("heads", []):
        assert isinstance(head_tuple, List) or isinstance(head_tuple, tuple), f"Elements of 'heads' must be tuples ([field], out_field, out_irreps). Found type {type(head_tuple)}"

        if len(head_tuple) == 3:
            field, out_field, out_irreps = head_tuple
        elif len(head_tuple) == 2:
            field = AtomicDataDict.NODE_FEATURES_KEY
            out_field, out_irreps = head_tuple
        else:
            raise Exception(f"Elements of 'heads' must be tuples of the following type ([field], out_field, out_irreps).")

        # # #! transformer head
        # layers.update({
        #     f"head_{out_field}": (
        #         TransformerBlock,
        #         dict(
        #             field=field,
        #             out_field=out_field,
        #         ),
        #     ),
        # })

        # # #! WTP heads
        # layers.update({
        #     f"head_{out_field}": (
        #         WeightedTP,
        #         dict(
        #             field=field,
        #             out_field=out_field,
        #         ),
        #     ),
        # })

        # #! GVP heads
        # layers.update({
        #     f"head_{out_field}": (
        #         GVPGeqTrain,
        #         dict(
        #             field=field,
        #             out_field=out_field,
        #         ),
        #     ),
        # })

        # ! ReadoutModule heads
        layers.update({
            f"head_{out_field}": (
                ReadoutModule,
                dict(
                    field=field,
                    out_field=out_field,
                    out_irreps=out_irreps,
                    strict_irreps=False,
                    ignore_amp=True,
                    # ensemble_attention=True,
                ),
            ),
        })

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )