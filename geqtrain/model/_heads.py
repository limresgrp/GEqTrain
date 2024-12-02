import logging
from typing import List, Optional

from geqtrain.data import AtomicDataDict
from geqtrain.nn import (
    SequentialGraphNetwork,
    ReadoutModule,
)

from geqtrain.data import AtomicDataDict
from torch.utils.data import ConcatDataset


def Heads(model, config, initialize: bool, dataset: Optional[ConcatDataset] = None) -> SequentialGraphNetwork:

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

        layers.update({
            f"head_{out_field}": (
                ReadoutModule,
                dict(
                    field=field,
                    out_field=out_field,
                    out_irreps=out_irreps,
                    strict_irreps=False,
                ),
            ),
        })

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )