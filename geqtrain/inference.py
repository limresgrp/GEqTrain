import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch

from geqtrain.train.components.checkpointing import CheckpointHandler
from geqtrain.train.components.inference import run_inference
from geqtrain.utils.config import Config
from geqtrain.utils.inference_metadata import (
    INFERENCE_METADATA_KEY,
    build_inference_metadata_bundle,
    load_inference_metadata_bundle,
)


def _as_config_dict(config: Any) -> Dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "as_dict") and callable(config.as_dict):
        return dict(config.as_dict())
    return {}


def _resolve_inference_metadata(
    model_config: Mapping[str, Any],
    raw_metadata: Optional[Mapping[str, str]],
) -> Dict[str, Any]:
    if isinstance(raw_metadata, Mapping):
        payload = raw_metadata.get(INFERENCE_METADATA_KEY, "")
        bundle = load_inference_metadata_bundle(payload)
        if len(bundle) > 0:
            return bundle
    return build_inference_metadata_bundle(model_config, normalization_stats_by_ensemble=None)


@dataclass
class InferenceSession:
    model: torch.nn.Module
    config: Config
    device: torch.device
    raw_metadata: Dict[str, str]
    inference_metadata: Dict[str, Any]

    @classmethod
    def from_model_path(
        cls,
        model_path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
    ) -> "InferenceSession":
        model, config, metadata = CheckpointHandler.load_model(str(model_path), device=device)
        config_obj = Config.from_dict(_as_config_dict(config))
        model_device = torch.device(device)
        model.to(model_device)
        model.eval()
        inference_metadata = _resolve_inference_metadata(config_obj.as_dict(), metadata)
        return cls(
            model=model,
            config=config_obj,
            device=model_device,
            raw_metadata=dict(metadata),
            inference_metadata=inference_metadata,
        )

    def predict(
        self,
        batch,
        *,
        denormalize_outputs: Optional[bool] = None,
        loss_fn=None,
        is_train: bool = False,
        current_epoch: int = 0,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Any, int]:
        run_cfg = copy.deepcopy(self.config.as_dict())
        if denormalize_outputs is not None:
            run_cfg["denormalize_inference_outputs"] = bool(denormalize_outputs)
        return run_inference(
            model=self.model,
            data=batch,
            device=self.device,
            config=run_cfg,
            inference_metadata=self.inference_metadata,
            loss_fn=loss_fn,
            already_computed_nodes=None,
            is_train=is_train,
            current_epoch=current_epoch,
        )


def load_model_for_inference(
    model_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> InferenceSession:
    return InferenceSession.from_model_path(model_path=model_path, device=device)
