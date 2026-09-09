"""One real-data optimization step per tutorial configuration, without caching."""

import argparse
from pathlib import Path

import numpy as np
import torch

from geqtrain.data import AtomicData
from geqtrain.model import model_from_config
from geqtrain.train.components.inference import run_inference
from geqtrain.train.components.setup import setup_loss, setup_metrics, setup_optimizer
from geqtrain.utils import load_hydra_config
from geqtrain.utils._global_options import apply_global_config
from geqtrain.utils.torch_geometric import Batch


def check_config(path, device):
    config = load_hydra_config(str(path))
    apply_global_config(config)
    torch.manual_seed(config["seed"])
    model, _ = model_from_config(config, initialize=False)
    model.to(device)
    source = config["train_dataset_list"][0]["dataset_input"]
    graphs = []
    with np.load(source, allow_pickle=False) as dataset:
        for frame in range(min(2, len(dataset["coords"]))):
            graphs.append(AtomicData.from_points(
                pos=torch.as_tensor(dataset["coords"][frame], dtype=torch.get_default_dtype()),
                r_max=config["r_max"],
                node_types=torch.as_tensor(dataset["atom_types"], dtype=torch.long).reshape(-1, 1),
                energy=torch.as_tensor(dataset["energy"][frame], dtype=torch.get_default_dtype()).reshape(1),
                forces=torch.as_tensor(dataset["forces"][frame], dtype=torch.get_default_dtype()),
            ))
    batch = Batch.from_data_list(graphs)
    loss = setup_loss(config)
    metrics = setup_metrics(config, target_irreps=model.irreps_out)
    optimizer = setup_optimizer(model, config)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    out, ref, _, _ = run_inference(model, batch, device, config.as_dict(), loss_fn=loss, is_train=True)
    value, contributions = loss(out, ref)
    assert torch.isfinite(value), "Nonfinite training loss"
    value.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients), "Invalid gradients"
    optimizer.step()
    model.eval()
    out, ref, _, _ = run_inference(model, batch, device, config.as_dict(), loss_fn=loss)
    assert out["forces"].shape == batch.forces.shape
    assert out["energy"].shape == (len(graphs), 1)
    assert torch.isfinite(loss(out, ref)[0])
    for key in ("radial_reconstruction_mask", "angular_reconstruction_mask"):
        if key in out:
            assert not out[key].any(), "Validation must not corrupt geometry"
    metrics(out, ref)
    print(f"{path.stem}: PASS; parameters={sum(p.numel() for p in model.parameters())}; "
          f"training losses={ {k: round(v.item(), 5) for k, v in contributions.items()} }; "
          f"clean-eval metrics={ {k: round(v.item(), 5) for k, v in metrics.current_result().items()} }", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config", type=Path, help="Check one experiment instead of all four.")
    args = parser.parse_args()
    # Keep tiny CPU checks fast and avoid oversubscribing shared workstations.
    torch.set_num_threads(1)
    paths = [args.config] if args.config else sorted((Path(__file__).parent / "config/experiment").glob("*.yaml"))
    for path in paths:
        print(f"Checking {path} ...", flush=True)
        check_config(path, torch.device(args.device))


if __name__ == "__main__":
    main()
