import contextlib
import copy
import os
import sys
import argparse
from typing import List, Optional
import logging
from pathlib import Path

from tqdm import tqdm

from geqtrain.data import AtomicDataDict

import torch
from torch.utils.data import ConcatDataset
from geqtrain.data._build import dataset_from_config
from geqtrain.data.dataloader import Collater, DataLoader
from geqtrain.scripts.deploy import load_deployed_model, R_MAX_KEY
from geqtrain.train import Trainer
from geqtrain.train.loss import Loss
from geqtrain.train.metrics import Metrics
from geqtrain.train.trainer import run_inference
from geqtrain.utils._global_options import _set_global_options
from geqtrain.utils import Config
from geqtrain.utils.auto_init import instantiate
from geqtrain.utils.savenload import load_file


def main(args=None, running_as_script: bool = True):
    # in results dir, do: geqtrain-deploy build --train-dir . deployed.pth
    parser = argparse.ArgumentParser(
    )
    parser.add_argument(
        "-td",
        "--train-dir",
        help="Path to a working directory from a training session.",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "-m",
        "--model",
        help="A deployed or pickled GEqTrain model to load. If omitted, defaults to `best_model.pth` in `train_dir`.",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "-c",
        "--config",
        help="A YAML config file specifying the dataset to load test data from. If omitted, `config.yaml` in `train_dir` will be used",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        help="Batch size to use. Larger is usually faster on GPU. If you run out of memory, lower this. You can also try to raise this for faster evaluation. Default: 16.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-d",
        "--device",
        help="Device to run the model on. If not provided, defaults to CUDA if available and CPU otherwise.",
        type=str,
        default='cpu',
    )
    # parser.add_argument(
    #     "--repeat",
    #     help=(
    #         "Number of times to repeat evaluating the test dataset. "
    #         "This can help compensate for CUDA nondeterminism, or can be used to evaluate error on models whose inference passes are intentionally nondeterministic. "
    #         "Note that `--repeat`ed passes over the dataset will also be `--output`ed if an `--output` is specified."
    #     ),
    #     type=int,
    #     default=1,
    # )
    parser.add_argument(
        "--test-indexes",
        help="Path to a file containing the indexes in the dataset that make up the test set. "
             "If omitted, all data frames *not* used as training or validation data in the training session `train_dir` will be used.",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "-s",
        "--stride",
        help="If dataset config is provided and test indexes are not provided, take all dataset idcs with this stride",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--use-deterministic-algorithms",
        help="Try to have PyTorch use deterministic algorithms. Will probably fail on GPU/CUDA.",
        type=bool,
        default=False,
    )
    # parser.add_argument(
    #     "--output",
    #     help="ExtXYZ (.xyz) file to write out the test set and model predictions to.",
    #     type=Path,
    #     default=None,
    # )
    # parser.add_argument(
    #     "--output-fields",
    #     help="Extra fields (names[:field] comma separated with no spaces) to write to the `--output`.\n"
    #          "Field options are: [node, edge, graph, long].\n"
    #          "If [:field] is omitted, the field with that name is assumed to be already registered by default.",
    #     type=str,
    #     default="",
    # )
    parser.add_argument(
        "--log",
        help="log file to store all the metrics and screen logging.debug",
        type=Path,
        default=None,
    )

    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit()
    # Parse args
    args = parser.parse_args(args=args)

    # Do the defaults:
    dataset_is_from_training: bool = False
    if args.train_dir:
        if args.config is None:
            args.config = args.train_dir / "config.yaml"
            dataset_is_from_training = True
        if args.model is None:
            args.model = args.train_dir / "best_model.pth"
        if args.test_indexes is None and dataset_is_from_training:
            # Find the remaining indexes that aren't train or val
            trainer = torch.load(
                str(args.train_dir / "trainer.pth"), map_location="cpu"
            )
            train_idcs = []
            dataset_offset = 0
            for tr_idcs in trainer["train_idcs"]:
                train_idcs.extend([tr_idx + dataset_offset for tr_idx in tr_idcs.tolist()])
                dataset_offset += len(tr_idcs)
            train_idcs = set(train_idcs)
            val_idcs = []
            dataset_offset = 0
            for v_idcs in trainer["val_idcs"]:
                val_idcs.extend([v_idx + dataset_offset for v_idx in v_idcs.tolist()])
                dataset_offset += len(v_idcs)
            val_idcs = set(val_idcs)
        else:
            train_idcs = val_idcs = None

    # validate
    if args.config is None:
        raise ValueError("--config or --train-dir must be provided")

    if args.model is None:
        raise ValueError("--model or --train-dir must be provided")

    # device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # logger
    logger = logging.getLogger("geqtrain-evaluate")
    logger.setLevel(logging.INFO)

    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(
            "WARNING: please note that models running on CUDA are usually nondeterministc and that this manifests in the final test errors; for a _more_ deterministic result, please use `--device cpu`",
        )

    if args.use_deterministic_algorithms:
        logger.info(
            "Telling PyTorch to try to use deterministic algorithms... please note that this will likely error on CUDA/GPU"
        )
        torch.use_deterministic_algorithms(True)

    # Load model
    model, config = load_model(args.model, device=args.device)

    # Load config file
    logger.info(
        f"Loading {'training' if dataset_is_from_training else 'test'} dataset...",
    )
    
    evaluate_config = Config.from_file(str(args.config), defaults={})
    config.update(evaluate_config)
    
    dataset_is_test: bool = False
    dataset_is_validation: bool = False
    try:
        # Try to get test dataset
        dataset = dataset_from_config(config, prefix="test_dataset")
        dataset_is_test = True
    except KeyError:
        pass
    if not dataset_is_test:
        try:
            # Try to get validation dataset
            dataset = dataset_from_config(config, prefix="validation_dataset")
            dataset_is_validation = True
        except KeyError:
            pass

    if not (dataset_is_test or dataset_is_validation):
        raise Exception("Either test or validation dataset must be provided.")
    logger.info(
        f"Loaded {'test_' if dataset_is_test else 'validation_' if dataset_is_validation else ''}dataset specified in {args.config.name}.",
    )

    if args.test_indexes is None:
        # Default to all frames
        test_idcs = [torch.arange(len(ds)) for ds in dataset.datasets]
        logger.info(
            f"Using all frames from the specified test dataset with stride {args.stride}, yielding a test set size of {len(test_idcs)} frames.",
        )
    else:
        # load from file
        test_idcs = load_file(
            supported_formats=dict(
                torch=["pt", "pth"], yaml=["yaml", "yml"], json=["json"]
            ),
            filename=str(args.test_indexes),
        )
        logger.info(
            f"Using provided test set indexes, yielding a test set size of {len(test_idcs)} frames.",
        )

    test_idcs = [torch.as_tensor(idcs, dtype=torch.long)[::args.stride] for idcs in test_idcs]
    # test_idcs = test_idcs.tile((args.repeat,))

    # Figure out what metrics we're actually computing
    try:
        metrics_components = config.get("metrics_components", None)
        metrics, _ = instantiate(
            builder=Metrics,
            prefix="metrics",
            positional_args=dict(components=metrics_components),
            all_args=config,
        )
        metrics.to(device=device)
        metrics_metadata = {
            'type_names'   : config["type_names"],
            'target_names' : config['target_names'],
        }
    except:
        raise Exception("Failed to load Metrics.")

    # dataloader
    _indexed_datasets = []
    for _dataset, _test_idcs in zip(dataset.datasets, test_idcs):
        _indexed_datasets.append(_dataset.index_select(_test_idcs))
    dataset_test = ConcatDataset(_indexed_datasets)

    dataloader = DataLoader(
        dataset=dataset_test,
        shuffle=False,
        batch_size=args.batch_size,
    )

    # run inference
    logger.info("Starting...")

    pbar = tqdm(dataloader)
    for data in pbar:

        already_computed_nodes = None
        while True:
            
            out, ref_data, batch_chunk_center_nodes, num_batch_center_nodes = run_inference(
                model=model,
                data=data,
                device=device,
                cm=torch.no_grad(),
                already_computed_nodes=already_computed_nodes,
                skip_chunking=True,
            )

            # accumulate metrics
            batch_metrics = metrics(pred=out, ref=ref_data)

            desc = '\t'.join(
                f'{k:>20s} = {v:< 20f}'
                for k, v in metrics.flatten_metrics(
                    batch_metrics,
                    metrics_metadata=metrics_metadata,
                )[0].items()
            )
            pbar.set_description(f"Metrics: {desc}")
            del out, ref_data

            # evaluate ending condition
            if already_computed_nodes is None: # already_computed_nodes is the stopping criteria to finish batch step
                if len(batch_chunk_center_nodes) < num_batch_center_nodes:
                    already_computed_nodes = batch_chunk_center_nodes
            elif len(already_computed_nodes) + len(batch_chunk_center_nodes) == num_batch_center_nodes:
                already_computed_nodes = None
            else:
                assert len(already_computed_nodes) + len(batch_chunk_center_nodes) < num_batch_center_nodes
                already_computed_nodes = torch.cat([already_computed_nodes, batch_chunk_center_nodes], dim=0)

            if already_computed_nodes is None:
                break

    logger.info("\n--- Final result: ---")
    
    logger.info(
        "\n".join(
            f"{k:>20s} = {v:< 20f}"
            for k, v in metrics.flatten_metrics(
                metrics.current_result(),
                metrics_metadata=metrics_metadata,
            )[0].items()
        )
    )


def load_model(model: Path, device="cpu"):
    logger = logging.getLogger("geqtrain-evaluate")
    logger.setLevel(logging.INFO)

    logger.info("Loading model... ")

    try:
        model, metadata = load_deployed_model(
            model,
            device=device,
            set_global_options=True,  # don't warn that setting
        )
        logger.info("loaded deployed model.")
        model_config = {
            "r_max": float(metadata[R_MAX_KEY]),
        }
        model.eval()
        return model, model_config
    except ValueError:  # its not a deployed model
        pass

    # global_config = model.parent / "config.yaml"
    # global_config = Config.from_file(str(global_config))
    # _set_global_options(global_config)
    # del global_config

    # load a training session model
    model, model_config = Trainer.load_model_from_training_session(
        traindir=model.parent, model_name=model.name, device=device
    )
    logger.info("loaded model from training session.")
    model.eval()

    return model, model_config

if __name__ == "__main__":
    main(running_as_script=True)