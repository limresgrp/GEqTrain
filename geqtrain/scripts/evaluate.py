import os
import sys
import time
import argparse
import logging
from typing import Union
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import ConcatDataset
from geqtrain.data._build import dataset_from_config
from geqtrain.data import AtomicDataDict
from geqtrain.data.dataloader import DataLoader
from geqtrain.scripts.deploy import load_deployed_model, CONFIG_KEY
from geqtrain.train import Trainer
from geqtrain.train.metrics import Metrics
from geqtrain.train.trainer import run_inference, remove_node_centers_for_NaN_targets_and_edges, _init
from geqtrain.train.utils import evaluate_end_chunking_condition
from geqtrain.utils import Config
from geqtrain.utils.auto_init import instantiate
from geqtrain.utils.savenload import load_file


def init_logger(log: bool):
    from geqtrain.utils import Output
    
    if log:
        # Initialize Output with specified settings
        output = Output.get_output(dict(
            root='./logs',
            run_name=time.strftime("%Y%m%d-%H%M%S"),
            append=False,
            screen=False,
            verbose="info",
        ))

        # Open the log files
        logfile = output.open_logfile('log', propagate=True)
        metricsfile = output.open_logfile('metrics.csv', propagate=True)
        csvfile = output.open_logfile('out.csv', propagate=True)
        outfile = output.open_logfile('out.xyz', propagate=True)
        
        # Set up the main logger to log at INFO level
        logger = logging.getLogger(logfile)
        logger.setLevel(logging.INFO)

        # Configure metricslogger to only write to csvfile without stdout
        metricslogger = logging.getLogger('metricslogger')
        metricslogger.setLevel(logging.INFO)
        metrics_handler = logging.FileHandler(metricsfile)  # File handler for csv logger
        metricslogger.addHandler(metrics_handler)           # Add file handler to metricslogger
        metricslogger.propagate = False                 # Prevent propagation to the root logger

        # Configure csvlogger to only write to csvfile without stdout
        csvlogger = logging.getLogger('csvlogger')
        csvlogger.setLevel(logging.INFO)
        csv_handler = logging.FileHandler(csvfile)  # File handler for csv logger
        csvlogger.addHandler(csv_handler)           # Add file handler to csvlogger
        csvlogger.propagate = False                 # Prevent propagation to the root logger

        # Configure xyzlogger to only write to outfile without stdout
        xyzlogger = logging.getLogger('xyzlogger')
        xyzlogger.setLevel(logging.INFO)
        out_handler = logging.FileHandler(outfile)  # File handler for out logger
        xyzlogger.addHandler(out_handler)           # Add file handler to xyzlogger
        xyzlogger.propagate = False                 # Prevent propagation to the root logger

    else:
        logfile = "EvaluateLogger"

        # Set up a dummy logger that does nothing
        dummylogger = logging.getLogger("DummyLogger")
        dummylogger.addHandler(logging.NullHandler())
        dummylogger.setLevel(logging.CRITICAL)  # Set a high level so nothing gets logged
        metricslogger = dummylogger
        csvlogger = dummylogger
        xyzlogger = dummylogger
    
    # Set up the main logger to log at INFO level
    logger = logging.getLogger(logfile)
    logger.setLevel(logging.INFO)

    return logger, metricslogger, csvlogger, xyzlogger

def infer(dataloader, model, device, per_node_outputs_keys=[], chunk_callbacks=[], batch_callbacks=[], **kwargs):
    pbar = tqdm(dataloader)
    for batch_index, data in enumerate(pbar):
        already_computed_nodes = None
        chunk_index = 0
        while True:
            out, ref_data, batch_chunk_center_nodes, num_batch_center_nodes = run_inference(
                model=model,
                data=data,
                device=device,
                cm=torch.no_grad(),
                already_computed_nodes=already_computed_nodes,
                per_node_outputs_keys=per_node_outputs_keys,
                **kwargs,
            )

            for callback in chunk_callbacks:
                callback(batch_index, chunk_index, out, ref_data, pbar, **kwargs)

            chunk_index += 1
            already_computed_nodes = evaluate_end_chunking_condition(already_computed_nodes, batch_chunk_center_nodes, num_batch_center_nodes)
            if already_computed_nodes is None:
                break

        for callback in batch_callbacks:
            callback(batch_index, **kwargs)

def load_model(model: Union[str, Path], device="cpu"):
    if isinstance(model, str):
        model = Path(model)
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

        import tempfile
        tmp = tempfile.NamedTemporaryFile()
        # Open the file for writing.
        with open(tmp.name, 'w') as f:
            f.write(metadata[CONFIG_KEY])
        model_config = Config.from_file(tmp.name)

        model.eval()
        return model, model_config
    except ValueError:  # its not a deployed model
        pass

    # load a training session model
    model, model_config = Trainer.load_model_from_training_session(
        traindir=model.parent, model_name=model.name, device=device
    )
    logger.info("loaded model from training session.")
    model.eval()

    return model, model_config

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
        "-tc",
        "--test-config",
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
    parser.add_argument(
        "--log",
        help="Use this flag to log all inference results. This creates 4 files inside a logs/[DATETIME] folder: "
             "\n-- log -- contains the logs that are printed also on std with info on evaaluation script"
             "\n-- metrics.csv -- contains the metrics evaluation on each chunk of each batch of the test dataset"
             "\n-- out.csv -- contains predicted and target value for each node in test set graphs"
             "\n-- out.xyz -- contains xyz formatted file of molecules in test set, together with inferenced outputs",
        default=False,
        action='store_true',
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

    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit()

    args = parser.parse_args(args=args)

    # Logging
    logger, metricslogger, csvlogger, xyzlogger = init_logger(args.log)

    # Do the defaults:
    dataset_is_from_training: bool = False
    # print_best_model_epoch: bool = False
    if args.train_dir:
        if args.test_config is None:
            args.test_config = args.train_dir / "config.yaml"
            dataset_is_from_training = True
        if args.model is None:
            # print_best_model_epoch = True
            args.model = args.train_dir / "best_model.pth"
        if args.test_indexes is None and dataset_is_from_training:
            # Find the remaining indexes that aren't train or val
            trainer = torch.load(
                str(args.train_dir / "trainer.pth"), map_location="cpu"
            )
            # if print_best_model_epoch:
            #     print(f"Loading model from epoch: {trainer['best_model_saved_at_epoch']}")
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
    if args.test_config is None:
        raise ValueError("--test-config or --train-dir must be provided")

    if args.model is None:
        raise ValueError("--model or --train-dir must be provided")

    # device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

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

    ## --- end of set up of arguments --- ##

    # Load model
    model, config = load_model(args.model, device=args.device)
    logger.info(f"\nUsing Model: {args.model}\n")

    # Load config file
    logger.info(
        f"Loading {'training' if dataset_is_from_training else 'test'} dataset...",
    )

    # Load test config
    evaluate_config = Config.from_file(str(args.test_config), defaults={})
    config.update(evaluate_config)

    # Get dataset
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
        f"Loaded {'test_' if dataset_is_test else 'validation_' if dataset_is_validation else ''}dataset specified in {args.test_config.name}.",
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
            'target_names' : config.get('target_names', list(metrics.keys)),
        }
    except:
        raise Exception("Failed to load Metrics.")

    # --- filter node target to train on based on node type or type name
    keep_type_names = config.get("keep_type_names", None)
    if keep_type_names is not None:
        from geqtrain.train.utils import find_matching_indices
        keep_node_types = torch.tensor(find_matching_indices(config["type_names"], keep_type_names))
    else:
        keep_node_types = None
    
    # --- exclude edges from center node to specified node types
    exclude_type_names_from_edges = config.get("exclude_type_names_from_edges", None)
    if exclude_type_names_from_edges is not None:
        from geqtrain.train.utils import find_matching_indices
        exclude_node_types_from_edges = torch.tensor(find_matching_indices(config["type_names"], exclude_type_names_from_edges))
    else:
        exclude_node_types_from_edges = None

    # dataloader
    per_node_outputs_keys = []
    _indexed_datasets = []
    for _dataset, _test_idcs in zip(dataset.datasets, test_idcs):
        _dataset = _dataset.index_select(_test_idcs)
        _dataset, per_node_outputs_keys = remove_node_centers_for_NaN_targets_and_edges(_dataset, metrics, keep_node_types, exclude_node_types_from_edges)
        if _dataset is not None:
            _indexed_datasets.append(_dataset)
    dataset_test = ConcatDataset(_indexed_datasets)

    dataloader = DataLoader(
        dataset=dataset_test,
        shuffle=False,
        batch_size=args.batch_size,
    )

    for loss_func in metrics.funcs.values():
        _init(loss_func, dataset_test, model)

    # run inference
    logger.info("Starting...")

    def metrics_callback(batch_index, chunk_index, out, ref_data, pbar, **kwargs): # Keep **kwargs or callback fails
        # accumulate metrics
        batch_metrics = metrics(pred=out, ref=ref_data)

        mat_str = f"{batch_index}, {chunk_index}"
        header = "batch,chunk"
        flatten_metrics, skip_keys = metrics.flatten_metrics(
            metrics=batch_metrics,
            metrics_metadata=metrics_metadata,
        )

        for key, value in flatten_metrics.items(): # log metrics
            mat_str += f",{value:16.5g}"
            header += f",{key}"
        
        if pbar.n == 0:
            metricslogger.info(header)
        metricslogger.info(mat_str)

        del out, ref_data
    
    def out_callback(batch_index, chunk_index, out, ref_data, pbar, **kwargs): # Keep **kwargs or callback fails

        def format_csv(data, ref_data, batch_index, chunk_index):
            try:
                # Extract fields from data
                node_type = data[AtomicDataDict.NODE_TYPE_KEY]
                dataset_id = data["dataset_id"].item()  # Scalar
                node_output = data[AtomicDataDict.NODE_OUTPUT_KEY] if AtomicDataDict.NODE_OUTPUT_KEY in data else None
                ref_node_output = ref_data[AtomicDataDict.NODE_OUTPUT_KEY] if AtomicDataDict.NODE_OUTPUT_KEY in ref_data else None

                # Initialize lines list for CSV format
                lines = []
                if pbar.n == 0:
                    lines.append("dataset_id,batch,chunk,node_type,pred,ref")

                if node_output is not None and ref_node_output is not None:
                    for _node_type, _node_output, _ref_node_output in zip(node_type, node_output, ref_node_output):
                        lines.append(f"{dataset_id:6},{batch_index:6},{chunk_index:4},{_node_type.item():6},{_node_output.item():10.4f},{_ref_node_output.item():10.4f}")
            except:
                return ''
            # Join all lines into a single string for XYZ format
            return "\n".join(lines)
        
        def format_xyz(data, ref_data):
            try:
                # Extract fields from data
                pos = data[AtomicDataDict.POSITIONS_KEY]
                node_type = data[AtomicDataDict.NODE_TYPE_KEY]
                dataset_id = data["dataset_id"].item()  # Scalar
                node_output = data[AtomicDataDict.NODE_OUTPUT_KEY] if AtomicDataDict.NODE_OUTPUT_KEY in data else None
                ref_node_output = ref_data[AtomicDataDict.NODE_OUTPUT_KEY] if AtomicDataDict.NODE_OUTPUT_KEY in ref_data else None

                n_atoms = pos.shape[0]

                # Initialize lines list for XYZ format
                lines = []

                # Line 1: Number of atoms
                lines.append(f"{n_atoms}")

                # Line 2: Header line (dataset_id)
                lines.append(f"DatasetID={dataset_id}")

                # Lines 3+: Atom lines with node_type, x, y, z, node_output
                for i in range(n_atoms):
                    atom_name = str(node_type[i].item())  # Convert node_type to string for atom name
                    x, y, z = pos[i].tolist()  # Get coordinates
                    output = node_output[i].item() if node_output is not None else ''  # Extract scalar from tensor
                    ref_output = ref_node_output[i].item() if ref_node_output is not None else ''  # Extract scalar from tensor
                    lines.append(f"{atom_name:6} {x:10.4f} {y:10.4f} {z:10.4f} {output:10.4f} {ref_output:10.4f}")
            except:
                return ''
            # Join all lines into a single string for XYZ format
            return "\n".join(lines)

        csvlogger.info(format_csv(out, ref_data, batch_index, chunk_index))
        xyzlogger.info(format_xyz(out, ref_data))

        del out, ref_data

    infer(dataloader, model, device, per_node_outputs_keys, chunk_callbacks=[metrics_callback, out_callback], **config)

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
    logger.info("\n--- End of evaluation ---")




if __name__ == "__main__":
    main(running_as_script=True)