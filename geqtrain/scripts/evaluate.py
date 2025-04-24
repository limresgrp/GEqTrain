import sys
import time
import argparse
import logging
from typing import Union
from pathlib import Path
from tqdm import tqdm

import torch
from geqtrain.data._build import dataset_from_config
from geqtrain.data import AtomicDataDict
from geqtrain.data.dataloader import DataLoader
from geqtrain.scripts.deploy import load_deployed_model, CONFIG_KEY
from geqtrain.train import Trainer
from geqtrain.train.metrics import Metrics
from geqtrain.train.trainer import get_output_keys, run_inference, _init
from geqtrain.train.utils import evaluate_end_chunking_condition
from geqtrain.utils import Config, INVERSE_ATOMIC_NUMBER_MAP
from geqtrain.utils.auto_init import instantiate
from geqtrain.train.sampler import EnsembleSampler

def init_logger(log: str = None):
    from geqtrain.utils import Output

    if log is not None:
        # Initialize Output with specified settings
        output = Output.get_output(dict(
            root=log,
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

def infer(dataloader, model, device, output_keys=[], per_node_outputs_keys=[], chunk_callbacks=[], batch_callbacks=[], **kwargs):
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
                output_keys=output_keys,
                per_node_outputs_keys=per_node_outputs_keys,
                **kwargs,
            )

            for callback in chunk_callbacks:
                callback(batch_index, chunk_index, out, ref_data, data, pbar, **kwargs)

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
    model, model_config = Trainer.load_model_from_training_session(traindir=model.parent, model_name=model.name, device=device, for_inference=True)
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
        default=2,
    )
    parser.add_argument(
        "-d",
        "--device",
        help="Device to run the model on. If not provided, defaults to CUDA if available and CPU otherwise.",
        type=str,
        default='cpu',
    )
    parser.add_argument(
        "-s",
        "--stride",
        help="If dataset config is provided and test indexes are not provided, take all dataset idcs with this stride",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-w",
        "--workers",
        help="Number of workers to process dataset. Default: 1 (single process)",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-l",
        "--log",
        nargs='?',
        const='./logs',
        default=None,
        help="Use this flag to log all inference results. This creates 4 files: "
             "\n-- log -- contains the logs that are printed also on std with info on evaaluation script"
             "\n-- metrics.csv -- contains the metrics evaluation on each chunk of each batch of the test dataset"
             "\n-- out.csv -- contains predicted and target value for each node in test set graphs"
             "\n-- out.xyz -- contains xyz formatted file of molecules in test set, together with inferenced outputs"
             "\n\nIf this argument is not specified, no logging is performed. If it is specified without a directory name, logs to the ./logs directory.",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit()

    args = parser.parse_args(args=args)

    # Logging
    if args.log is not None and args.batch_size > 1:
        logger.warning("Logging with batch_size > 1 does not support storing 'out.csv' and 'out.xyz' logs."
                       "\nIf you want to log that information, use the default batch_size of 1.")
    logger, metricslogger, csvlogger, xyzlogger = init_logger(args.log)

    # Do the defaults:
    trainer = None
    if args.train_dir:
        if args.test_config is None:
            args.test_config = args.train_dir / "config.yaml"
        if args.model is None:
            args.model = args.train_dir / "best_model.pth"
            trainer = torch.load(str(args.train_dir / "trainer.pth"), map_location="cpu")


    # Validate
    if args.test_config is None:
        raise ValueError("--test-config or --train-dir must be provided")

    if args.model is None:
        raise ValueError("--model or --train-dir must be provided")

    # Device
    device = torch.device(args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.warning("Please note that models running on CUDA are usually nondeterministc and that this manifests in the final test errors; for a _more_ deterministic result, please use `--device cpu`",)
    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading model...")
    model, config = load_model(args.model, device=args.device)
    if trainer:
        logger.info(f"Model loaded:\n\t{args.model}\n\tSaved at epoch {trainer['progress']['best_epoch']}")

    # Check model convergence with WeightWatcher
    # TODO: make this somehow conditional from cmd line; care: it must be wrt a key
    ww = False #True
    if ww:
        import weightwatcher as ww
        watcher = ww.WeightWatcher(model = model)
        details = watcher.analyze(plot=True)
        print(details)

        # Write dataframe into a file in args.train_dir
        if args.train_dir:
            details_file = args.train_dir / "weightwatcher_details.csv"
            details.to_csv(details_file, index=False)
            logger.info(f"WeightWatcher details saved to {details_file}")

    # Load config file
    logger.info(f"Loading config file...")
    evaluate_config = Config.from_file(str(args.test_config), defaults={})
    config.update(evaluate_config)
    config["dataset_num_workers"] = args.workers
    logger.info(f"Config file loaded!")

    # Load metrics (if specified)
    metrics = None
    try:
        logger.info(f"Metrics loading ... ")
        metrics_components = config.get("metrics_components", None)
        if metrics_components is not None:
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
            logger.info(f"Metrics loaded")
    except:
        raise Exception("Failed to load Metrics.")

    # Load dataset
    logger.info(f"Loading dataset...")
    try:
        dataset = dataset_from_config(config, prefix="test_dataset")
    except KeyError:
        try:
            dataset = dataset_from_config(config, prefix="validation_dataset")
        except KeyError:
            dataset = dataset_from_config(config)

    logger.info(f"Dataset specified in {args.test_config.name} loaded!")

    if metrics is not None:
        for loss_func in metrics.funcs.values():
            _init(loss_func, dataset, model)

    # set up dataloader
    dl_kwargs = {'dataset':dataset,'shuffle':False}
    # evaluate wheter to use EnsembleSampler or not
    dataset_mode = config.get("dataset_mode", "single")
    assert dataset_mode in ["single", "ensemble"], f"Expected 'single' or 'ensemble', got {dataset_mode}"
    use_ensemble = dataset_mode == 'ensemble'

    # use_ensemble not yet working
    use_ensemble = False
    if use_ensemble:
        sampler = EnsembleSampler(dataset, args.batch_size)
        dl_kwargs.update(dict(sampler=sampler))
    else:
        dl_kwargs.update(dict(batch_size=args.batch_size))

    dataloader = DataLoader(**dl_kwargs)

    # run inference
    logger.info("Starting...")

    def metrics_callback(batch_index, chunk_index, out, ref_data, data, pbar, **kwargs): # Keep **kwargs or callback fails
        # accumulate metrics
        batch_metrics = metrics(pred=out, ref=ref_data)

        mat_str = f"{batch_index}, {chunk_index}"
        header = "batch,chunk"
        flatten_metrics = metrics.flatten_metrics(
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

    def out_callback(batch_index, chunk_index, out, ref_data, data, pbar, **kwargs): # Keep **kwargs or callback fails

        def format_csv(data, ref_data, batch_index, chunk_index, dataset_raw_file_name):
            try:
                # Extract fields from data
                node_output = data[AtomicDataDict.NODE_OUTPUT_KEY] if AtomicDataDict.NODE_OUTPUT_KEY in data else None
                if node_output is None:
                    return ''
                if not isinstance(node_output, torch.Tensor): node_output = node_output.mean[0]
                node_type = data[AtomicDataDict.NODE_TYPE_KEY]
                atom_number = data.get(AtomicDataDict.ATOM_NUMBER_KEY, node_type)
                ref_node_output = ref_data[AtomicDataDict.NODE_OUTPUT_KEY] if AtomicDataDict.NODE_OUTPUT_KEY in ref_data else None
                node_centers = data[AtomicDataDict.EDGE_INDEX_KEY][0].unique()
                num_node_centers = len(node_centers)
                if len(atom_number) > num_node_centers: atom_number = atom_number[node_centers]
                if len(node_output) > num_node_centers: node_output = node_output[node_centers]
                if ref_node_output is not None and len(ref_node_output) > num_node_centers: ref_node_output = ref_node_output[node_centers]

                # Initialize lines list for CSV format
                lines = []
                if pbar.n == 0:
                    lines.append("batch,chunk,atom_number,node_type,pred,ref,dataset_raw_file_name")

                for idx, (_atom_number, _node_type, _node_output) in enumerate(zip(atom_number, node_type, node_output)):
                    _ref_node_output = ref_node_output[idx].item() if ref_node_output is not None else 0
                    lines.append(f"{batch_index:6},{chunk_index:4},{_atom_number.item():6},{_node_type.item():6},{_node_output.item():10.4f},{_ref_node_output:10.4f},{dataset_raw_file_name}")
            except:
                return ''
            # Join all lines into a single string for XYZ format
            return "\n".join(lines)

        def format_xyz(data, ref_data):
            return ''
            try:
                # Extract fields from data
                pos = data[AtomicDataDict.POSITIONS_KEY]
                node_type = data[AtomicDataDict.NODE_TYPE_KEY]
                atom_number = data.get(AtomicDataDict.ATOM_NUMBER_KEY, node_type)
                dataset_raw_file_name = data[AtomicDataDict.DATASET_RAW_FILE_NAME]
                node_output = data[AtomicDataDict.NODE_OUTPUT_KEY] if AtomicDataDict.NODE_OUTPUT_KEY in data else None
                ref_node_output = ref_data[AtomicDataDict.NODE_OUTPUT_KEY] if AtomicDataDict.NODE_OUTPUT_KEY in ref_data else None
                node_centers = data[AtomicDataDict.EDGE_INDEX_KEY][0].unique()
                if ref_node_output is not None:
                    ref_node_output = ref_node_output[node_centers]

                n_atoms = pos.shape[0]
                if not isinstance(node_output, torch.Tensor): node_output = node_output.mean[0]

                # Initialize lines list for XYZ format
                lines = []

                # Line 1: Number of atoms
                lines.append(f"{n_atoms}")

                # Line 2: Header line (dataset_id)
                lines.append(f"DatasetID={dataset_id}")

                # Lines 3+: Atom lines with node_type, x, y, z, node_output
                for i in range(n_atoms):
                    atom_name = INVERSE_ATOMIC_NUMBER_MAP.get(atom_number[i].item(), 'X')
                    atom_type = str(node_type[i].item())
                    x, y, z = pos[i].tolist()  # Get coordinates
                    output = node_output[i].item() if node_output is not None else ''  # Extract scalar from tensor
                    ref_output = ref_node_output[i].item() if ref_node_output is not None else 0  # Extract scalar from tensor
                    lines.append(f"{atom_name:2} {x:10.4f} {y:10.4f} {z:10.4f} {output:10.4f} {ref_output:10.4f} {atom_type:6}")
            except:
                return ''
            # Join all lines into a single string for XYZ format
            return "\n".join(lines)

        csvlogger.info(format_csv(out, ref_data, batch_index, chunk_index, data[AtomicDataDict.DATASET_RAW_FILE_NAME][0]))
        xyzlogger.info(format_xyz(out, ref_data))

        del out, ref_data

    chunk_callbacks = [out_callback]
    if metrics is not None:
        chunk_callbacks.append(metrics_callback)

    output_keys, per_node_outputs_keys = get_output_keys(metrics)

    # TODO: make this somehow conditional from cmd line; care: it must be wrt a key
    use_accuracy = True
    if use_accuracy:
        from geqtrain.utils.evaluate_utils import AccuracyMetric
        for k in set(output_keys):
            if k in ['bace_head']:
                chunk_callbacks += [AccuracyMetric(k)]

    config.pop("device")
    infer(dataloader, model, device, output_keys, per_node_outputs_keys, chunk_callbacks=chunk_callbacks, **config)

    if use_accuracy:
        for cb in chunk_callbacks:
            if isinstance(cb, AccuracyMetric):
                cb.print_current_result()

    if metrics is not None:
        logger.info("\n--- Final result: ---")
        logger.info(
            "\n".join(
                f"{k:>20s} = {v:< 20f}"
                for k, v in metrics.flatten_metrics(
                    metrics.current_result(),
                    metrics_metadata=metrics_metadata,
                ).items()
            )
        )
    logger.info("\n--- End of evaluation ---")


if __name__ == "__main__":
    main(running_as_script=True)