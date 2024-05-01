from typing import List
import logging
from pathlib import Path
from geqtrain.data import AtomicDataDict

import torch

# from geqtrain.data import AtomicData, Collater, dataset_from_config, register_fields
# from geqtrain.scripts._logger import set_up_script_logger
from geqtrain.scripts.deploy import load_deployed_model, R_MAX_KEY
from geqtrain.train import Trainer
from geqtrain.utils._global_options import _set_global_options
from geqtrain.utils import Config

"""
def main(args=None, running_as_script: bool = True):
    # in results dir, do: nequip-deploy build --train-dir . deployed.pth
    parser = argparse.ArgumentParser(
    )
    parser.add_argument(
        "--train-dir",
        help="Path to a working directory from a training session.",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--model",
        help="A deployed or pickled NequIP model to load. If omitted, defaults to `best_model.pth` in `train_dir`.",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--dataset-config",
        help="A YAML config file specifying the dataset to load test data from. If omitted, `config.yaml` in `train_dir` will be used",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--metrics-config",
        help="A YAML config file specifying the metrics to compute. If omitted, `config.yaml` in `train_dir` will be used. If the config does not specify `metrics_components`, the default is to logging.debug MAEs and RMSEs for all fields given in the loss function. If the literal string `None`, no metrics will be computed.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--test-indexes",
        help="Path to a file containing the indexes in the dataset that make up the test set. If omitted, all data frames *not* used as training or validation data in the training session `train_dir` will be used.",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--stride",
        help="If dataset config is provided and test indexes are not provided, take all dataset idcs with this stride",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--batch-size",
        help="Batch size to use. Larger is usually faster on GPU. If you run out of memory, lower this. You can also try to raise this for faster evaluation. Default: 50.",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--repeat",
        help=(
            "Number of times to repeat evaluating the test dataset. "
            "This can help compensate for CUDA nondeterminism, or can be used to evaluate error on models whose inference passes are intentionally nondeterministic. "
            "Note that `--repeat`ed passes over the dataset will also be `--output`ed if an `--output` is specified."
        ),
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
        "--device",
        help="Device to run the model on. If not provided, defaults to CUDA if available and CPU otherwise.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output",
        help="ExtXYZ (.xyz) file to write out the test set and model predictions to.",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-fields",
        help="Extra fields (names[:field] comma separated with no spaces) to write to the `--output`.\n"
             "Field options are: [node, edge, graph, long].\n"
             "If [:field] is omitted, the field with that name is assumed to be already registered by default.",
        type=str,
        default="",
    )
    parser.add_argument(
        "--log",
        help="log file to store all the metrics and screen logging.debug",
        type=Path,
        default=None,
    )
    # Something has to be provided
    # See https://stackoverflow.com/questions/22368458/how-to-make-argparse-logging.debug-usage-when-no-option-is-given-to-the-code
    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit()
    # Parse the args
    args = parser.parse_args(args=args)

    # Do the defaults:
    dataset_is_from_training: bool = False
    if args.train_dir:
        if args.dataset_config is None:
            args.dataset_config = args.train_dir / "config.yaml"
            dataset_is_from_training = True
        if args.metrics_config is None:
            args.metrics_config = args.train_dir / "config.yaml"
        if args.model is None:
            args.model = args.train_dir / "best_model.pth"
        if args.test_indexes is None:
            # Find the remaining indexes that arent train or val
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
    # update
    if args.metrics_config == "None":
        args.metrics_config = None
    elif args.metrics_config is not None:
        args.metrics_config = Path(args.metrics_config)
    do_metrics = args.metrics_config is not None
    # validate
    if args.dataset_config is None:
        raise ValueError("--dataset-config or --train-dir must be provided")
    if args.metrics_config is None and args.output is None:
        raise ValueError(
            "Nothing to do! Must provide at least one of --metrics-config, --train-dir (to use training config for metrics), or --output"
        )
    if args.model is None:
        raise ValueError("--model or --train-dir must be provided")
    output_type: Optional[str] = None
    if args.output is not None:
        if args.output.suffix != ".xyz":
            raise ValueError("Only .xyz format for `--output` is supported.")
        args.output_fields = [register_field(e) for e in args.output_fields.split(",") if e != ""] + [
            ORIGINAL_DATASET_INDEX_KEY
        ]
        output_type = "xyz"
    else:
        assert args.output_fields == ""
        args.output_fields = []

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if running_as_script:
        set_up_script_logger(args.log)
    logger = logging.getLogger("nequip-evaluate")
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

    # Load model:
    loaded_deployed_model, model_r_max, model, model_config, model_root = load_model(args, device, logger)

    # Load a config file
    logger.info(
        f"Loading {'original ' if dataset_is_from_training else ''}dataset...",
    )
    defaults = {"r_max": model_r_max}
    if not loaded_deployed_model:
        defaults.update({
            "root": model_root
        })
    dataset_config = Config.from_file(
        str(args.dataset_config), defaults=defaults
    )
    if dataset_config["r_max"] != model_r_max:
        raise RuntimeError(
            f"Dataset config has r_max={dataset_config['r_max']}, but model has r_max={model_r_max}!"
        )

    dataset_is_test: bool = False
    dataset_is_validation: bool = False
    try:
        # Try to get test dataset
        dataset, _ = dataset_from_config(dataset_config, prefix="test_dataset")
        dataset_is_test = True
    except KeyError:
        pass
    if not dataset_is_test:
        try:
            # Try to get validation dataset
            dataset, _ = dataset_from_config(dataset_config, prefix="validation_dataset")
            dataset_is_validation = True
        except KeyError:
            pass
    
    if not (dataset_is_test or dataset_is_validation):
        # Get shared train + validation dataset
        # prefix `dataset`
        dataset, _ = dataset_from_config(dataset_config)
    logger.info(
        f"Loaded {'test_' if dataset_is_test else 'validation_' if dataset_is_validation else ''}dataset specified in {args.dataset_config.name}.",
    )

    c = Collater.for_dataset(dataset, exclude_keys=[])

    # Determine the test set
    # this makes no sense if a dataset is given seperately
    if (
        args.test_indexes is None
        and dataset_is_from_training
        and train_idcs is not None
    ):
        if dataset_is_test:
            test_idcs = torch.arange(len(dataset))
            logger.info(
                f"Using all frames from original test dataset, yielding a test set size of {len(test_idcs)} frames.",
            )
        else:
            # we know the train and val, get the rest
            all_idcs = set(range(len(dataset)))
            # set operations
            if dataset_is_validation:
                test_idcs = list(all_idcs - val_idcs)
                logger.info(
                    f"Using origial validation dataset ({len(dataset)} frames) minus validation set frames ({len(val_idcs)} frames), yielding a test set size of {len(test_idcs)} frames.",
                )
            else:
                test_idcs = list(all_idcs - train_idcs - val_idcs)
                assert set(test_idcs).isdisjoint(train_idcs)
                logger.info(
                    f"Using origial training dataset ({len(dataset)} frames) minus training ({len(train_idcs)} frames) and validation frames ({len(val_idcs)} frames), yielding a test set size of {len(test_idcs)} frames.",
                )
            # No matter what it should be disjoint from validation:
            assert set(test_idcs).isdisjoint(val_idcs)
            if not do_metrics:
                logger.info(
                    "WARNING: using the automatic test set ^^^ but not computing metrics, is this really what you wanted to do?",
                )
    elif args.test_indexes is None:
        # Default to all frames
        test_idcs = torch.arange(len(dataset))
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
    test_idcs = torch.as_tensor(test_idcs, dtype=torch.long)[::args.stride]
    test_idcs = test_idcs.tile((args.repeat,))

    # Figure out what metrics we're actually computing
    if do_metrics:
        metrics_config = Config.from_file(str(args.metrics_config))
        metrics_components = metrics_config.get("metrics_components", None)
        # See trainer.py: init() and init_metrics()
        # Default to loss functions if no metrics specified:
        if metrics_components is None:
            loss, _ = instantiate(
                builder=Loss,
                prefix="loss",
                positional_args=dict(coeffs=metrics_config.loss_coeffs),
                all_args=metrics_config,
            )
            metrics_components = []
            for key, func in loss.funcs.items():
                params = {
                    "PerSpecies": type(func).__name__.startswith("PerSpecies"),
                }
                metrics_components.append((key, "mae", params))
                metrics_components.append((key, "rmse", params))

        metrics, _ = instantiate(
            builder=Metrics,
            prefix="metrics",
            positional_args=dict(components=metrics_components),
            all_args=metrics_config,
        )
        metrics.to(device=device)

    batch_i: int = 0
    batch_size: int = args.batch_size
    stop = False
    already_computed_nodes = None

    logger.info("Starting...")
    context_stack = contextlib.ExitStack()
    with contextlib.ExitStack() as context_stack:
        # "None" checks if in a TTY and disables if not
        prog = context_stack.enter_context(tqdm(total=len(test_idcs), disable=None))
        if do_metrics:
            display_bar = context_stack.enter_context(
                tqdm(
                    bar_format=""
                    if prog.disable  # prog.ncols doesn't exist if disabled
                    else ("{desc:." + str(prog.ncols) + "}"),
                    disable=None,
                )
            )

        if output_type is not None:
            output = []
            output_target = []
            dataset_idx_to_idx = dict()
            for idx, ds in enumerate(dataset.datasets):
                ds_filename = ".".join(os.path.split(ds.file_name)[-1].split(".")[:-1])
                path, out_filename = os.path.split(args.output)
                out_filename_split = out_filename.split(".")
                dataset_idx_to_idx[ds.dataset_idx] = idx
                output_filename = ".".join([f"ds_{ds.dataset_idx}__{ds_filename}__" + ".".join(out_filename_split[:-1])] + out_filename_split[-1:])
                output.append(context_stack.enter_context(open(os.path.join(path, output_filename), "w")))
                output_target_filename = ".".join([f"ds_{ds.dataset_idx}__{ds_filename}__" + ".".join(out_filename_split[:-1]) + "_target"] + out_filename_split[-1:])
                output_target.append(context_stack.enter_context(open(os.path.join(path, output_target_filename), "w")))
        else:
            output = None
            output_target = None

        while True:
            complete_out = None
            torch.cuda.empty_cache()
            while True:
                this_batch_test_indexes = test_idcs[
                    batch_i * batch_size : (batch_i + 1) * batch_size
                ]
                try:
                    datas = [dataset[int(idex)] for idex in this_batch_test_indexes]
                except ValueError: # Most probably an atom in pdb that is missing in model
                    batch_i += 1
                    prog.update(len(this_batch_test_indexes))
                    continue
                if len(datas) == 0:
                    stop = True
                    break

                out, batch, already_computed_nodes = evaluate(c, datas, device, model, already_computed_nodes, model_config)
                if complete_out is None:
                    complete_out = copy.deepcopy(batch)
                    if AtomicDataDict.PER_ATOM_ENERGY_KEY in out:
                        complete_out[AtomicDataDict.PER_ATOM_ENERGY_KEY] = torch.zeros(
                            (len(batch[AtomicDataDict.POSITIONS_KEY]), *out[AtomicDataDict.PER_ATOM_ENERGY_KEY].shape[1:]),
                            dtype=torch.get_default_dtype(),
                            device=out[AtomicDataDict.PER_ATOM_ENERGY_KEY].device
                        )
                    if AtomicDataDict.TOTAL_ENERGY_KEY in out:
                        complete_out[AtomicDataDict.TOTAL_ENERGY_KEY] = torch.zeros_like(
                            out[AtomicDataDict.TOTAL_ENERGY_KEY]
                        )
                if AtomicDataDict.PER_ATOM_ENERGY_KEY in complete_out:
                    original_nodes = out[AtomicDataDict.ORIG_EDGE_INDEX_KEY][0].unique()
                    nodes = out[AtomicDataDict.EDGE_INDEX_KEY][0].unique()
                    complete_out[AtomicDataDict.PER_ATOM_ENERGY_KEY][original_nodes] = out[AtomicDataDict.PER_ATOM_ENERGY_KEY][nodes].detach()
                
                if AtomicDataDict.FORCE_KEY in complete_out:
                    complete_out[AtomicDataDict.FORCE_KEY][original_nodes] = out[AtomicDataDict.FORCE_KEY][nodes].detach()

                if AtomicDataDict.TOTAL_ENERGY_KEY in complete_out:
                    complete_out[AtomicDataDict.TOTAL_ENERGY_KEY] += out[AtomicDataDict.TOTAL_ENERGY_KEY].detach()
                del out

                if already_computed_nodes is None:
                    break
            if stop:
                break

            with torch.no_grad():
                # Write output
                if output_type == "xyz":
                    # add test frame to the output:
                    complete_out[ORIGINAL_DATASET_INDEX_KEY] = torch.LongTensor(
                        this_batch_test_indexes
                    )
                    batch[ORIGINAL_DATASET_INDEX_KEY] = torch.LongTensor(
                        this_batch_test_indexes
                    )
                    # append to the file
                    for dataset_idx in torch.unique(complete_out['dataset_idx']).to('cpu').tolist():
                        idx = dataset_idx_to_idx[dataset_idx]
                        ase.io.write(
                            output[idx],
                            AtomicData.from_AtomicDataDict(complete_out)
                            .to(device="cpu")
                            .to_ase(
                                type_mapper=dataset.datasets[idx].type_mapper,
                                extra_fields=args.output_fields,
                                filter_idcs=(complete_out['dataset_idx'] == dataset_idx).to('cpu'),
                            ),
                            format="extxyz",
                            append=True,
                        )
                        ase.io.write(
                            output_target[idx],
                            AtomicData.from_AtomicDataDict(batch)
                            .to(device="cpu")
                            .to_ase(
                                type_mapper=dataset.datasets[idx].type_mapper,
                                filter_idcs=(complete_out['dataset_idx'] == dataset_idx).to('cpu'),
                            ),
                            format="extxyz",
                            append=True,
                        )

            # Accumulate metrics
            if do_metrics:
                try:
                    metrics(complete_out, batch)
                    display_bar.set_description_str(
                        " | ".join(
                            f"{k} = {v:4.4f}"
                            for k, v in metrics.flatten_metrics(
                                metrics.current_result(),
                                type_names=dataset.datasets[0].type_mapper.type_names,
                            )[0].items()
                        )
                    )
                except:
                    display_bar.set_description_str(
                        "No metrics available for this dataset. Ground truth may be missing."
                    )

            batch_i += 1
            prog.update(len(batch['ptr'] - 1))

        prog.close()
        if do_metrics:
            display_bar.close()

    if do_metrics:
        logger.info("\n--- Final result: ---")
        logger.critical(
            "\n".join(
                f"{k:>20s} = {v:< 20f}"
                for k, v in metrics.flatten_metrics(
                    metrics.current_result(),
                    type_names=dataset.datasets[0].type_mapper.type_names,
                )[0].items()
            )
        )
"""     

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
    
    global_config = model.parent / "config.yaml"
    global_config = Config.from_file(str(global_config))
    _set_global_options(global_config)
    del global_config

    # load a training session model
    model, model_config = Trainer.load_model_from_training_session(
        traindir=model.parent, model_name=model.name, device=device
    )
    logger.info("loaded model from training session.")
    model.eval()

    return model, model_config

def evaluate(model, batch, node_out_keys: List[str] = [], extra_out_keys: List[str] = []):
    device = next(model.parameters()).device
    batch_index = batch[AtomicDataDict.EDGE_INDEX_KEY]
    num_batch_center_nodes = len(batch_index[0].unique())

    results = {}
    already_computed_nodes = None
    while True:

        input_data, _, batch_chunk_center_nodes = Trainer.prepare_chunked_input_data(
            already_computed_nodes=already_computed_nodes,
            batch=batch,
            data=batch,
            device=device,
        )

        with torch.no_grad():
            out = model(input_data)
        del input_data

        for node_out_key in node_out_keys:
            chunk_node_out = out[node_out_key]
            if node_out_key not in results:
                results[node_out_key] = torch.zeros(
                    (num_batch_center_nodes, chunk_node_out.shape[-1]),
                    dtype=torch.get_default_dtype(),
                    device=chunk_node_out.device
                )
            results[node_out_key][batch_chunk_center_nodes] = chunk_node_out[out[AtomicDataDict.EDGE_INDEX_KEY][0].unique()]
        
        for extra_out_key in extra_out_keys:
            chunk_extra_out = out[extra_out_key]
            if extra_out_key not in results:
                results[extra_out_key] = chunk_extra_out
            else:
                fltr = torch.argwhere(~torch.isnan(chunk_extra_out[:, 0])).flatten()
                results[extra_out_key][fltr] = chunk_extra_out[fltr]
        
        # from e3nn import o3
        # R = o3.Irreps('1o').D_from_angles(*[torch.tensor(x) for x in [0, 90, 0]]).to(batch_['pos'].device)
        # batch_['pos'] = torch.einsum("ij,zj->zi", R, batch_['pos'])
        # out2 = model(batch_)
        # R2 = o3.Irreps('1o').D_from_angles(*[torch.tensor(x) for x in [0, -90, 0]]).to(batch_['pos'].device)
        # o1 = out['forces']
        # o2 = torch.einsum("ij,zj->zi", R2, out2['forces'])

        if already_computed_nodes is None:
            if len(batch_chunk_center_nodes) < num_batch_center_nodes:
                already_computed_nodes = batch_chunk_center_nodes
            else:
                assert len(batch_chunk_center_nodes) == num_batch_center_nodes
                return results
        elif len(already_computed_nodes) + len(batch_chunk_center_nodes) == num_batch_center_nodes:
            return results
        else:
            assert len(already_computed_nodes) + len(batch_chunk_center_nodes) < num_batch_center_nodes
            already_computed_nodes = torch.cat([already_computed_nodes, batch_chunk_center_nodes], dim=0)


if __name__ == "__main__":
    pass
    # main(running_as_script=True)