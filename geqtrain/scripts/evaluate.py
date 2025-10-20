# geqtrain/scripts/evaluate.py
import logging
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import numpy as np

from geqtrain.data.dataloader import DataLoader
from geqtrain.utils import Config
from geqtrain.data import AtomicDataDict, _NODE_FIELDS, _EDGE_FIELDS, _GRAPH_FIELDS

# Import the refactored components
from geqtrain.train.components.setup import setup_loss, setup_metrics
from geqtrain.train.components.inference import run_inference
from geqtrain.train.components.dataset_builder import DatasetBuilder
from geqtrain.train.components.checkpointing import CheckpointHandler
from geqtrain.utils._global_options import apply_global_config


class Evaluator:
    """
    Encapsulates the entire evaluation process for a trained model.
    """
    def __init__(self, model, dataloader, loss_fn, metrics, device, config, loggers):
        self.model = model
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.device = device
        self.config = config
        self.logger, self.csv_loggers = loggers
        self.model.eval()

        # Determine the final set of keys to log
        keys = set()
        if self.loss_fn: keys.update(self.loss_fn.keys)
        if self.metrics: keys.update(self.metrics.keys)
        keys.update(self.config.get('extra_fields_to_log', []))
        self.keys_to_log = sorted(list(keys))
        self.logger.info(f"Logging the following fields: {self.keys_to_log}")

    def run(self):
        """Execute the evaluation loop."""
        self.logger.info("Starting evaluation...")
        
        self.graph_idx_offset = 0
        if self.loss_fn: self.loss_fn.reset(); self.loss_fn.to(self.device)
        if self.metrics: self.metrics.reset(); self.metrics.to(self.device)

        self._write_csv_headers()

        pbar = tqdm(self.dataloader, desc="Evaluating")
        for data in pbar:
            out, ref_data, _, _ = run_inference(
                model=self.model, data=data, device=self.device,
                loss_fn=self.loss_fn or self.metrics, config=self.config.as_dict()
            )
            
            loss_contrib = {}
            if self.loss_fn:
                batch_loss, loss_contrib = self.loss_fn(pred=out, ref=ref_data)
                self.loss_fn.loss_stat(batch_loss, loss_contrib)
            if self.metrics:
                self.metrics(pred=out, ref=ref_data)

            self._log_batch_results(out, ref_data, loss_contrib)
        
        self.logger.info("Evaluation finished. See final results below.")
        self._log_final_summary()

    def _format_tensor(self, tensor: torch.Tensor) -> str:
        """Formats a tensor for CSV logging, handling scalars and vectors."""
        if tensor is None: return "N/A"
        if tensor.numel() == 1: return f"{tensor.item():.6f}"
        vals = [f"{x:.6f}" for x in tensor.flatten().tolist()]
        return f"\"[{'_'.join(vals)}]\""

    def _write_csv_headers(self):
        """Write the headers for each of the detailed output CSV files."""
        for key in self.keys_to_log:
            clean_key = self.loss_fn.remove_suffix(key) if self.loss_fn else self.metrics.remove_suffix(key)
            logger = self.csv_loggers.get(clean_key)
            if not logger: continue

            header = ""
            if clean_key in _GRAPH_FIELDS:
                header = "graph_idx,pred,ref"
                if self.loss_fn and key in self.loss_fn.keys:
                    header += ",loss"
            elif clean_key in _NODE_FIELDS:
                header = "graph_idx,atom_idx,pred,ref"
            elif clean_key in _EDGE_FIELDS:
                header = "graph_idx,src_atom_idx,trg_atom_idx,pred,ref"
            
            logger.info(header)

    def _log_batch_results(self, pred, ref, loss_contrib):
        """Log per-atom, per-edge, or per-graph outputs for the current batch."""
        batch_tensor = ref.get(AtomicDataDict.BATCH_KEY)
        edge_index = ref.get(AtomicDataDict.EDGE_INDEX_KEY)
        num_graphs = 1 if batch_tensor is None else batch_tensor.max().item() + 1

        for key in self.keys_to_log:
            clean_key = self.loss_fn.remove_suffix(key) if self.loss_fn else self.metrics.remove_suffix(key)
            logger = self.csv_loggers.get(clean_key)
            if not logger: continue

            p, r, l = pred.get(clean_key), ref.get(clean_key), loss_contrib.get(key)
            
            for i in range(num_graphs):
                graph_idx = self.graph_idx_offset + i
                
                if clean_key in _GRAPH_FIELDS:
                    pred_val = self._format_tensor(p[i]) if p is not None else "N/A"
                    ref_val = self._format_tensor(r[i]) if r is not None else "N/A"
                    loss_val = self._format_tensor(l) if l is not None else None
                    msg = f"{graph_idx},{pred_val},{ref_val}"
                    if loss_val is not None:
                        msg += f",{loss_val}"
                    logger.info(msg)

                elif clean_key in _NODE_FIELDS:
                    node_mask = (batch_tensor == i)
                    abs_indices = torch.where(node_mask)[0]
                    for atom_idx, abs_idx in enumerate(abs_indices):
                        pred_val = self._format_tensor(p[abs_idx]) if p is not None else "N/A"
                        ref_val = self._format_tensor(r[abs_idx]) if r is not None else "N/A"
                        logger.info(f"{graph_idx},{atom_idx},{pred_val},{ref_val}")

                elif clean_key in _EDGE_FIELDS:
                    node_mask = (batch_tensor == i)
                    # Find edges where the source atom is in the current graph
                    edge_mask = node_mask[edge_index[0]]
                    abs_indices = torch.where(edge_mask)[0]
                    for abs_idx in abs_indices:
                        src_node, trg_node = edge_index[:, abs_idx].tolist()
                        pred_val = self._format_tensor(p[abs_idx]) if p is not None else "N/A"
                        ref_val = self._format_tensor(r[abs_idx]) if r is not None else "N/A"
                        logger.info(f"{graph_idx},{src_node},{trg_node},{pred_val},{ref_val}")

        self.graph_idx_offset += num_graphs

    def _log_final_summary(self):
        """Log the final, aggregated metrics for the entire dataset."""
        self.logger.info("\n--- Final Results ---")
        if self.loss_fn:
            loss_results = self.loss_fn.current_result()
            self.logger.info("Losses:")
            for key, val in loss_results.items(): self.logger.info(f"  {key:<25}: {val:.4f}")
        if self.metrics:
            metric_results = self.metrics.current_result()
            flat_metrics = self.metrics.flatten_metrics(metric_results, self.config.get('metrics_metadata', {}))
            self.logger.info("Metrics:")
            for key, val in flat_metrics.items(): self.logger.info(f"  {key:<25}: {val:.4f}")
        self.logger.info("--- Evaluation Complete ---")


def main(args=None):
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a test dataset.")
    parser.add_argument("model", help="Path to a training checkpoint (`.pth`) model.", type=Path)
    parser.add_argument("dataset_config", help="Path to a YAML config file for the test dataset.", type=Path)
    parser.add_argument("-b", "--batch-size", type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument("-d", "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on.")
    parser.add_argument("-l", "--log", help="Directory to save detailed log files. A timestamped folder will be created inside.")
    parser.add_argument("-e", "--extra-fields-to-log", nargs='+', default=[], help="List of extra fields to log predictions for (e.g., node_features).")
    
    args = parser.parse_args(args)

    # 1. Load Model and its original training config
    model, train_config = CheckpointHandler.load_model_from_training_session(
        traindir=args.model.parent, model_name=args.model.name, device=args.device
    )

    # 2. Load Test Dataset Config and merge it to original training config
    test_config = Config.from_file(args.dataset_config)
    config = train_config
    config.update(test_config)
    config['extra_fields_to_log'] = args.extra_fields_to_log
    apply_global_config(config)

    # 2. Create the dataset
    builder = DatasetBuilder(config, np.random.default_rng(config.get('dataset_seed')))
    final_test_dset = builder.build_test()
    dataloader = DataLoader(final_test_dset, batch_size=args.batch_size, shuffle=False)

    # 3. Setup Loss and Metrics
    loss_fn = setup_loss(config) if config.get("loss_coeffs") else None
    metrics = setup_metrics(config) if config.get("metrics_components") else None
    
    # 4. Setup logging
    loggers = init_loggers(args.log, loss_fn, metrics, config.get('extra_fields_to_log', []))
    
    # 5. Create and run the Evaluator
    evaluator = Evaluator(model, dataloader, loss_fn, metrics, args.device, config, loggers)
    evaluator.run()

def init_loggers(log_dir: str = None, loss_fn=None, metrics=None, extra_fields=[]):
    """Initializes and returns console and a dictionary of CSV loggers."""
    if log_dir is None:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        return logging.getLogger("geqtrain-evaluate"), {}

    # Create a unique, timestamped directory for this run
    run_dir = Path(log_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(exist_ok=True, parents=True)
    
    # Main console logger
    main_logger = logging.getLogger("geqtrain-evaluate")
    main_logger.setLevel(logging.INFO)
    if not main_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        main_logger.addHandler(console_handler)
        main_logger.addHandler(logging.FileHandler(run_dir / "log.txt"))
    
    # Determine all fields that need a CSV logger
    keys_to_log = set(extra_fields)
    if loss_fn: keys_to_log.update(loss_fn.keys)
    if metrics: keys_to_log.update(metrics.keys)
    
    csv_loggers = {}
    for key in keys_to_log:
        clean_key = loss_fn.remove_suffix(key) if loss_fn else metrics.remove_suffix(key)
        if clean_key in csv_loggers: continue

        logger = logging.getLogger(f"csv_{clean_key}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not logger.handlers:
            handler = logging.FileHandler(run_dir / f"{clean_key}.csv")
            logger.addHandler(handler)
        csv_loggers[clean_key] = logger

    return main_logger, csv_loggers

if __name__ == "__main__":
    main()