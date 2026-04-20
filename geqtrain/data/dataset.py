""" Adapted from https://github.com/mir-group/nequip
"""

import traceback
from typing import (
    List,
    Optional,
    Tuple,
    Union,
    Dict
)
import bisect
import numpy as np
import logging
import inspect
import yaml
import hashlib
import torch
import copy
import os
from os.path import dirname, basename, abspath
from e3nn.o3 import Irreps
from typing import Tuple, Dict, Any, List, Union, Optional, Callable

from geqtrain.utils.torch_geometric import Batch, Dataset, Compose
from geqtrain.utils.torch_geometric.data import Data
from geqtrain.utils.torch_geometric.dataset import IndexType
from geqtrain.utils.torch_geometric.utils import download_url, extract_zip
from geqtrain.utils.pytorch_scatter import scatter_mean, scatter_std


import geqtrain
from geqtrain.utils import load_callable, instantiate
from torch.utils.data import ConcatDataset
from geqtrain.data import (
    AtomicData,
    AtomicDataDict,
    _NODE_FIELDS,
    _EDGE_FIELDS,
    _GRAPH_FIELDS,
    _EXTRA_FIELDS,
    _FIXED_FIELDS,
)
from geqtrain.utils.savenload import atomic_write
from geqtrain.utils.normalization import (
    GLOBAL_MODE,
    PER_TYPE_MODE,
    apply_forward_transform,
    fit_transform_parameters,
    resolve_normalization_map,
    serialize_transform_params,
)
from .AtomicData import _process_dict


def fix_batch_dim(arr):
    if arr is None:
        return None
    if len(arr.shape) == 0:
        return arr.reshape(1)
    return arr


def _has_binning(options: Dict) -> bool:
    return any(key in options for key in ("min_value", "max_value", "bin_edges", "bins"))


def _build_bin_edges(options: Dict, num_types: int) -> np.ndarray:
    if "bin_edges" in options:
        bins = np.asarray(options["bin_edges"], dtype=np.float32)
    elif "bins" in options:
        bins = np.asarray(options["bins"], dtype=np.float32)
    else:
        min_value = options.get("min_value")
        max_value = options.get("max_value")
        if min_value is None or max_value is None:
            raise ValueError("Binning requires 'min_value' and 'max_value' or explicit 'bin_edges'.")
        bins = np.linspace(float(min_value), float(max_value), num_types + 1, dtype=np.float32)

    if bins.ndim != 1 or bins.size < 2:
        raise ValueError("Binning edges must be a 1D array with at least two entries.")

    if num_types > 0 and bins.size - 1 != num_types:
        raise ValueError(
            f"Binning edges imply {bins.size - 1} bins, but num_types={num_types}."
        )
    return bins


def _bin_values(values: np.ndarray, options: Dict, num_types: int) -> np.ndarray:
    bins = _build_bin_edges(options, num_types)
    binned = np.digitize(values, bins) - 1
    return np.clip(binned, 0, bins.size - 2)


def parse_attrs(
    _attributes: Dict,
    _fields: Dict,
    _fixed_fields: Dict = {},
) -> Dict[str, Any]:
    """
    Parse attribute fields with support for categorical, numerical, and binned numerical inputs.

    Attribute config schema (node/edge/graph/extra):
      - attribute_type: "categorical" (default) or "numerical"
      - embedding_mode: "embedding" (default), "one_hot", or "positional"
      - num_types: number of categorical bins/classes (required for categorical or binned numerical)
      - can_be_undefined: if True, allows NaN and adds an "unknown" bin at index num_types
      - embedding_dimensionality: used by embedding/positional layers (only enforced downstream)

    Binning for numerical attributes:
      - Provide attribute_type: numerical plus either:
        - min_value and max_value (uniform bins), or
        - bin_edges / bins (explicit bin edges)
      - Values are mapped to integer bins in [0, num_types-1], with NaNs mapped to num_types
        if can_be_undefined is True.
    """
    for key, options in _attributes.items():
        if key in _fields or key in _fixed_fields:

            if key in _fields:
                val: Optional[np.ndarray] = _fields[key]
            elif key in _fixed_fields:
                val: Optional[np.ndarray] = _fixed_fields[key]

            input_val = val
            attribute_type = options.get('attribute_type', 'categorical')
            assert attribute_type in ['categorical', 'numerical']
            embedding_mode = str(options.get('embedding_mode', 'embedding')).lower()
            assert embedding_mode in ['embedding', 'one_hot', 'positional']
            is_binned_numerical = attribute_type == "numerical" and _has_binning(options)

            if attribute_type == 'numerical' and not is_binned_numerical:
                if input_val is not None:
                    input_val = input_val.astype(np.float32)
            else:
                if embedding_mode in ["embedding", "positional"] and "embedding_dimensionality" not in options:
                    continue  # skip if embedding is requested but dimensionality is missing
                if val is None:
                    val = np.array([np.nan], dtype=np.float32)
                if "num_types" not in options:
                    raise ValueError(f"Attribute {key} requires 'num_types' for categorical/binned input.")
                num_types = int(options['num_types'])
                can_be_undefined = options.get('can_be_undefined', False)
                mask = np.isnan(val)
                if np.any(mask):
                    if not can_be_undefined:
                        raise Exception(f"Found NaN value for attribute {key}. If this is allowed set 'can_be_undefined' to True in config file for this attribute.")
                    assert num_types + 1 == options.get('actual_num_types')

                if is_binned_numerical:
                    input_val = _bin_values(val.astype(np.float32), options, num_types)
                else:
                    input_val = val

                input_val[mask] = num_types  # unknown token has value 'num_types'
                input_val = input_val.astype(np.int64)
            if input_val is not None: # input_val can be None if it is computed by a submodule and is not present in the dataset
                if key in _fields:
                    _fields[key] = torch.from_numpy(input_val)
                elif key in _fixed_fields:
                    _fixed_fields[key] = torch.from_numpy(input_val)

    return _fields, _fixed_fields


def compute_per_type_statistics(dataset: Optional[ConcatDataset], field: str, num_types: int, irreps: Optional[Irreps] = None):
    """
    Computes the mean (bias) and standard deviation for a given field, grouped by node type.

    Args:
        dataset (ConcatDataset): The dataset to compute statistics from.
        field (str): The key for the data field to be analyzed (e.g., "energy").
        num_types (int): The total number of node types.
        irreps (e3nn.o3.Irreps, optional): The irreps of the field. If provided, statistics
            are computed on the norm of each irrep component. Defaults to None.

    Returns:
        Tuple[List[float], List[float]]: A tuple containing two lists:
        - The per-type means (biases).
        - The per-type standard deviations.
    """
    if dataset is None:
        return None, None
    
    field_values_list = [data[field] for data in dataset]
    
    if irreps is not None:
        # Compute norm for each irrep and concatenate
        norm_values_list = []
        for values in field_values_list:
            norms = []
            for (mul, ir), slice in zip(irreps, irreps.slices()):
                if ir.l == 0:
                    norms.append(values[:, slice])
                else:
                    norms.append(torch.linalg.norm(values[:, slice], dim=-1, keepdim=True))
            norm_values_list.append(torch.cat(norms, dim=-1))
        all_field_values = torch.cat(norm_values_list, dim=0)
    else:
        all_field_values = torch.cat(field_values_list, dim=0)

    all_node_types = torch.cat([data[AtomicDataDict.NODE_TYPE_KEY] for data in dataset], dim=0).squeeze()

    means = scatter_mean(all_field_values, all_node_types, dim=0, dim_size=num_types)
    stds = scatter_std(all_field_values, all_node_types, dim=0, dim_size=num_types)

    # Fallback for types with no samples to avoid NaN
    for i, std in enumerate(stds):
        if torch.any(torch.isnan(std)) or torch.all(std < 1.e-3):
            stds[i] = torch.where(torch.isnan(std) | (std < 1.e-3), 1.0, std)

    return means, stds

def compute_global_statistics(dataset: Optional[ConcatDataset], field: str, irreps: Optional[Irreps] = None):
    """
    Computes the global mean and standard deviation for a given field.

    Args:
        dataset (ConcatDataset): The dataset to compute statistics from.
        field (str): The key for the data field to be analyzed.
        irreps (e3nn.o3.Irreps, optional): The irreps of the field. If provided, statistics
            are computed on the norm of each irrep component. Defaults to None.

    Returns:
        Tuple[float, float]: A tuple containing:
        - The global mean.
        - The global standard deviation.
    """
    if dataset is None:
        return None, None

    field_values_list = [data[field] for data in dataset]

    if irreps is not None:
        # Compute norm for each irrep and concatenate
        norm_values_list = []
        for values in field_values_list:
            norms = []
            for ir, slice in irreps.slices():
                if ir.l == 0:
                    norms.append(values[:, slice])
                else:
                    norms.append(torch.linalg.norm(values[:, slice], dim=-1, keepdim=True))
            norm_values_list.append(torch.cat(norms, dim=-1))
        all_field_values = torch.cat(norm_values_list, dim=0)
    else:
        all_field_values = torch.cat(field_values_list, dim=0)
    
    mean = torch.mean(all_field_values).item()
    std = torch.std(all_field_values).item()

    return mean, std


class InMemoryConcatDataset(ConcatDataset):

    def __init__(self, datasets):
        super().__init__(datasets)
        self._n_observations = np.diff(self.cumulative_sizes, prepend=0)
        self._ensemble_indices = np.array([dataset.ensemble_index for dataset in datasets])

    @property
    def n_observations(self):
        return self._n_observations

    @property
    def ensemble_indices(self):
        return self._ensemble_indices

    def __getdataset__(self, dataset_idx):
        return self.datasets[dataset_idx]


class LazyLoadingConcatDataset(Dataset):
    datasets_list: List[dict]
    class_name: str
    prefix: str

    def __init__(self, class_name, prefix, datasets_list: List[dict]):
        super().__init__()
        self._class_name       = class_name
        self._prefix           = prefix
        self._lazy_datasets    = [i.pop('lazy_dataset') for i in datasets_list]
        self._ensemble_indices = np.array([i.get(f'{prefix}_ensemble_index') for i in datasets_list])
        self._datasets_list    = datasets_list
        self._cumsum           = None

    @property
    def _n_observations(self):
        # list of num_of_obs present in each npz
        return np.array([len(_idcs) for _idcs in self._lazy_datasets])

    @property
    def n_observations(self):
        return self._n_observations

    @property
    def ensemble_indices(self):
        return self._ensemble_indices

    @property
    def config(self):
        return self._config

    @property
    def cumsum(self):
        if self._cumsum is not None:
            return self._cumsum
        self._cumsum = np.cumsum(self.n_observations)
        return self._cumsum

    def from_indexed_dataset(self, indexed_dataset):
        instance = copy.deepcopy(self)
        instance.set_lazy_datasets(indexed_dataset)
        return instance

    def set_lazy_datasets(self, dataset):
        self._lazy_datasets = dataset
        self._cumsum = None  # Force to recompute cumsum as indices changes

    def __len__(self):
        return sum(self.n_observations)

    @property
    def datasets(self):
        return self._lazy_datasets

    def __getitem__(self, idx):
        '''
        instanciate each NpzDataset,
        if first time of loading of npz, writes it in preprocessed_path
        if not first time loads preprocessed_path/file and instanciates the NpzDataset instead of keeping it in mem
        '''

        # 1) Find dataset_idx to index in _datasets_list (_datasets_list: list of AtomicInMemoryDataset)
        # dataset_idx: index of the npz file that contains the mol requested
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumsum, idx)

        # 2) instanciate AtomicInMemoryDataset
        instance = self.__getdataset__(dataset_idx)

        # 3) index into it: find sample_idx
        # sample_idx: index of the mol in the npz file, already handled by the AtomicInMemoryDataset.get_example
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumsum[dataset_idx - 1]
        return instance[self._lazy_datasets[dataset_idx][sample_idx]]

    def __getdataset__(self, dataset_idx):
        """
        Instantiates a child dataset using the specific configuration
        stored for it in the datasets_list.
        """
        # Retrieve the exact config needed to reconstruct THIS child dataset
        instance_config = self._datasets_list[dataset_idx]['config']
        
        # Instantiate with the retrieved config
        instance, _ = instantiate(
            self._class_name, # dataset type selected for instanciation eg NpzDataset
            prefix=self._prefix, # looks for {prefix}_ in yaml to select ctor params (default: 'dataset')
            positional_args={},
            optional_args=instance_config,
        )

        return instance


class AtomicDataset(Dataset):
    """The base class for all datasets."""

    fixed_fields: Dict[str, Any]
    root: str

    def __init__(
        self,
        root: str,
        ensemble_index: int = 0,
        transforms: Optional[List[str]] = None,
    ):
        '''
        transforms: list of strings that point to the callable function e.g. pkgName.moduleName.transformName
        '''
        self.ensemble_index = ensemble_index
        super().__init__(
            root=root,
            transform=Compose([load_callable(transf) for transf in transforms]) if transforms else None
        )

    def __getitem__(
        self,
        idx: Union[int, np.integer, IndexType],
    ) -> Union["Dataset", Data]:
        r"""In case :obj:`idx` is of type integer, will return the data object
        at index :obj:`idx` (and transforms it in case :obj:`transform` is
        present).
        In case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a list, a
        tuple, a PyTorch :obj:`LongTensor` or a :obj:`BoolTensor`, or a numpy
        :obj:`np.array`, will return a subset of the dataset at the specified
        indices."""
        if (
            isinstance(idx, (int, np.integer))
            or (isinstance(idx, torch.Tensor) and idx.dim() == 0)
            or (isinstance(idx, np.ndarray) and np.isscalar(idx))
        ):

            data = self.get(self.indices()[idx])
            data = data if self.transform is None else self.transform(copy.deepcopy(data)) # Call deepcopy to avoid stacking transforms over epochs
            return data

        else:
            return self.index_select(idx)

    def _get_parameters(self) -> Dict[str, Any]:
        """Get a dict of the parameters used to build this dataset."""
        pnames = list(inspect.signature(self.__init__).parameters)
        IGNORE_KEYS = {
            "embedding_dimensionality",
            "transforms",
        }

        def filter_attributes(self, pnames, IGNORE_KEYS):
            def filter_dict(d):
                # Recursively filter dictionary to exclude keys in IGNORE_KEYS
                return {
                    k: (filter_dict(v) if isinstance(v, dict) else v)
                    for k, v in d.items()
                    if k not in IGNORE_KEYS
                }

            params = {
                k: (
                    filter_dict(getattr(self, k))
                    if isinstance(getattr(self, k), dict) else getattr(self, k)
                )
                for k in pnames
                if k not in IGNORE_KEYS and hasattr(self, k)
            }

            return params

        params = filter_attributes(self, pnames, IGNORE_KEYS)
        if hasattr(self, "_normalization_specs") and len(getattr(self, "_normalization_specs", {})) > 0:
            # Bump cache key when standardization math changes.
            params["standardization_impl_version"] = 3
        # Add other relevant metadata:
        params["dtype"] = str(torch.float32)
        params["geqtrain_version"] = geqtrain.__version__
        return params

    @property
    def processed_dir(self) -> str:
        # We want the file name to change when the parameters change
        # So, first we get all parameters:
        params = self._get_parameters()
        # Make some kind of string of them:
        # we don't care about this possibly changing between python versions,
        # since a change in python version almost certainly means a change in
        # versions of other things too, and is a good reason to recompute
        buffer = yaml.dump(params).encode("ascii")
        # And hash it:
        param_hash = hashlib.sha1(buffer).hexdigest()
        return f"{self.root}/processed_datasets/processed_dataset_{param_hash}"


class AtomicInMemoryDataset(AtomicDataset):
    r"""Base class for all datasets that fit in memory.

    Please note that, as a ``pytorch_geometric`` dataset, it must be backed by some kind of disk storage.
    By default, the raw file will be stored at root/raw and the processed torch
    file will be at root/process.

    Subclasses must implement:
     - ``raw_file_names``
     - ``get_data()``

    Subclasses may implement:
     - ``download()`` or ``self.url`` or ``ClassName.URL``

    Args:
        root (str, optional): Root directory where the dataset should be saved. Defaults to current working directory.
        file_name (str, optional): file name of data source. only used in children class
        url (str, optional): url to download data source
        extra_fixed_fields (dict, optional): extra key that are not stored in data but needed for AtomicData initialization
        include_frames (list, optional): the frames to process with the constructor.
        target_indices (list, optional): If the dataset has multiple targets, you can optionally select a subset.
        target_key (str, optional): If 'target_indices' is given, also specify which is the target key.
    """

    def __init__(
        self,
        root: str,
        ensemble_index: int,
        file_name: Optional[str] = None,
        url: Optional[str] = None,
        ignore_fields: List = [],
        extra_fixed_fields: Dict[str, Any] = {},
        include_frames: Optional[List[int]] = None,
        target_indices: Optional[List[int]] = None,
        target_key: Optional[str] = None,
        node_attributes: Dict = {},
        edge_attributes: Dict = {},
        graph_attributes: Dict = {},
        extra_attributes: Dict = {},
        normalization: Optional[Dict[str, Any]] = None,
        transforms: Optional[List[Callable]] = None,
    ):
        self.file_name = getattr(type(self), "FILE_NAME", None) if file_name is None else file_name
        self.url = getattr(type(self), "URL", url)

        ignore_fields.extend([AtomicDataDict.R_MAX_KEY, AtomicDataDict.DATASET_RAW_FILE_NAME])
        self.ignore_fields = ignore_fields
        self.extra_fixed_fields = extra_fixed_fields
        self.include_frames = include_frames
        self.target_indices = target_indices
        self.target_key = target_key

        self.data: Optional[Batch] = None
        self.fixed_fields = None

        self.node_attributes = node_attributes
        self.edge_attributes = edge_attributes
        self.graph_attributes = graph_attributes
        self.extra_attributes = extra_attributes
        self.normalization = normalization or {}
        self._normalization_specs = resolve_normalization_map(
            {"normalization": self.normalization},
        )
        self._normalization_specs = {
            field: spec
            for field, spec in self._normalization_specs.items()
            if bool(spec.get("apply_on_dataset", True))
        }
        self.means = {}
        self.stds = {}


        # !!! don't delete this block.
        # otherwise the inherent children class
        # will ignore the download function here
        class_type = type(self)
        if class_type != AtomicInMemoryDataset:
            if "download" not in self.__class__.__dict__:
                class_type.download = AtomicInMemoryDataset.download
            if "process" not in self.__class__.__dict__:
                class_type.process = AtomicInMemoryDataset.process

        # Initialize the InMemoryDataset, which runs download and process
        # See https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html#creating-in-memory-datasets
        # Then pre-process the data if disk files are not found
        # disk files are:
        # for each .npz exists a folder in /processed_datasets, the folder is named via unique hash
        # eg: ['/processed_datasets/processed_dataset_51e456f.../data.pth', '/processed_datasets/processed_dataset_51e456f.../params.yaml']
        # each mol can be loaded in ram via .pth
        # for the not-in-memory version files are written once and reloaded every time the npz is sampled via dataloader
        super().__init__(root=root, ensemble_index=ensemble_index, transforms=transforms)
        if self.data is None:
            self.data, self.fixed_fields, include_frames = torch.load(  # load hashed (already) processed data
                self.processed_paths[0],
                weights_only=False,
            )
            if not np.all(include_frames == self.include_frames):
                raise ValueError(f"the include_frames is changed. Please delete the processed folder and rerun {self.processed_paths[0]}")
        if self.target_indices is not None:
            assert self.target_key is not None
            self.data[self.target_key] = self.data[self.target_key][..., np.array(self.target_indices)]

    def len(self):
        if self.data is None:
            return 0
        return self.data.num_graphs

    @property
    def raw_file_names(self):
        raise NotImplementedError()

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pth", "params.yaml"]

    @property
    def config(self) -> Dict[str, Any]:
        return {
            k: self.data[k]
            for k in [
            ] if k in self.data
        }

    def get_data(
        self,
    ) -> Union[Tuple[Dict[str, Any], Dict[str, Any]], List[AtomicData]]:
        """Get the data --- called from ``process()``, can assume that ``raw_file_names()`` exist.

        Note that parameters for graph construction such as ``pbc`` and ``r_max`` should be included here as (likely, but not necessarily, fixed) fields.

        Returns:
        A two-tuple of:
            fields: dict
                mapping a field name ('pos', 'cell') to a list-like sequence of tensor-like objects giving that field's value for each example.
            fixed_fields: dict
                mapping field names to their constant values for every example in the dataset.
        Or:
            data_list: List[AtomicData]
        """
        raise NotImplementedError

    def download(self):
        if (not hasattr(self, "url")) or (self.url is None):
            # Don't download, assume present. Later could have FileNotFound if the files don't actually exist
            pass
        else:
            download_path = download_url(self.url, self.raw_dir)
            if download_path.endswith(".zip"):
                extract_zip(download_path, self.raw_dir)

    def save_processed(self):
        """Save the processed data to disk."""
        with atomic_write(self.processed_paths[0], binary=True) as f:
            torch.save((self.data, self.fixed_fields, self.include_frames), f)
        with atomic_write(self.processed_paths[1], binary=False) as f:
            yaml.dump(self._get_parameters(), f)
        logging.info(f"Saved filtered processed data to disk at {self.processed_paths[0]}")

    def _is_distribution_candidate(self, field: str, values: torch.Tensor) -> bool:
        if not torch.is_tensor(values):
            return False
        if not torch.is_floating_point(values):
            return False
        if field.endswith("__mask__") or field.startswith("_"):
            return False
        excluded = {
            AtomicDataDict.POSITIONS_KEY,
            AtomicDataDict.EDGE_INDEX_KEY,
            AtomicDataDict.NODE_TYPE_KEY,
            AtomicDataDict.BATCH_KEY,
            "ptr",
            AtomicDataDict.CELL_KEY,
            AtomicDataDict.PBC_KEY,
            AtomicDataDict.R_MAX_KEY,
            AtomicDataDict.DATASET_RAW_FILE_NAME,
            AtomicDataDict.ENSEMBLE_INDEX_KEY,
        }
        return field not in excluded

    def _select_distribution_fields(self, data_list: List[AtomicData], normalization_specs: Dict[str, Dict]) -> List[str]:
        fields = set()
        if len(data_list) == 0:
            return []

        # Always prioritize explicitly normalized targets.
        for field in normalization_specs:
            if field in data_list[0]:
                fields.add(field)

        # Include other floating non-structural fields as likely training targets.
        for field in data_list[0].keys:
            values = data_list[0][field]
            if self._is_distribution_candidate(field, values):
                fields.add(field)

        return sorted(fields)

    def _sample_flattened_tensor(self, values: torch.Tensor, max_points: int = 200000) -> torch.Tensor:
        flat = values.reshape(-1).detach().to(dtype=torch.float32, device="cpu")
        flat = flat[torch.isfinite(flat)]
        if flat.numel() <= max_points:
            return flat
        step = max(1, flat.numel() // max_points)
        return flat[::step][:max_points]

    def _collect_data_list_distribution_samples(
        self,
        data_list: List[AtomicData],
        fields: List[str],
        max_points: int = 200000,
    ) -> Dict[str, torch.Tensor]:
        out = {}
        for field in fields:
            chunks = []
            for entry in data_list:
                if field not in entry:
                    continue
                values = entry[field]
                if self._is_distribution_candidate(field, values):
                    chunks.append(values)
            if len(chunks) == 0:
                continue
            cat = torch.cat([c.reshape(-1) for c in chunks], dim=0)
            sample = self._sample_flattened_tensor(cat, max_points=max_points)
            if sample.numel() > 0:
                out[field] = sample
        return out

    def _collect_batch_distribution_samples(
        self,
        data: Batch,
        fields: List[str],
        max_points: int = 200000,
    ) -> Dict[str, torch.Tensor]:
        out = {}
        for field in fields:
            if field not in data:
                continue
            values = data[field]
            if not self._is_distribution_candidate(field, values):
                continue
            sample = self._sample_flattened_tensor(values, max_points=max_points)
            if sample.numel() > 0:
                out[field] = sample
        return out

    def _save_distribution_plots(
        self,
        raw_samples: Dict[str, torch.Tensor],
        normalized_samples: Dict[str, torch.Tensor],
    ) -> None:
        if len(raw_samples) == 0 and len(normalized_samples) == 0:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            logging.warning(f"Could not save target distribution plots (matplotlib unavailable): {exc}")
            return

        os.makedirs(self.processed_dir, exist_ok=True)

        def _safe_name(name: str) -> str:
            return name.replace("/", "_").replace(":", "_")

        def _plot(values: torch.Tensor, title: str, out_path: str):
            arr = values.numpy()
            if arr.size == 0:
                return
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(arr, bins=100)
            ax.set_title(title)
            ax.set_xlabel("value")
            ax.set_ylabel("count")
            fig.tight_layout()
            fig.savefig(out_path, dpi=150)
            plt.close(fig)

        for field, values in raw_samples.items():
            _plot(
                values,
                title=f"{field} distribution (raw)",
                out_path=os.path.join(self.processed_dir, f"target_dist_{_safe_name(field)}_raw.png"),
            )
        for field, values in normalized_samples.items():
            _plot(
                values,
                title=f"{field} distribution (normalized)",
                out_path=os.path.join(self.processed_dir, f"target_dist_{_safe_name(field)}_normalized.png"),
            )

    def process(self):
        data = self.get_data()
        if len(data) == 5:

            # Get our data
            node_fields, edge_fields, graph_fields, extra_fields, fixed_fields = data

            fixed_fields.update(self.extra_fixed_fields)
            fixed_fields[AtomicDataDict.DATASET_RAW_FILE_NAME] = self.raw_file_names[0]

            # node fields
            node_fields, fixed_fields = parse_attrs(
                _attributes=self.node_attributes,
                _fields=node_fields,
                _fixed_fields=fixed_fields,
            )

            # edge fields
            edge_fields, fixed_fields = parse_attrs(
                _attributes=self.edge_attributes,
                _fields=edge_fields,
                _fixed_fields=fixed_fields,
            )

            # graph fields
            graph_fields, fixed_fields = parse_attrs(
                _attributes=self.graph_attributes,
                _fields=graph_fields,
                _fixed_fields=fixed_fields,
            )

            # check keys and ensure 1d arrays become 2d with shape (d, 1)
            node_fields  = {k:v for k,v in node_fields.items()  if v is not None}
            edge_fields  = {k:v for k,v in edge_fields.items()  if v is not None}
            graph_fields = {k:v for k,v in graph_fields.items() if v is not None}
            extra_fields = {k:v for k,v in extra_fields.items() if v is not None}

            all_keys = set(node_fields.keys()).union(edge_fields.keys()).union(graph_fields.keys()).union(extra_fields.keys()).union(fixed_fields.keys())
            assert len(all_keys) == len(node_fields) + len(edge_fields) + len(graph_fields) + len(extra_fields) + len(fixed_fields), "No overlap in keys between data and fixed_fields allowed!"
            # Check bad key combinations, but don't require that this be a graph yet.
            AtomicDataDict.validate_keys(all_keys, graph_required=False)

            # check dimesionality
            num_examples = set([len(x) for x in [val for val in node_fields.values() if val is not None]])
            if not len(num_examples) == 1:
                err_dict = {f: v.shape for f, v in node_fields.items()}
                _dir = self.raw_dir + "/" + self.raw_file_names[0]
                raise ValueError(f"Dataset {_dir} is invalid: expected all node_fields to have same length (same number of examples), but they had shapes {err_dict}")
            num_examples = next(iter(num_examples))

            # Check that the number of frames is consistent for all node and edge fields
            assert all([len(v) == num_examples for v in node_fields.values() if v is not None])
            # assert all([len(v) == num_examples for v in edge_fields.values() if v is not None]) !!! TODO

            include_frames = self.include_frames  # all frames by default
            if include_frames is None:
                include_frames = range(num_examples)

            # Make AtomicData from it:
            if AtomicDataDict.EDGE_INDEX_KEY in all_keys:
                # This is already a graph, just build it
                constructor = AtomicData.with_edge_index
            else:
                # do neighborlist from points
                constructor = AtomicData.from_points
                assert AtomicDataDict.R_MAX_KEY in all_keys
                assert AtomicDataDict.POSITIONS_KEY in all_keys

            data_list = [  # list of AtomicData-pyg-object objects
                constructor(
                    **{
                        **{f: v[i] for f, v in node_fields.items()  if v is not None},
                        **{f: v[i] for f, v in edge_fields.items()  if v is not None},
                        **{f: v[i] for f, v in graph_fields.items() if v is not None},
                        **{f: v[i] for f, v in extra_fields.items() if v is not None},
                        **fixed_fields,
                    }, ignore_fields=self.ignore_fields)
                for i in include_frames
            ]

        else:
            raise ValueError("Invalid return from `self.get_data()`")

        normalization_specs = copy.deepcopy(self._normalization_specs)
        distribution_fields = self._select_distribution_fields(data_list, normalization_specs)
        raw_distribution_samples = self._collect_data_list_distribution_samples(
            data_list,
            distribution_fields,
        )
        for field, spec in normalization_specs.items():
            if field not in data_list[0]:
                raise ValueError(f"Cannot normalize: field `{field}` not in data.")

            irreps_str = spec.get("irreps")
            irreps = Irreps(irreps_str) if irreps_str else None
            transform_cfg = fit_transform_parameters(
                values=torch.cat([entry[field] for entry in data_list], dim=0),
                transform_cfg=spec.get("transform", {"name": "none"}),
                irreps=irreps,
            )
            spec["transform"] = transform_cfg

            if transform_cfg.get("name", "none") != "none":
                for entry in data_list:
                    entry[field] = apply_forward_transform(
                        entry[field],
                        transform_cfg,
                        irreps=irreps,
                    ).to(entry[field].dtype)

            fixed_fields.update(serialize_transform_params(field, transform_cfg))

        # Batch it for efficient saving
        # This limits an AtomicInMemoryDataset to a maximum of LONG_MAX atoms _overall_, but that is a very big number and any dataset that large is probably not "InMemory" anyway
        data = Batch.from_data_list(data_list, exclude_keys=fixed_fields.keys())

        # Define prefixes for standardization keys
        MEAN_KEY_PREFIX = "_mean_"
        STD_KEY_PREFIX = "_std_"
        PER_TYPE_PREFIX = "per_type"
        GLOBAL_PREFIX = "global"

        for field, spec in normalization_specs.items():
            if field not in data:
                raise ValueError(f"Cannot normalize: field `{field}` not in data.")

            mode = spec.get("mode")
            irreps_str = spec.get("irreps")
            irreps = Irreps(irreps_str) if irreps_str else None

            if mode not in [PER_TYPE_MODE, GLOBAL_MODE]:
                raise ValueError(
                    f"Invalid normalization mode '{mode}' for field '{field}'. "
                    f"Must be one of: '{PER_TYPE_MODE}', '{GLOBAL_MODE}'."
                )

            if mode == PER_TYPE_MODE:
                if len(data_list) == 0 or AtomicDataDict.NODE_TYPE_KEY not in data_list[0]:
                    raise ValueError("`standardize_per_type` is True, but node types are not available in the data.")
                num_types = self.node_attributes.get(AtomicDataDict.NODE_TYPE_KEY, {}).get('num_types')
                if num_types is None:
                    raise ValueError("`num_types` for node types must be provided in `node_attributes` for per-type standardization.")
                
                mean_vals, std_vals = compute_per_type_statistics(data_list, field, num_types, irreps=irreps)
                # For l > 0 irreps, only std scaling is invertible on vectors.
                # Keep per-type means only for scalar (l=0) components.
                mean_vals_to_store = mean_vals
                if irreps:
                    mean_vals_to_store = mean_vals.clone()
                    i = 0
                    for mul, ir in irreps:
                        if ir.l > 0:
                            mean_vals_to_store[:, i:i + 1] = 0.0
                        i += 1

                self.means[field] = mean_vals_to_store.tolist()
                self.stds[field] = std_vals.tolist()
                
                mean_tensor = mean_vals_to_store.to(dtype=data[field].dtype)
                std_tensor = std_vals.to(dtype=data[field].dtype)
                node_types = torch.cat(
                    [d[AtomicDataDict.NODE_TYPE_KEY] for d in data_list], dim=0
                ).squeeze().to(dtype=torch.long, device=data[field].device)

                if node_types.shape[0] != data[field].shape[0]:
                    raise ValueError(
                        f"Per-type standardization for field '{field}' requires a node-level field "
                        f"with first dimension matching the number of nodes, but got "
                        f"{data[field].shape[0]} values and {node_types.shape[0]} node types."
                    )

                if irreps:
                    means_expanded = mean_tensor[node_types]
                    stds_expanded = std_tensor[node_types]
                    i = 0
                    for (mul, ir), slice in zip(irreps, irreps.slices()):
                        if ir.l == 0:
                            data[field][:, slice] -= means_expanded[:, i:i+1]
                            data[field][:, slice] /= stds_expanded[:, i:i+1]
                        else: # l > 0, std-only scaling to preserve invertibility
                            data[field][:, slice] /= stds_expanded[:, i:i+1]
                        i += 1
                else:
                    data[field] -= mean_tensor[node_types].view(data[field].shape)
                    data[field] /= std_tensor[node_types].view(data[field].shape)
                
                # Store stats in fixed_fields
                mean_key = f"{MEAN_KEY_PREFIX}.{PER_TYPE_PREFIX}.{field}"
                std_key = f"{STD_KEY_PREFIX}.{PER_TYPE_PREFIX}.{field}"
                fixed_fields[mean_key] = mean_vals_to_store
                fixed_fields[std_key] = std_vals
                logging.info(f"Standardized field '{field}' per type.")

            elif mode == GLOBAL_MODE:
                mean_val, std_val = compute_global_statistics(data_list, field, irreps=irreps)
                self.means[field] = mean_val
                self.stds[field] = std_val
                
                if std_val > 1e-8:
                    if irreps:
                        raise NotImplementedError("Global standardization for equivariant fields is not yet implemented.")
                    else:
                        data[field] -= mean_val
                        data[field] /= std_val

                    # Store stats in fixed_fields
                    mean_key = f"{MEAN_KEY_PREFIX}.{GLOBAL_PREFIX}.{field}"
                    std_key = f"{STD_KEY_PREFIX}.{GLOBAL_PREFIX}.{field}"
                    fixed_fields[mean_key] = torch.tensor(mean_val)
                    fixed_fields[std_key] = torch.tensor(std_val)
                    logging.info(f"Standardized field '{field}' globally with mean={mean_val:.4f} and std={std_val:.4f}.")                    
                else:
                    logging.warning(f"Standard deviation of field '{field}' is very small ({std_val:.4f}), skipping standardization.")

        normalized_distribution_samples = {}
        if len(normalization_specs) > 0:
            normalized_distribution_samples = self._collect_batch_distribution_samples(
                data,
                [f for f in distribution_fields if f in normalization_specs],
            )
        self._save_distribution_plots(raw_distribution_samples, normalized_distribution_samples)

        del data_list
        del node_fields
        del edge_fields
        del graph_fields

        # type conversion. ignore_fields: fields mapped in yaml but not casted to tensor
        _process_dict(fixed_fields, ignore_fields=self.ignore_fields)

        total_MBs = sum(item.numel() * item.element_size() for _, item in data) / (1024 * 1024)
        logging.info(f"Loaded data: {data}\n    processed data size: ~{total_MBs:.2f} MB")
        del total_MBs

        self.data = data
        self.fixed_fields = fixed_fields

        # Save the processed data
        self.save_processed()
        logging.info("Cached processed data to disk")

    def get(self, idx):
        out = self.data.get_example(idx)
        
        # Add back fixed fields
        for f, v in self.fixed_fields.items():
            out[f] = v

        # If 'ensemble_index' is loaded from the npz, it will be a tensor for the specific frame.
        # We use it and convert to a scalar integer.
        # Otherwise, if it's not present, we fall back to the dataset's default integer index.
        if AtomicDataDict.ENSEMBLE_INDEX_KEY in out:
            # It's a 1-element tensor (e.g., tensor([3])), get the scalar value
            out.ensemble_index = int(out.ensemble_index.item())
        else:
            out.ensemble_index = self.ensemble_index
            
        return out


class NpzDataset(AtomicInMemoryDataset):
    """Load data from an npz file.

    To avoid loading unneeded data, keys are ignored by default unless they are in ``key_mapping``
    or ``extra_fixed_fields``.

    Args:
        key_mapping (Dict[str, str]): mapping of npz keys to ``AtomicData`` keys. Optional
        include_keys (list): the attributes to be processed and stored. Optional
        npz_fixed_field_keys: the attributes that only have one instance but apply to all frames. Optional
            Note that the mapped keys (as determined by the _values_ in ``key_mapping``) should be used in
            ``npz_fixed_field_keys``, not the original npz keys from before mapping. If an npz key is not
            present in ``key_mapping``, it is mapped to itself, and this point is not relevant.

    Example: Given a npz file with 10 configurations, each with 14 nodes.

        position:    (10, 14, 3)
        node_types:  (14)
        user_label1: (10)        # per graph
        user_label2: (10, 14, 3) # per node

    The input yaml should be

    ```yaml
    dataset: npz
    dataset_file_name: path/to/example.npz
    include_keys:
      - user_label1
      - user_label2
    npz_fixed_field_keys:
      - node_types
    key_mapping:
      position: pos
      node_types: node_types
    ```
    """

    def __init__(
        self,
        root: str,
        ensemble_index: int,
        key_mapping: Dict[str, str] = {},
        file_name: Optional[str] = None,
        url: Optional[str] = None,
        ignore_fields: List = [],
        extra_fixed_fields: Dict[str, Any] = {},
        include_frames: Optional[List[int]] = None,
        target_indices: Optional[List[int]] = None,
        target_key: Optional[str] = None,
        node_attributes: Dict = {},
        edge_attributes: Dict = {},
        graph_attributes: Dict = {},
        extra_attributes: Dict = {},
        normalization: Optional[Dict[str, Any]] = None,
        transforms: Optional[List[str]] = None,
    ):
        self.key_mapping = key_mapping

        try:
            super().__init__(
                file_name=file_name,
                ensemble_index=ensemble_index,
                url=url,
                root=root,
                ignore_fields=ignore_fields,
                extra_fixed_fields=extra_fixed_fields,
                include_frames=include_frames,
                target_indices=target_indices,
                target_key=target_key,
                node_attributes=node_attributes,
                edge_attributes=edge_attributes,
                graph_attributes=graph_attributes,
                extra_attributes=extra_attributes,
                normalization=normalization,
                transforms=transforms,
            )
        except Exception as e:
            logging.error(f"Error in file {self.raw_file_names}. Traceback: {traceback.format_exc()}")
            raise e

    @property
    def raw_file_names(self):
        return [basename(self.file_name)]

    @property
    def raw_dir(self):
        return dirname(abspath(self.file_name))

    def get_data(self):

        # loads each sing .npz and get all keys + maps wrt yaml keys
        _dir = self.raw_dir + "/" + self.raw_file_names[0]
        print(_dir)
        try:
            data = np.load(_dir, allow_pickle=True, mmap_mode='r')
        except:
            logging.error(f"File {_dir} not found")
            raise FileNotFoundError(f"File {_dir} not found")

        # only the keys explicitly mentioned in the yaml file will be parsed (registered via register fields section)
        keys = set(list(self.key_mapping.keys()))
        # Check that all mapped keys exist in the data
        ################ AT INFERENCE TIME SOME KEYS MAY BE MISSING
        # missing_keys = [k for k in self.key_mapping.keys() if k not in data]
        # if missing_keys:
        #     raise KeyError(f"The following mapped keys are missing in the npz data: {missing_keys}")
        ################
        keys.update(list(self.extra_fixed_fields.keys()))
        keys = keys.intersection(set(list(data.keys())))

        mapped = {self.key_mapping.get(k, k): data[k] for k in keys}
        for structure_fields in [_NODE_FIELDS, _EDGE_FIELDS, _GRAPH_FIELDS]:
            for k in structure_fields:
                if k not in mapped:
                    mapped[k] = None

        # note that we don't deal with extra_fixed_fields here; AtomicInMemoryDataset does that.
        fixed_fields = {
            k: fix_batch_dim(v) for k, v in mapped.items()
            if  self.node_attributes.get(k, {}).get('fixed', False)
            or  self.edge_attributes.get(k, {}).get('fixed', False)
            or self.graph_attributes.get(k, {}).get('fixed', False)
            or self.extra_attributes.get(k, {}).get('fixed', False)
            or k in _FIXED_FIELDS
        }

        node_fields = {
            k: v for k, v in mapped.items()
            if (k in _NODE_FIELDS) and (k not in fixed_fields.keys())
        }

        edge_fields = {
            k: v for k, v in mapped.items()
            if (k in _EDGE_FIELDS.union([AtomicDataDict.EDGE_INDEX_KEY])) and (k not in fixed_fields.keys())
        }

        graph_fields = {
            k: fix_batch_dim(v) for k, v in mapped.items()
            if (k in _GRAPH_FIELDS) and (k not in fixed_fields.keys())
        }

        extra_fields = {
            k: fix_batch_dim(v) for k, v in mapped.items()
            if (k in _EXTRA_FIELDS) and (k not in fixed_fields.keys())
        }

        for key in mapped.keys():
            for fields in [node_fields, edge_fields, graph_fields, extra_fields, fixed_fields]:
                if key in fields and fields[key] is not None and np.issubdtype(fields[key].dtype, np.integer):
                    fields[key] = fields[key].astype(np.int64) # keep int64 since cross entropy based pytorch loss functions require this dtype
                if key in fields and fields[key] is not None and np.issubdtype(fields[key].dtype, bool):
                    fields[key] = fields[key].astype(np.float32)

        # k:v k field, v the value grabbed from the npz in np.array form
        return node_fields, edge_fields, graph_fields, extra_fields, fixed_fields
