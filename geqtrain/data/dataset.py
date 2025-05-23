""" Adapted from https://github.com/mir-group/nequip
"""

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
from os.path import dirname, basename, abspath
from typing import Tuple, Dict, Any, List, Union, Optional, Callable

from geqtrain.utils.torch_geometric import Batch, Dataset, Compose
from geqtrain.utils.torch_geometric.data import Data
from geqtrain.utils.torch_geometric.dataset import IndexType
from geqtrain.utils.torch_geometric.utils import download_url, extract_zip


import geqtrain
from geqtrain.utils import load_callable
from geqtrain.data import (
    AtomicData,
    AtomicDataDict,
    _NODE_FIELDS,
    _EDGE_FIELDS,
    _GRAPH_FIELDS,
    _EXTRA_FIELDS,
)
from geqtrain.utils.savenload import atomic_write
from .AtomicData import _process_dict
from torch.utils.data import ConcatDataset

from geqtrain.utils import (
    instantiate,
    get_w_prefix,
    Config
)


def fix_batch_dim(arr):
    if arr is None:
        return None
    if len(arr.shape) == 0:
        return arr.reshape(1)
    return arr


def parse_attrs(
    _attributes: Dict,
    _fields: Dict,
    _fixed_fields: Dict = {},
) -> Dict[str, Any]:
    '''
    parses field properties

    handles:

    num_types: 8
    node_attributes:
      node_types: # this kword must match the red kword in key_mapping
        # num_types: 8 # 5 ! a +1 is always added due to "unspecified" class
        embedding_dimensionality: 16
        fixed: true # if equal for each frame, if so they must not have the batch dim in the npz
        # unspecified: t/f

    todo: describe logic
    '''
    for key, options in _attributes.items():
        if key in _fields or key in _fixed_fields:

            if key in _fields:
                val: Optional[np.ndarray] = _fields[key]
            elif key in _fixed_fields:
                val: Optional[np.ndarray] = _fixed_fields[key]

            if options.get('attribute_type', 'categorical') == 'numerical':
                _input_type = val
            else:
                if "embedding_dimensionality" not in options:  # this is not an attribute to parse
                    continue
                if val is None:
                    val = np.array([np.nan])
                num_types = int(options['num_types'])
                can_be_undefined = options.get('can_be_undefined', False)
                if 'min_value' in options or 'max_value' in options:
                    mask = np.isnan(val)
                    if np.any(mask) and not can_be_undefined:
                        raise Exception(f"Found NaN value for attribute {key}. If this is allowed set 'can_be_undefined' to True in config file for this attribute.")
                    val[mask] = float(options['max_value'])
                    # goes from 0 to 'num_types' (excluded). You have  'num_types' bins between 'min_value' and 'max_value'.
                    # values smaller than 'min_value' or greater than 'max_value' are included in the smallest/largest bins
                    # the actual number of bins is 'num_types' [+ 1 if can_be_undefined is True]
                    # e.g. 'min_value' 0, 'max_value' 20, 'num_types' 4 and can_be_undefined=True becomes [-inf<5 | 5<10 | 10<15 | 15<+inf | unknown]
                    bins = np.linspace(float(options['min_value']), float(options['max_value']), num_types)
                    _input_type = np.digitize(val, bins)
                    _input_type[_input_type == num_types] -= 1
                    _input_type[_input_type > 0] -= 1
                    _input_type[mask] = num_types
                else:
                    mask = np.isnan(val)
                    if np.any(mask) and not can_be_undefined:
                        raise Exception(f"Found NaN value for attribute {key}. If this is allowed set 'can_be_undefined' to True in config file for this attribute.")
                    _input_type = val.astype(np.int64)
                    _input_type[mask] = num_types
                    # 'unkown' token has value 'num_types', while defined tokens have range [0, 'num_types')
            if key in _fields:
                _fields[key] = torch.from_numpy(_input_type)
            elif key in _fixed_fields:
                _fixed_fields[key] = torch.from_numpy(_input_type)

    return _fields, _fixed_fields

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

    def __init__(self, class_name, prefix, config, datasets_list: List[dict]):
        super().__init__()
        self._class_name       = class_name
        self._prefix           = prefix
        self._config           = config
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

        # 1) Find dataset_idx to index in _datasets_list (_datasets_list: list of NpzDataset)
        # dataset_idx: index of the npz file that contains the mol requested
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumsum, idx)

        # 2) instanciate NpzDataset
        instance = self.__getdataset__(dataset_idx)

        # 3) index into it: find sample_idx
        # sample_idx: index of the mol in the npz file, already handled by the NpzDataset.get_example
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumsum[dataset_idx - 1]
        return instance[self._lazy_datasets[dataset_idx][sample_idx]]

    def __getdataset__(self, dataset_idx):
        _config = copy.deepcopy(self.config)
        file_name_key = f"{self._prefix}_file_name"
        _config[file_name_key] = self._datasets_list[dataset_idx][file_name_key]
        ensemble_index_key = f"{self._prefix}_ensemble_index"
        _config[ensemble_index_key] = self._datasets_list[dataset_idx][ensemble_index_key]

        instance, _ = instantiate(
            self._class_name, # dataset type selected for instanciation eg NpzDataset
            prefix=self._prefix, # looks for {prefix}_ in yaml to select ctor params (default: 'dataset')
            positional_args={},
            optional_args=_config, # the whole yaml parsed and wrapped in a dict
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
        pbc: bool = False,
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
        transforms: Optional[List[Callable]] = None,
    ):
        self.pbc = pbc
        self.file_name = getattr(type(self), "FILE_NAME", None) if file_name is None else file_name
        self.url = getattr(type(self), "URL", url)

        ignore_fields.extend([AtomicDataDict.R_MAX_KEY, AtomicDataDict.DATASET_RAW_FILE_NAME])
        self.ignore_fields = ignore_fields
        self.extra_fixed_fields = extra_fixed_fields
        self.include_frames = include_frames
        self.target_indices = target_indices
        self.target_key = target_key

        self.data = None
        self.fixed_fields = None

        self.node_attributes = node_attributes
        self.edge_attributes = edge_attributes
        self.graph_attributes = graph_attributes
        self.extra_attributes = extra_attributes

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
                raise ValueError(f"This dataset is invalid: expected all node_fields to have same length (same number of examples), but they had shapes {err_dict}")
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
                    }, pbc=self.pbc, ignore_fields=self.ignore_fields)
                for i in include_frames
            ]

        else:
            raise ValueError("Invalid return from `self.get_data()`")

        # Batch it for efficient saving
        # This limits an AtomicInMemoryDataset to a maximum of LONG_MAX atoms _overall_, but that is a very big number and any dataset that large is probably not "InMemory" anyway
        data = Batch.from_data_list(data_list, exclude_keys=fixed_fields.keys())
        del data_list
        del node_fields
        del edge_fields
        del graph_fields

        # type conversion. ignore_fields: fields mapped in yaml but not casted to tensor
        _process_dict(fixed_fields, ignore_fields=self.ignore_fields)

        total_MBs = sum(item.numel() * item.element_size() for _, item in data) / (1024 * 1024)
        logging.info(f"Loaded data: {data}\n    processed data size: ~{total_MBs:.2f} MB")
        del total_MBs

        # use atomic writes to avoid race conditions between
        # different trainings that use the same dataset
        # since those separate trainings should all produce the same results,
        # it doesn't matter if they overwrite each others cached'
        # datasets. It only matters that they don't simultaneously try
        # to write the _same_ file, corrupting it.
        with atomic_write(self.processed_paths[0], binary=True) as f:
            torch.save((data, fixed_fields, self.include_frames), f)
        with atomic_write(self.processed_paths[1], binary=False) as f:
            yaml.dump(self._get_parameters(), f)

        logging.info("Cached processed data to disk")

        self.data = data
        self.fixed_fields = fixed_fields

    def get(self, idx):
        out = self.data.get_example(idx)
        out.ensemble_index = self.ensemble_index
        # Add back fixed fields
        for f, v in self.fixed_fields.items():
            out[f] = v
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
        pbc: bool = False,
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
        transforms: Optional[List[str]] = None,
    ):
        self.key_mapping = key_mapping

        try:
            super().__init__(
                pbc=pbc,
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
                transforms=transforms,
            )
        except Exception as e:
            logging.error(f"Error in file {self.raw_file_names}")
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
