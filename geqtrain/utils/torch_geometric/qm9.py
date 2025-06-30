from collections.abc import Sequence
from functools import partial
import os
import os.path as osp
import ssl
import sys
from typing import Any, List, Optional
import zipfile
import numpy as np

import fsspec
import torch
from torch import Tensor
# from torch_scatter import scatter
from geqtrain.utils.pytorch_scatter import scatter_sum
from tqdm import tqdm
import urllib

HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])

atomrefs = {
    6: [0., 0., 0., 0., 0.],
    7: [
        -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
        -2713.48485589
    ],
    8: [
        -13.5745904, -1029.82456413, -1485.26398105, -2042.5727046,
        -2713.44632457
    ],
    9: [
        -13.54887564, -1029.79887659, -1485.2382935, -2042.54701705,
        -2713.42063702
    ],
    10: [
        -13.90303183, -1030.25891228, -1485.71166277, -2043.01812778,
        -2713.88796536
    ],
    11: [0., 0., 0., 0., 0.],
}


def download_url(
    url: str,
    folder: str,
    log: bool = True,
    filename: Optional[str] = None,
):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (str): The URL.
        folder (str): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
        filename (str, optional): The filename of the downloaded file. If set
            to :obj:`None`, will correspond to the filename given by the URL.
            (default: :obj:`None`)
    """
    if filename is None:
        filename = url.rpartition('/')[2]
        filename = filename if filename[0] == '?' else filename.split('?')[0]

    path = osp.join(folder, filename)

    if osp.isfile(path):  # pragma: no cover
        if log:
            print(f'Using existing file {filename}', file=sys.stderr)
        return path

    if log:
        print(f'Downloading {url}', file=sys.stderr)

    os.makedirs(folder, exist_ok=True)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with fsspec.open(path, 'wb') as f:
        # workaround for https://bugs.python.org/issue42853
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path

def extract_zip(path: str, folder: str, log: bool = True) -> None:
    r"""Extracts a zip archive to a specific folder.

    Args:
        path (str): The path to the tar archive.
        folder (str): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(folder)

def one_hot(indices, num_classes):
    """One-hot encoding for a tensor of indices."""
    return torch.eye(num_classes)[indices]

def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]

def furthest_point_sampling(data, num_samples):
    """
    Furthest Point Sampling (FPS) on a set of points.
    Args:
        data (np.ndarray): Array of shape (N, D)
        num_samples (int): Number of samples to select
    Returns:
        indices (list): Indices of selected samples
    """
    N = data.shape[0]
    indices = [0]
    distances = np.full(N, np.inf)
    for _ in range(1, num_samples):
        last = data[indices[-1]]
        dist = np.linalg.norm(data - last, axis=-1)
        distances = np.minimum(distances, dist)
        if np.all(distances == 0):
            break
        next_idx = np.argmax(distances)
        indices.append(next_idx.item())
    num_sampled = len(indices)
    if num_sampled < num_samples:
        remaining = list(set(range(N)) - set(indices))
        num_needed = num_samples - num_sampled
        print(f"Sampled all unique values after {num_sampled} samples. Sampling the remaining {num_needed} indices using random uniform sampling.")
        if num_needed > 0 and remaining:
            extra = np.random.choice(remaining, size=min(num_needed, len(remaining)), replace=False)
            indices.extend(extra.tolist())
    return indices

def load_and_flatten(fname, key):
    arr = np.load(fname, allow_pickle=True)[key]
    return arr.flatten()

def split_npz_by_fps(folder, key, num_train_samples=None, num_valid_samples=None):
    print(f"Scanning folder: {folder}")
    # Gather all .npz files and their key values
    npz_files = [os.path.join(folder, f) for f in os.listdir(folder)]
    print(f"Found {len(npz_files)} files.")
    values = []
    print(f"Reading files.")

    import multiprocessing as mp
    func = partial(load_and_flatten, key=key)
    with mp.Pool(mp.cpu_count()) as pool:
        values = pool.map(func, npz_files)
    values = np.stack(values)
    print(f"Loaded key '{key}' from all files. Shape: {values.shape}")
    
    N = len(npz_files)
    npz_files = np.array(npz_files)
    print("Performing furthest point sampling for train split...")
    idx_train = furthest_point_sampling(values, num_train_samples or int(0.8 * N))
    filenames_train = npz_files[idx_train]
    print(f"Selected {len(filenames_train)} samples for training.")
    remaining = list(set(range(N)) - set(idx_train))
    print("Performing furthest point sampling for validation split...")
    idx_val = furthest_point_sampling(values[remaining], num_valid_samples or len(remaining) // 2)
    idx_val = [remaining[i] for i in idx_val]
    filenames_val = npz_files[idx_val]
    print(f"Selected {len(filenames_val)} samples for validation.")
    idx_test = list(set(range(N)) - set(idx_train) - set(idx_val))
    filenames_test = npz_files[idx_test]
    print(f"Selected {len(filenames_test)} samples for test.")
    
    # Move files
    for filenames, sub in zip([filenames_train, filenames_val, filenames_test], ['train', 'val', 'test']):
        # Write indices to txt file
        split_file = os.path.join(os.path.dirname(folder), f"{key}.{sub}.txt")
        np.savetxt(split_file, filenames, fmt='%s')
        print(f"Saved {len(filenames)} dataset filenames to {split_file}")

class QM9:
    r"""The QM9 dataset from the `"MoleculeNet: A Benchmark for Molecular
    Machine Learning" <https://arxiv.org/abs/1703.00564>`_ paper, consisting of
    about 130,000 molecules with 19 regression targets.
    Each molecule includes complete spatial information for the single low
    energy conformation of the atoms in the molecule.
    In addition, we provide the atom features from the `"Neural Message
    Passing for Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper.

    Args:
        - dataset_folder: str

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - #graphs
          - #nodes
          - #edges
          - #features
          - #tasks
        * - 130,831
          - ~18.0
          - ~37.3
          - 11
          - 19
    """  # noqa: E501

    raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
               'molnet_publish/qm9.zip')
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    processed_url = 'https://data.pyg.org/datasets/qm9_v3.zip'

    def __init__(
        self,
        dataset_folder: str
    ) -> None:
        self.raw_dir = dataset_folder
        self.processed_dir = os.path.join(self.raw_dir, "processed")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def mean(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())

    def std(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())

    def atomref(self, target: int) -> Optional[Tensor]:
        if target in atomrefs:
            out = torch.zeros(100)
            out[torch.tensor([1, 6, 7, 8, 9])] = torch.tensor(atomrefs[target])
            return out.view(-1, 1)
        return None

    @property
    def raw_file_names(self) -> List[str]:
        try:
            import rdkit  # noqa
            return ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']
        except ImportError:
            return ['qm9_v3.pt']
    
    @property
    def raw_paths(self) -> List[str]:
        r"""The filepaths to find in order to skip the download."""
        files = to_list(self.raw_file_names)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self) -> str:
        return 'data_v3.pt'

    def download(self) -> None:
        try:
            import rdkit  # noqa
            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)

            file_path = download_url(self.raw_url2, self.raw_dir)
            os.rename(osp.join(self.raw_dir, '3195404'),
                      osp.join(self.raw_dir, 'uncharacterized.txt'))
        except ImportError:
            path = download_url(self.processed_url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)

    def process(self) -> None:
        from rdkit import Chem, RDLogger
        from rdkit.Chem.rdchem import BondType as BT
        from rdkit.Chem.rdchem import HybridizationType
        RDLogger.DisableLog('rdApp.*')  # type: ignore

        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
        hybridizations = {HybridizationType.UNSPECIFIED: 0, HybridizationType.SP: 1, HybridizationType.SP2: 2, HybridizationType.SP3: 3}

        with open(self.raw_paths[1]) as f:
            target = [[float(x) for x in line.split(',')[1:20]]
                      for line in f.read().split('\n')[1:-1]]
            y = torch.tensor(target, dtype=torch.float)
            y = torch.cat([y[:, 3:], y[:, :3]], dim=-1)
            y = y * conversion.view(1, -1)

        with open(self.raw_paths[2]) as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False,
                                   sanitize=False)

        for i, mol in enumerate(tqdm(suppl)):
            if i in skip:
                continue

            N = mol.GetNumAtoms()

            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float).unsqueeze(dim=0)

            atomic_number = []
            atom_type = []
            aromatic = []
            hybridization = []
            for atom in mol.GetAtoms():
                atomic_number.append(atom.GetAtomicNum())
                atom_type.append(types[atom.GetSymbol()])
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization.append(hybridizations[atom.GetHybridization()])

            atomic_number = torch.tensor(atomic_number, dtype=torch.long)
            atom_type = torch.tensor(atom_type)
            aromatic = torch.tensor(aromatic)
            hybridization = torch.tensor(hybridization)

            # Create bond edges and types
            rows, cols, edge_types = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                rows += [start, end]
                cols += [end, start]
                edge_types += 2 * [bonds[bond.GetBondType()] + 1] # Add 1 to keep 0 for non-bonded

            edge_bond = torch.tensor([rows, cols], dtype=torch.long)
            edge_bond_type = torch.tensor(edge_types, dtype=torch.long)

            row, col = edge_bond
            hs = (atomic_number == 1).to(torch.float)
            num_hs = scatter_sum(hs[row], col, dim_size=N, reduce='sum').int()

            # Create all2all edge_index (including self-loops)
            N = mol.GetNumAtoms()
            edge_index = torch.cartesian_prod(torch.arange(N), torch.arange(N)).t()
            edge_index = edge_index[:, edge_index[0] != edge_index[1]].unsqueeze(0)

            # Assign bond type to all2all edges (0 for non-bonded)
            bond_dict = {(u.item(), v.item()): t.item() for u, v, t in zip(edge_bond[0], edge_bond[1], edge_bond_type)}
            edge_type = []
            for u, v in edge_index[0].t():
                edge_type.append(bond_dict.get((u.item(), v.item()), 0))
            edge_type = torch.tensor(edge_type, dtype=torch.long).unsqueeze(0)

            name = mol.GetProp('_Name')
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

            _y = y[i].unsqueeze(0)
            mu = _y[:,  :1]
            alpha = _y[:, 1:2]
            homo = _y[:, 2:3]
            lumo = _y[:, 3:4]
            gap = _y[:, 4:5]
            r2 = _y[:, 5:6]
            zpve = _y[:, 6:7]
            u0 = _y[:, 7:8]
            u298 = _y[:, 8:9]
            h298 = _y[:, 9:10]
            g298 = _y[:, 10:11]
            cv = _y[:, 11:12]
            u0_atom = _y[:, 12:13]
            u298_atom = _y[:, 13:14]
            h298_atom = _y[:, 14:15]
            g298_atom = _y[:, 15:16]
            A = _y[:, 16:17]
            B = _y[:, 17:18]
            C = _y[:, 18:19]

            # Save data as .npz file
            np.savez_compressed(
                osp.join(self.processed_dir, f"data_{i}.npz"),
                pos=pos.numpy(),
                atom_type=atom_type.numpy(),
                edge_index=edge_index.numpy(),
                edge_type=edge_type.numpy(),
                atomic_number=atomic_number.numpy(),
                aromatic=aromatic.numpy(),
                hybridization=hybridization.numpy(),
                num_hs=num_hs.numpy(),
                mu=mu.numpy(),
                alpha=alpha.numpy(),
                homo=homo.numpy(),
                lumo=lumo.numpy(),
                gap=gap.numpy(),
                r2=r2.numpy(),
                zpve=zpve.numpy(),
                u0=u0.numpy(),
                u298=u298.numpy(),
                h298=h298.numpy(),
                g298=g298.numpy(),
                cv=cv.numpy(),
                u0_atom=u0_atom.numpy(),
                u298_atom=u298_atom.numpy(),
                h298_atom=h298_atom.numpy(),
                g298_atom=g298_atom.numpy(),
                A=A.numpy(),
                B=B.numpy(),
                C=C.numpy(),
                smiles=smiles,
                name=name,
                idx=i,
            )
    
    def split(self, split_on_key: str, num_train_samples=None, num_valid_samples=None):
        split_npz_by_fps(self.processed_dir, split_on_key, num_train_samples=num_train_samples, num_valid_samples=num_valid_samples)