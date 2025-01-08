from typing import Callable, List, Union
from .data import Data
import copy
from abc import ABC
from typing import Any


class BaseTransform(ABC):
    r"""

    Taken from: https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/transforms/base_transform.py#L6
    Removed the HeteroData dependecy

    An abstract base class for writing transforms.

    Transforms are a general way to modify and customize
    :class:`~torch_geometric.data.Data` or
    passing them as an argument to a :class:`~torch_geometric.data.Dataset`, or
    by applying them explicitly to individual
    :class:`~torch_geometric.data.Data`

    .. code-block:: python

        import torch_geometric.transforms as T
        from torch_geometric.datasets import TUDataset

        transform = T.Compose([T.ToUndirected(), T.AddSelfLoops()])

        dataset = TUDataset(path, name='MUTAG', transform=transform)
        data = dataset[0]  # Implicitly transform data on every access.

        data = TUDataset(path, name='MUTAG')[0]
        data = transform(data)  # Explicitly transform data.
    """
    def __call__(self, data: Any) -> Any:
        # Shallow-copy the data so that we prevent in-place data modification.
        return self.forward(copy.copy(data))

    def forward(self, data: Any) -> Any:
        pass

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class Compose(BaseTransform):
    r"""

    Taken as-is from: https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/transforms/compose.py

    Composes several transforms together.

    Args:
        transforms (List[Callable]): List of transforms to compose.
    """
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def forward(
        self,
        data: Union[Data],
    ) -> Union[Data]:
        for transform in self.transforms:
            if isinstance(data, (list, tuple)):
                data = [transform(d) for d in data]
            else:
                data = transform(data)
        return data

    def __repr__(self) -> str:
        args = [f'  {transform}' for transform in self.transforms]
        return '{}([\n{}\n])'.format(self.__class__.__name__, ',\n'.join(args))