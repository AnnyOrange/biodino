"""
Dataset registry for bio_segmentation.

To add a new dataset:
1. Create a module in this package (e.g. my_dataset.py).
2. Define a Dataset class and a get_paths() function.
3. Register them in DATASET_REGISTRY below.
"""

from .cellpose import CellposeDataset, get_cellpose_paths
from .csc import CSCDataset, get_csc_paths

# Maps dataset name -> (DatasetClass, get_paths_fn)
DATASET_REGISTRY = {
    'cellpose': (CellposeDataset, get_cellpose_paths),
    'csc':      (CSCDataset,      get_csc_paths),
}

__all__ = [
    'DATASET_REGISTRY',
    'CellposeDataset', 'get_cellpose_paths',
    'CSCDataset', 'get_csc_paths',
]
