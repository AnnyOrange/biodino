"""
Dataset registry for bio_segmentation.

To add a new dataset:
1. Create a module in this package (e.g. my_dataset.py).
2. Define a Dataset class and a get_paths() function.
3. Register them in DATASET_REGISTRY below.

Registry format
---------------
For file-based datasets (img_paths + mask_paths):
    'name': (DatasetClass, get_paths_fn, 'file')

For array-based datasets (numpy arrays / special loaders):
    'name': (DatasetClass, get_paths_fn, 'array')

For COCO-JSON-based datasets:
    'name': (DatasetClass, get_paths_fn, 'coco')
"""

from .cellpose import CellposeDataset, get_cellpose_paths
from .csc      import CSCDataset,      get_csc_paths
from .bbbc038  import BBBC038Dataset,  get_bbbc038_paths
from .conic    import CoNICDataset,    get_conic_paths
from .livecell import LIVECellDataset, get_livecell_paths
from .monuseg  import MoNuSegDataset,  get_monuseg_paths
from .pannuke  import PanNukeDataset,  get_pannuke_paths
from .tissuenet import TissueNetDataset, get_tissuenet_paths

# Maps dataset name -> (DatasetClass, get_paths_fn, loader_type)
DATASET_REGISTRY = {
    # ---- existing ----
    'cellpose': (CellposeDataset, get_cellpose_paths, 'file'),
    'csc':      (CSCDataset,      get_csc_paths,      'file'),
    # ---- new bio benchmarks ----
    'bbbc038':  (BBBC038Dataset,  get_bbbc038_paths,  'file'),
    'conic':    (CoNICDataset,    get_conic_paths,    'array'),
    'livecell': (LIVECellDataset, get_livecell_paths, 'coco'),
    'monuseg':  (MoNuSegDataset,  get_monuseg_paths,  'file'),
    'pannuke':  (PanNukeDataset,  get_pannuke_paths,  'array'),
    'tissuenet':(TissueNetDataset,get_tissuenet_paths,'array'),
}

__all__ = [
    'DATASET_REGISTRY',
    # existing
    'CellposeDataset', 'get_cellpose_paths',
    'CSCDataset',      'get_csc_paths',
    # new
    'BBBC038Dataset',    'get_bbbc038_paths',
    'CoNICDataset',      'get_conic_paths',
    'LIVECellDataset',   'get_livecell_paths',
    'MoNuSegDataset',    'get_monuseg_paths',
    'PanNukeDataset',    'get_pannuke_paths',
    'TissueNetDataset',  'get_tissuenet_paths',
]
