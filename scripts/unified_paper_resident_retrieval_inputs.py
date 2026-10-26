"""Open resident copies while retaining the canonical cross-model sample IDs."""
import os
from pathlib import Path
CANONICAL='/mnt/huawei_deepcad/benchmark'
def actual_path(path):
 text=str(path);base=os.environ.get('UNIFIED_PAPER_BENCHMARK_ROOT',CANONICAL)
 return Path(base+text[len(CANONICAL):])if text.startswith(CANONICAL+'/')else Path(path)
def canonical_path(path):
 text=str(path);base=os.environ.get('UNIFIED_PAPER_BENCHMARK_ROOT',CANONICAL)
 return CANONICAL+text[len(base):]if text.startswith(base+'/')else text
class CanonicalImageIds:
 def __init__(self,dataset):self.dataset=dataset
 def __len__(self):return len(self.dataset)
 def __getitem__(self,index):
  image,label,ident=self.dataset[index]
  return image,label,canonical_path(ident)
