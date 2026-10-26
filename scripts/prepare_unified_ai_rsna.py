"""Persistent independent RSNA source transfer using the shared native verifier."""
import runpy,sys
from pathlib import Path
if __name__=='__main__':
    sys.argv.extend(['--dataset','rsna_pneumonia'])
    runpy.run_path(str(Path(__file__).with_name('prepare_unified_ai_native.py')),run_name='__main__')
