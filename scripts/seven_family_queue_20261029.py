"""Dedicated queue entrypoint: never revive historical cpu-cluster queues."""
import sys
from run_unified_paper_seven_family_20261029 import main
if __name__=='__main__':
    sys.argv.insert(1,'--worker')
    main()
