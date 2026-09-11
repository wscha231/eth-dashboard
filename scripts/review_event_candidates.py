#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from signal_pipeline.optimization import run_review
p=argparse.ArgumentParser();p.add_argument('--root',default='lake/signals');p.add_argument('--budget-seconds',type=int,default=1500)
a=p.parse_args();result=run_review(a.root,budget_seconds=a.budget_seconds)
print(__import__('json').dumps({k:v for k,v in result.items() if k!='horizons'}))
