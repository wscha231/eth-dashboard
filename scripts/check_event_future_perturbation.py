import argparse,json,time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from signal_pipeline.data import read_bars,build_features,utc
from signal_pipeline.models import labels
from signal_pipeline.engine import source_hash
from signal_pipeline.optimization import train_candidate,predict_candidate
parser=argparse.ArgumentParser(description='Change only future prices and compare independently refitted past predictions.')
parser.add_argument('--root',default='lake/signals');parser.add_argument('--cutoff',default='2024-01-01');parser.add_argument('--output')
args=parser.parse_args();root=args.root
bars=read_bars(root);cutoff=utc(args.cutoff);features=build_features(bars)
altered=bars.copy();future=altered.close_time>cutoff
altered.loc[future,['open','high','low','close']]*=5
altered.loc[future,'volume']*=3
altered.loc[future,'content_hash']='future-perturbation'
changed=build_features(altered)
pd.testing.assert_frame_equal(features.loc[:cutoff],changed.loc[:cutoff])
assert source_hash(bars,cutoff)==source_hash(altered,cutoff)
assert source_hash(bars,utc('2025-01-01'))!=source_hash(altered,utc('2025-01-01'))
results={}
for h in (6,720):
 started=time.monotonic(); y=labels(bars,features,h);dy=labels(altered,changed,h)
 a=train_candidate(features,y,cutoff,h);b=train_candidate(changed,dy,cutoff,h)
 print('identities',h,a['fitted_identity'],b['fitted_identity'],a['choices'],b['choices'],flush=True)
 assert a['choices']==b['choices']
 pd.testing.assert_frame_equal(y.loc[y.target_end<cutoff],dy.loc[dy.target_end<cutoff])
 pa=predict_candidate(a,features.loc[[cutoff]]);pb=predict_candidate(b,changed.loc[[cutoff]])
 for output in pa:
  print(output,float(np.abs(pa[output]-pb[output]).max()),flush=True)
  np.testing.assert_array_equal(pa[output],pb[output])
 results[str(h)]={'head_choices_identical':True,'checkpoint_ids_identical':a['model_version']==b['model_version'],'all_forecast_outputs_identical':True,'seconds':time.monotonic()-started,'model_version':a['model_version']}
 print(h,'future perturbation passed',flush=True)
result={'status':'passed','source_snapshot':source_hash(bars,bars.close_time.max()+pd.Timedelta(hours=1)),'cutoff':cutoff.isoformat(),'future_price_multiplier':5,'future_volume_multiplier':3,'past_features_identical':True,'past_only_source_key_identical':True,'future_source_key_changed':True,'horizons':results}
result['note']='Separately refitted checkpoint IDs differ; all numeric forecast outputs and earlier head selections were compared exactly. This audit does not assert byte-identical independent refits.'
if args.output:Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
