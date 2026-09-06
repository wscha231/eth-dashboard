"""Continue settling old issued forecasts without creating another old-model forecast."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from hybrid_pipeline.ledger import publish_issued

root=Path('lake/hybrid')
payload=json.loads((root/'hybrid_forecast.json').read_text())
master_path=Path('lake/gold/eth_master_daily.csv')
master=pd.read_csv(master_path,index_col=0,parse_dates=True)
payload=publish_issued(payload,master,root/'issued.db',issue_new=False,
                       settlement_source_hash=hashlib.sha256(master_path.read_bytes()).hexdigest())
payload['retired_from_issuance']=True
payload['retirement_reason']='Superseded by separately evaluated hourly event forecasts; old issued outcomes remain auditable.'
(root/'hybrid_forecast.json').write_text(json.dumps(payload,indent=2,allow_nan=False))
print('Archived hybrid settlement:',payload['prospective']['resolved'],'resolved; no new issues')
