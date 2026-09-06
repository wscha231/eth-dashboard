"""Only a forecast genuinely issued today enters the immutable live record."""
import numpy as np

from forecasting.daily_data import utc_timestamp
from research_pipeline.forward import connect, save_issue, settle, nonoverlap_count, block_interval
from hybrid_pipeline.protocol import PROTOCOL_HASH


def publish_issued(payload, master, path, *, now=None, issue_new=True, settlement_source_hash=None):
    import json
    current=utc_timestamp(now);conn=connect(path)
    revised=settle(conn,master,source_hash=settlement_source_hash or payload['source_hashes']['master'],now=current)
    saved=0
    for h,d in (payload['horizons'].items() if issue_new else []):
        origin=utc_timestamp(d['current']['origin'])
        if origin != current.floor('D'):
            raise ValueError('Do not backdate a historical replay as a live forecast')
        issue=dict(d['current'],horizon=int(h),candidate='optimized_hybrid',protocol_hash=PROTOCOL_HASH,
                   issued_at=current.isoformat(),runtime_hash=payload['runtime_hash'],source_hashes=payload['source_hashes'])
        saved+=save_issue(conn,issue)
        row=conn.execute('SELECT payload FROM issued WHERE protocol_hash=? AND origin=? AND horizon=? AND candidate=?',
                         (PROTOCOL_HASH,issue['origin'],int(h),'optimized_hybrid')).fetchone()
        d['current']=json.loads(row['payload'])
    rows=conn.execute('''SELECT i.payload,a.price actual_price FROM issued i LEFT JOIN actual_revisions a
        ON a.id=(SELECT MAX(id) FROM actual_revisions WHERE issued_id=i.id)
        WHERE i.protocol_hash=? ORDER BY i.origin,i.horizon''',(PROTOCOL_HASH,)).fetchall()
    issued=[dict(json.loads(row['payload']),actual_price=row['actual_price']) for row in rows]
    stats=[]
    for h in (7,30):
        matured=[r for r in issued if r['horizon']==h and r['actual_price'] is not None]
        metric={'horizon':h,'resolved':len(matured),'review_eligible':False}
        if matured:
            actual=np.array([r['actual_price']/r['reference_price']-1 for r in matured]);pred=np.array([r['predicted_return'] for r in matured])
            loss=abs(pred-actual);baseline=abs(actual)
            enough=len(matured)>=180 and nonoverlap_count([r['origin'] for r in matured],h)>=6
            ci=block_interval(baseline-loss,[r['origin'] for r in matured],h) if enough else None
            metric.update(return_mae=float(loss.mean()),mae_skill_vs_no_change=float(1-loss.mean()/baseline.mean()) if baseline.mean() else None,
                          review_eligible=bool(enough and ci[0]>0 and np.mean((pred-actual)**2)<np.mean(actual**2)),
                          mae_improvement_95ci=ci)
        stats.append(metric)
    conn.close()
    payload['prospective']={'issued':len(issued),'resolved':sum(r['actual_price'] is not None for r in issued),
                            'new_issues':saved,'actual_revisions':revised,'metrics':stats,
                            'recent_issued':issued[-60:],'note':'Retried runs cannot rewrite an issued forecast; historical replay points are never inserted as issued records.'}
    payload['generated_at']=current.isoformat()
    return payload
