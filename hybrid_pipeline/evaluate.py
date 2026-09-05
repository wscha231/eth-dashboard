"""Past-only ensemble optimization and all-origin error / interval evaluation."""
import itertools

import numpy as np
import pandas as pd

from forecasting.daily_data import utc_timestamp
from hybrid_pipeline.data import targets, runtime_hash
from hybrid_pipeline.protocol import BASE_MODELS, PROTOCOL, PROTOCOL_HASH
from research_pipeline.forward import block_interval

MODEL_NAMES={
    'no_change':'No-change reference','cat_short':'CatBoost · short window','cat_long':'CatBoost · long window',
    'transformer_short':'Transformer · 32-day sequence','transformer_long':'Transformer · 64-day sequence',
    'equal_hybrid':'CatBoost + Transformer · equal','optimized_hybrid':'CatBoost + Transformer · past-only optimized',
    'safe_policy':'Past-only model / reference selection',
}


def simple_return(z,sigma):
    # Fixed numerical bound, disclosed through clipping counts in the report.
    return np.expm1(np.clip(np.asarray(z)*np.asarray(sigma),-3,3))


def blend(values, choice):
    return choice['amplitude']*(choice['cat_weight']*np.asarray(values[choice['cat']])
                                +(1-choice['cat_weight'])*np.asarray(values[choice['transformer']]))


def optimize(available):
    default={'cat':'cat_short','transformer':'transformer_short','cat_weight':.5,'amplitude':1.0}
    if len(available) < PROTOCOL['selection_min_rows']:
        return default,False,None
    trials=[]
    for cat,trans,w,a in itertools.product(BASE_MODELS[:2],BASE_MODELS[2:],PROTOCOL['blend_weights_cat'],PROTOCOL['amplitudes']):
        choice={'cat':cat,'transformer':trans,'cat_weight':w,'amplitude':a}
        loss=float(np.mean(abs(simple_return(blend(available,choice),available.sigma)-available.actual_return)))
        trials.append((loss,choice))
    # The grid order deterministically breaks exact ties; no full-test score.
    loss,choice=min(trials,key=lambda t:t[0])
    baseline=float(abs(available.actual_return).mean())
    return choice,bool(loss < baseline),{'hybrid_mae':loss,'reference_mae':baseline}


def make_paths(base,raw):
    records=[]; decisions=[]
    for horizon,data in base.groupby('horizon'):
        data=data.sort_values('origin').copy();y,sigma=targets(raw,int(horizon))
        cfg=PROTOCOL['event_threshold'][str(horizon)]
        for period,current in data.groupby(data.origin.dt.to_period('M')):
            cutoff=period.start_time
            available=data[(data.target < cutoff)&(data.target >= cutoff-pd.Timedelta(days=PROTOCOL['selection_days'][str(horizon)]))&data.actual_return.notna()]
            choice,use_model,losses=optimize(available)
            if len(available) >= 60:
                truth_z=np.log1p(available.actual_return.to_numpy())/available.sigma.to_numpy()
                residual=truth_z-blend(available,choice)
                reference_residual=truth_z
                calibration='past_out_of_sample_residuals'
            else:
                # Warmup uses only labels already matured before this fit.
                valid=(y.index+pd.Timedelta(days=int(horizon)) < cutoff-pd.Timedelta(days=PROTOCOL['embargo_days'][str(horizon)]))&y.notna()
                residual=y.loc[valid].tail(730).to_numpy();reference_residual=residual
                calibration='matured_training_distribution_warmup'
            if not len(residual) or not np.isfinite(residual).all():
                raise ValueError('No finite past-only uncertainty calibration values')
            decision={'horizon':int(horizon),'month':cutoff.date().isoformat(),'choice':choice,
                      'selection_rows':len(available),'selection_latest_target':available.target.max().date().isoformat() if len(available) else None,
                      'safe_uses_model':use_model,'validation':losses,'calibration':calibration,'calibration_rows':len(residual)}
            decisions.append(decision)
            for row in current.itertuples():
                values={name:float(getattr(row,name)) for name in BASE_MODELS}
                chosen=float(blend(values,choice)); zvalues={**values,'no_change':0.,
                     'equal_hybrid':.5*(values['cat_short']+values['transformer_short']),
                     'optimized_hybrid':chosen,'safe_policy':chosen if use_model else 0.}
                threshold=float(np.clip(row.sigma*cfg['multiplier'],cfg['floor'],cfg['cap']))
                for name,z in zvalues.items():
                    item={'origin':row.origin,'target':row.target,'horizon':int(horizon),'model':name,
                          'reference_price':row.reference_price,'actual_price':row.actual_price,'actual_return':row.actual_return,
                          'predicted_return':float(simple_return(z,row.sigma)),'event_threshold':threshold,
                          'log_prediction_clipped':bool(abs(z*row.sigma)>3)}
                    if name in ('optimized_hybrid','safe_policy','no_change'):
                        errors=reference_residual if name=='no_change' or (name=='safe_policy' and not use_model) else residual
                        draws=simple_return(z+errors,row.sigma)
                        classes=np.where(draws >= threshold,2,np.where(draws <= -threshold,0,1))
                        probabilities=(np.bincount(classes,minlength=3)+1)/(len(draws)+3)
                        item.update(lower_return=float(np.quantile(draws,.1)),upper_return=float(np.quantile(draws,.9)),
                                    p_down=float(probabilities[0]),p_flat=float(probabilities[1]),p_up=float(probabilities[2]))
                    records.append(item)
    return pd.DataFrame(records),decisions


def metrics(data, *, confidence=False):
    a=data.actual_return.to_numpy();p=data.predicted_return.to_numpy();loss=abs(p-a);base=abs(a)
    actual=np.where(a >= data.event_threshold,2,np.where(a <= -data.event_threshold,0,1))
    predicted=np.where(p >= data.event_threshold,2,np.where(p <= -data.event_threshold,0,1))
    scores={'rows':len(data),'return_mae':float(loss.mean()),'return_rmse':float(np.sqrt(np.mean((p-a)**2))),
            'price_rmse':float(np.sqrt(np.mean(((p-a)*data.reference_price)**2))),
            'mae_skill_vs_no_change':float(1-loss.mean()/base.mean()) if base.mean() else None,
            'state_accuracy':float(np.mean(actual==predicted)),
            'up_recall':float(np.mean(predicted[actual==2]==2)) if (actual==2).any() else None,
            'down_recall':float(np.mean(predicted[actual==0]==0)) if (actual==0).any() else None,
            'up_precision':float(np.mean(actual[predicted==2]==2)) if (predicted==2).any() else None,
            'up_false_positive_rate':float(np.mean(predicted[actual!=2]==2)) if (actual!=2).any() else None,
            'clipped_predictions':int(data.log_prediction_clipped.sum()),
            'mae_improvement_95ci':block_interval(base-loss,data.origin,int(data.horizon.iloc[0])) if confidence else None}
    if data.p_up.notna().all():
        probability=data[['p_down','p_flat','p_up']].to_numpy()
        scores.update(brier=float(np.mean(np.sum((probability-np.eye(3)[actual])**2,axis=1))),
                      coverage80=float(np.mean((a>=data.lower_return)&(a<=data.upper_return))),
                      mean_interval_width=float(np.mean(data.upper_return-data.lower_return)))
    return scores


def report(base,raw,runtimes,source_hashes,*,now=None,legacy=None):
    if base.duplicated(['horizon','origin']).any() or not np.isfinite(base[BASE_MODELS].to_numpy()).all():
        raise ValueError('Invalid or duplicate candidate cohorts')
    if set(base.horizon)!={7,30}:
        raise ValueError('Both horizons are required for publication')
    paths,decisions=make_paths(base,raw)
    resolved=paths[paths.actual_return.notna()].copy(); horizons={}
    for h,data in resolved.groupby('horizon'):
        leaderboard=[{'model':name,**metrics(group,confidence=name in ('optimized_hybrid','safe_policy'))} for name,group in data.groupby('model')]
        leaderboard.sort(key=lambda r:r['return_mae'])
        recent=[{'model':name,**metrics(group)} for name,group in data[data.target>data.target.max()-pd.Timedelta(days=365)].groupby('model')]
        yearly=[{'year':int(year),'model':name,**metrics(group)} for (year,name),group in data.groupby([data.target.dt.year,'model'])]
        first=base[base.horizon.eq(h)].origin.min()
        expected=pd.date_range(first,raw.index[-1]-pd.Timedelta(days=int(h)))
        origins=pd.DatetimeIndex(data[data.model.eq('optimized_hybrid')].origin)
        if len(expected.difference(origins)):
            raise ValueError('Missing eligible calendar origins; refuse an incomplete full-history chart')
        points=[]
        for origin,rows in data.groupby('origin'):
            indexed=rows.set_index('model');one=indexed.loc['optimized_hybrid']
            points.append({'origin':origin.date().isoformat(),'target':one.target.date().isoformat(),
                           'reference_price':float(one.reference_price),'actual_price':float(one.actual_price),'actual_return':float(one.actual_return),
                           'returns':rows.set_index('model').predicted_return.to_dict(),
                           'lower_return':float(one.lower_return),'upper_return':float(one.upper_return),
                           'probability_down_flat_up':[float(one.p_down),float(one.p_flat),float(one.p_up)]})
        current=paths[(paths.horizon==h)&(paths.origin==raw.index[-1])].set_index('model')
        one=current.loc['optimized_hybrid'];last_choice=next(d for d in reversed(decisions) if d['horizon']==h)
        current_payload={'origin':one.origin.date().isoformat(),'target':one.target.date().isoformat(),
                         'reference_price':float(one.reference_price),'predicted_return':float(one.predicted_return),
                         'predicted_price':float(one.reference_price*(1+one.predicted_return)),
                         'lower_price':float(one.reference_price*(1+one.lower_return)),
                         'upper_price':float(one.reference_price*(1+one.upper_return)),
                         'probability_down_flat_up':[float(one.p_down),float(one.p_flat),float(one.p_up)],
                         'choice':last_choice,'status':'prospective_research','verified_predictive_edge':False}
        learned=[r for r in leaderboard if r['model'] not in ('no_change','safe_policy')]
        horizons[str(h)]={'matched_origins':len(points),'first_origin':points[0]['origin'],'last_target':points[-1]['target'],
                          'best_fixed_retrospective':min(learned,key=lambda r:r['return_mae'])['model'],
                          'default_chart':'optimized_hybrid','leaderboard':leaderboard,'recent':recent,'yearly':yearly,
                          'points':points,'current':current_payload}
    legacy_comparison=[]
    if legacy is not None:
        for h in (7,30):
            new=resolved[(resolved.horizon==h)&resolved.model.eq('optimized_hybrid')]
            old=legacy[legacy.horizon.eq(h)]
            for name,group in old.groupby('model'):
                paired=group.merge(new,on=['origin','target'],suffixes=('_old','_new'))
                paired=paired[np.isclose(paired.reference_price_old,paired.reference_price_new)&np.isclose(paired.actual_price_old,paired.actual_price_new)]
                if len(paired):
                    old_mae=float(abs(paired.predicted_return_old-paired.actual_return_old).mean())
                    new_mae=float(abs(paired.predicted_return_new-paired.actual_return_new).mean())
                    legacy_comparison.append({'horizon':h,'retired_candidate':name,'matched_origins':len(paired),
                                              'old_mae':old_mae,'new_hybrid_mae':new_mae,'new_skill':1-new_mae/old_mae if old_mae else None})
    return {'schema_version':1,'generated_at':utc_timestamp(now).isoformat(),'status':'historical_optimization_and_prospective_research',
            'protocol':PROTOCOL,'protocol_hash':PROTOCOL_HASH,'runtime_hash':runtime_hash(),'source_hashes':source_hashes,
            'model_names':MODEL_NAMES,'runtimes':runtimes,'horizons':horizons,'monthly_decisions':decisions,
            'legacy_matched_comparison':legacy_comparison},paths
