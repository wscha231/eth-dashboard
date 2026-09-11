import numpy as np
import pandas as pd
import pytest
from signal_pipeline import optimization as opt


def test_purge_requires_full_target_before_next_partition():
    idx=pd.date_range('2020-01-01',periods=240,freq='h',tz='UTC')
    f=pd.DataFrame({'eth_ret_24':1.,'reference_price':100.},index=idx)
    y=pd.DataFrame({'return':0.,'target_end':idx+pd.Timedelta(hours=25)},index=idx)
    result=opt.calibration_indices(f,y,idx[0],idx[-1])
    assert (y.loc[result,'target_end']<idx[-1]-pd.Timedelta(hours=1)).all()
    assert len(result)>0


def test_interval_score_penalizes_misses_and_width():
    truth=np.array([0.])
    assert opt.interval_loss(np.array([[-.1,0,.1]]),truth)<opt.interval_loss(np.array([[.2,.3,.4]]),truth)


def test_calibrator_rejects_different_model():
    with pytest.raises(ValueError,match='identity mismatch'):
        opt.predict_candidate({'models':{},'fitted_identity':'different'},pd.DataFrame())


def test_selection_calibration_and_threshold_use_disjoint_mature_targets(monkeypatch):
    idx=pd.date_range('2015-01-01','2021-01-01',freq='6h',tz='UTC')
    f=pd.DataFrame({'eth_ret_24':np.sin(np.arange(len(idx))), 'reference_price':100.},index=idx)
    y=pd.DataFrame({'return':0.,'target_end':idx+pd.Timedelta(hours=7),'terminal':1,'up':0,'down':0},index=idx)
    fits=[]
    def fit(features,outcomes,indices):
        fits.append(indices.copy());return {'identity':len(fits)}
    def predict(model,features):
        n=len(features);return {'climatology':{'terminal':np.tile([.1,.8,.1],(n,1)),'path':np.tile([.2,.2],(n,1)),'quantiles':np.tile([-.1,.01,.1],(n,1))}}
    monkeypatch.setattr(opt,'fit_models',fit);monkeypatch.setattr(opt,'predict_models',predict)
    cutoff=pd.Timestamp('2020-12-01',tz='UTC')
    bundle=opt.train_candidate(f,y,cutoff,6)
    assert bundle['choices']['point']=='no_change'
    assert pd.Timestamp(bundle['training_target_end'])<cutoff-pd.Timedelta(days=90,hours=1)
    assert pd.Timestamp(bundle['calibration_target_end'])<cutoff-pd.Timedelta(days=45,hours=1)
    assert pd.Timestamp(bundle['threshold_target_end'])<cutoff-pd.Timedelta(hours=1)
    assert bundle['alert_thresholds']['up']>=0
    original=opt.predict_candidate(bundle,f.iloc[:1])
    changed=y.copy();changed.loc[changed.index>=cutoff,'return']=10.
    fits.clear();again=opt.train_candidate(f,changed,cutoff,6)
    assert bundle['model_version']==again['model_version']
    assert np.array_equal(original['quantiles'],opt.predict_candidate(again,f.iloc[:1])['quantiles'])


def test_real_fitted_model_identity_survives_checkpoint_reload(tmp_path):
    import joblib
    from signal_pipeline.models import fit_models,predict_models
    index=pd.date_range('2020-01-01',periods=240,freq='6h',tz='UTC')
    rng=np.random.default_rng(73)
    features=pd.DataFrame({'eth_ret_24':rng.normal(size=len(index)),'reference_price':100.},index=index)
    outcomes=pd.DataFrame({'return':rng.normal(0,.03,len(index)),'terminal':np.arange(len(index))%3,
                           'up':np.arange(len(index))%2,'down':(np.arange(len(index))//2)%2},index=index)
    models={365:fit_models(features,outcomes,index)}
    identity=opt.model_identity(models)
    file=tmp_path/'candidate.joblib';joblib.dump(models,file);restored=joblib.load(file)
    assert opt.model_identity(restored)==identity
    before=predict_models(models[365],features.iloc[:4]);after=predict_models(restored[365],features.iloc[:4])
    for model in before:
        for output in before[model]:np.testing.assert_array_equal(before[model][output],after[model][output])
    restored[365]['baseline']['terminal'][0]+=.01
    assert opt.model_identity(restored)!=identity
