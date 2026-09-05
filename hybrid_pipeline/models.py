"""Real CPU CatBoost and patch attention models with purged early stopping."""
from copy import deepcopy
import time

import numpy as np
import torch
from torch import nn
from catboost import CatBoostRegressor

from hybrid_pipeline.data import PastScaler, inner_split
from hybrid_pipeline.protocol import PROTOCOL


class PatchTransformer(nn.Module):
    def __init__(self, features, config):
        super().__init__()
        self.config = config
        tokens = config['lookback']//config['patch']
        self.projection = nn.Linear(features*config['patch'],config['width'])
        self.position = nn.Parameter(torch.zeros(1,tokens,config['width']))
        layer = nn.TransformerEncoderLayer(config['width'],config['heads'],dim_feedforward=config['width']*2,
                                           dropout=PROTOCOL['neural_training']['dropout'],batch_first=True,activation='gelu')
        self.encoder = nn.TransformerEncoder(layer,config['layers'],enable_nested_tensor=False)
        self.head = nn.Linear(config['width'],1)
        self.skip = nn.Linear(features,1,bias=False)
        nn.init.normal_(self.position,std=.02)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias); nn.init.zeros_(self.skip.weight)

    def forward(self, x):
        patch = self.config['patch']
        tokens = x.reshape(x.shape[0],x.shape[1]//patch,-1)
        encoded = self.encoder(self.projection(tokens)+self.position)
        # The entire input window ends at its origin; no future target tokens.
        return (self.head(encoded.mean(dim=1))+self.skip(x[:,-1])).squeeze(-1)


def sequences(values, positions, length):
    indices = np.asarray(positions)[:,None]-np.arange(length-1,-1,-1)[None,:]
    if indices.min() < 0:
        raise ValueError('A sequence crossed the start of recorded history')
    return torch.from_numpy(np.asarray(values[indices],dtype=np.float32))


def _neural_fit(values, y, positions, config, *, validation=None, epochs=None):
    torch.manual_seed(PROTOCOL['seed'])
    torch.set_num_threads(2)
    model = PatchTransformer(values.shape[1],config)
    optimizer = torch.optim.AdamW(model.parameters(),lr=PROTOCOL['neural_training']['learning_rate'],
                                   weight_decay=PROTOCOL['neural_training']['weight_decay'])
    train_x = sequences(values,positions,config['lookback'])
    train_y = torch.tensor(np.asarray(y)[positions],dtype=torch.float32)
    val_x = sequences(values,validation,config['lookback']) if validation is not None else None
    val_y = torch.tensor(np.asarray(y)[validation],dtype=torch.float32) if validation is not None else None
    best, best_epoch, bad = float('inf'),1,0
    count = epochs if epochs is not None else config['epochs']
    generator = torch.Generator().manual_seed(PROTOCOL['seed'])
    for epoch in range(1,count+1):
        model.train()
        for batch in torch.randperm(len(positions),generator=generator).split(PROTOCOL['neural_training']['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            # Smooth absolute loss in volatility-normalized log-return units.
            loss = nn.functional.smooth_l1_loss(model(train_x[batch]),train_y[batch],beta=.2)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),PROTOCOL['neural_training']['gradient_clip'])
            optimizer.step()
        if validation is not None:
            model.eval()
            with torch.no_grad():
                score = float(torch.mean(torch.abs(model(val_x)-val_y)))
            if score < best-1e-5:
                best,best_epoch,bad = score,epoch,0
            else:
                bad += 1
            if bad >= PROTOCOL['neural_training']['patience']:
                break
    return model,best_epoch if validation is not None else count


def fit_model(name, x, y, positions, horizon):
    started = time.monotonic()
    kind = name.rsplit('_',1)[-1]
    train,val = inner_split(x.index,positions,horizon)
    # Fit all preprocessing again after selecting iterations; validation data
    # never supplies scaling or imputation to the inner training fit.
    inner_scaler = PastScaler().fit(x.iloc[train])
    inner_values = inner_scaler.transform(x)
    scaler = PastScaler().fit(x.iloc[positions])
    values = scaler.transform(x)
    if name.startswith('cat_'):
        params = dict(PROTOCOL['cat'][kind],loss_function='MAE',eval_metric='MAE',thread_count=2,
                      random_seed=PROTOCOL['seed'],verbose=False,allow_writing_files=False,has_time=True)
        inner = CatBoostRegressor(**params)
        inner.fit(inner_values[train],y.iloc[train],eval_set=(inner_values[val],y.iloc[val]),
                  early_stopping_rounds=25,use_best_model=True)
        steps = max(1,inner.tree_count_)
        params['iterations'] = steps
        model = CatBoostRegressor(**params).fit(values[positions],y.iloc[positions])
        state = {'cat_blob':bytes(model._serialize_model())}
    else:
        config = PROTOCOL['transformer'][kind]
        _,steps = _neural_fit(inner_values,y,train,config,validation=val)
        model,_ = _neural_fit(values,y,positions,config,epochs=steps)
        state = {'torch_state':deepcopy(model.state_dict())}
    state.update(name=name,scaler=scaler.state(),feature_count=x.shape[1],steps=steps)
    return state, {'training_rows':len(positions),'inner_train_rows':len(train),'inner_validation_rows':len(val),
                   'last_training_target':(x.index[positions[-1]]+np.timedelta64(horizon,'D')).isoformat(),
                   'inner_first_validation':x.index[val[0]].isoformat(),
                   'inner_last_training_target':(x.index[train[-1]]+np.timedelta64(horizon,'D')).isoformat(),
                   'steps':steps,'seconds':time.monotonic()-started}


def predict_model(state, x, positions):
    values = PastScaler.from_state(state['scaler']).transform(x)
    if state['name'].startswith('cat_'):
        model = CatBoostRegressor();model.load_model(blob=state['cat_blob'])
        result = model.predict(values[positions])
    else:
        config = PROTOCOL['transformer'][state['name'].rsplit('_',1)[-1]]
        torch.set_num_threads(2)
        model = PatchTransformer(state['feature_count'],config)
        model.load_state_dict(state['torch_state']);model.eval()
        with torch.no_grad():
            result = torch.cat([model(batch) for batch in sequences(values,positions,config['lookback']).split(256)]).numpy()
    if not np.isfinite(result).all():
        raise ValueError('Non-finite candidate prediction')
    return np.asarray(result,dtype=float)
