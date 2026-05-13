from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

FEATURE_COLS = [
    'a_mean', 'area_ratio', 'pct_green', 'pct_yellow', 'pct_brown',
    'L_mean', 'b_mean', 'a_std', 'L_std',
    'variety_enc',  # COS=0, GOK=1
    # texture features (GLCM) — เพิ่ม Phase 9b, ให้ MAE ดีขึ้น ~0.027 วัน, R²=0.904
    'contrast', 'correlation', 'energy', 'homogeneity',
]
TARGET = 'day'


def prepare_features(features_csv: str | Path = 'data/features.csv') -> pd.DataFrame:
    df = pd.read_csv(features_csv)
    df['variety_enc'] = (df['variety'] == 'GOK').astype(int)
    return df


def train_cv(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[XGBRegressor, dict]:
    """
    Train XGBoost ด้วย GroupKFold by plant_id
    Returns best model (refit on all data) และ cv metrics
    """
    X = df[FEATURE_COLS].values
    y = df[TARGET].values
    groups = df['plant_id'].values

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )

    gkf = GroupKFold(n_splits=n_splits)
    mae_list, rmse_list, r2_list = [], [], []
    oof_pred = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        m = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            random_state=random_state, n_jobs=-1, verbosity=0,
        )
        m.fit(X[train_idx], y[train_idx])
        pred = m.predict(X[val_idx])
        oof_pred[val_idx] = pred
        mae_list.append(mean_absolute_error(y[val_idx], pred))
        rmse_list.append(np.sqrt(mean_squared_error(y[val_idx], pred)))
        r2_list.append(r2_score(y[val_idx], pred))
        print(f'  Fold {fold+1}: MAE={mae_list[-1]:.3f}  RMSE={rmse_list[-1]:.3f}  R2={r2_list[-1]:.3f}')

    cv_metrics = {
        'mae_mean':  float(np.mean(mae_list)),
        'mae_std':   float(np.std(mae_list)),
        'rmse_mean': float(np.mean(rmse_list)),
        'rmse_std':  float(np.std(rmse_list)),
        'r2_mean':   float(np.mean(r2_list)),
        'r2_std':    float(np.std(r2_list)),
        'oof_pred':  oof_pred,
    }

    # refit on all data
    model.fit(X, y)
    return model, cv_metrics


def evaluate_by_variety(df: pd.DataFrame, oof_pred: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    df['pred'] = oof_pred
    df['error'] = df['pred'] - df[TARGET]
    rows = []
    for var in ['COS', 'GOK', 'ALL']:
        sub = df if var == 'ALL' else df[df['variety'] == var]
        rows.append({
            'variety': var,
            'MAE':  round(mean_absolute_error(sub[TARGET], sub['pred']), 3),
            'RMSE': round(np.sqrt(mean_squared_error(sub[TARGET], sub['pred'])), 3),
            'R2':   round(r2_score(sub[TARGET], sub['pred']), 3),
        })
    return pd.DataFrame(rows)


def save_model(model: XGBRegressor, path: str | Path = 'models/xgb_model.json'):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    print(f'Model saved: {path}')


def load_model(path: str | Path = 'models/xgb_model.json') -> XGBRegressor:
    model = XGBRegressor()
    model.load_model(str(path))
    return model
