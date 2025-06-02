# src/models/xgb_model.py

import xgboost as xgb


def build_xgb_model(params: dict = None):
    """
    Returns an XGBoost classifier instance.
    """
    default_params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "learning_rate": 0.1,
        "max_depth": 4,
        "n_estimators": 100,
        "use_label_encoder": False,
        "eval_metric": "mlogloss"
    }
    if params:
        default_params.update(params)
    return xgb.XGBClassifier(**default_params)
