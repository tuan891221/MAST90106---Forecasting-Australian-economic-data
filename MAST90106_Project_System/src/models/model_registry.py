from __future__ import annotations

from src.models.ar_model import ARModel
from src.models.bvar_model import BVARModel
from src.models.factor_model import FactorModel
from src.models.mean_model import MeanModel
from src.models.naive_model import NaiveModel
from src.models.var_model import VARModel


def get_model(model_name: str, config: dict):
    name = str(model_name).lower()

    if name == "naive":
        return NaiveModel(config)
    if name == "mean":
        return MeanModel(config)
    if name == "ar":
        return ARModel(config)
    if name == "var":
        return VARModel(config)
    if name == "bvar":
        return BVARModel(config)
    if name == "factor":
        return FactorModel(config)

    raise ValueError(f"Unknown model: {model_name}")