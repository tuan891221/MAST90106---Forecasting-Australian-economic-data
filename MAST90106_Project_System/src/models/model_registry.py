from __future__ import annotations

from src.models.ar_model import ARModel
from src.models.bvar_model import BVARModel
from src.models.factor_model import FactorModel
from src.models.mean_model import MeanModel
from src.models.naive_model import NaiveModel
from src.models.var_model import VARModel


def get_model(name: str, config: dict):
    if name == "naive":
        return NaiveModel()
    if name == "mean":
        return MeanModel()
    if name == "ar":
        return ARModel(max_lag=config["models"]["ar"]["max_lag"])
    if name == "var":
        return VARModel(max_lag=config["models"]["var"]["max_lag"])
    if name == "bvar":
        bcfg = config["models"]["bvar"]
        return BVARModel(
            max_lag=bcfg["max_lag"],
            lambda_1=bcfg["lambda_1"],
            lambda_2=bcfg["lambda_2"],
            lambda_3=bcfg["lambda_3"],
        )
    if name == "factor":
        fcfg = config["models"]["factor"]
        return FactorModel(max_lag=fcfg["max_lag"], n_factors=fcfg["n_factors"])
    raise ValueError(f"Unknown model: {name}")