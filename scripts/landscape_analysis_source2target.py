import numpy as np
import pandas as pd
from surprise import Trainset
import sys
from tqdm import tqdm
from typing import List

sys.path.append("../")

from src import io_util as io
from src.dataset import Dataset
from src.dataset.ml_transfer import ExplicitMLTransfer
from src.models.CythonSVD import CythonSVD
from src.util import get_target_model_path


def get_kwargs(dataset_foldername):
    target_kwargs = {
        "target_size": "1m",
        "dataset_foldername": dataset_foldername,
        "ratings_filename": "ml1m",
        "items_filename": "movies.dat",
        "items_separator": "::",
        "is_source": False,
    }
    return target_kwargs


def loss(data, model):
    loss = 0
    for (u, i, r) in data:
        pred = model.global_bias + model.bu[u - 1] + model.bi[i - 1]
        pred += np.dot(model.pu[u - 1], model.qi[i - 1])
        loss += (r - pred) ** 2 + model.reg * (
            model.bu[u - 1] ** 2
            + model.bi[i - 1] ** 2
            + np.linalg.norm(model.pu[u - 1]) ** 2
            + np.linalg.norm(model.qi[i - 1]) ** 2
        )
    return loss


def get_interpolated_rmse(ds, eval_data, model_1, model_2, _min, _max):
    alphas = np.linspace(_min, _max, 50)

    model_ = CythonSVD(
        ds,
        128,
        "random",
        {"stddev": 0.1},
        0.4,
        model_path="interpolate",
        skip_folder_creation=True,
    )

    pu_0 = model_1.pu
    qi_0 = model_1.qi
    bu_0 = model_1.bu
    bi_0 = model_1.bi

    pu_1 = model_2.pu
    qi_1 = model_2.qi
    bu_1 = model_2.bu
    bi_1 = model_2.bi

    rmses = []

    for alpha in alphas:
        pu = (1 - alpha) * pu_0 + alpha * pu_1
        qi = (1 - alpha) * qi_0 + alpha * qi_1
        bu = (1 - alpha) * bu_0 + alpha * bu_1
        bi = (1 - alpha) * bi_0 + alpha * bi_1
        model_.set_params(pu, bu, qi, bi)
        data = eval_data if isinstance(eval_data, list) else eval_data.all_ratings()

        pred = model_.predictions(data)
        rmse = model_.rmse(pred)
        rmses.append(rmse)

    return alphas, rmses


def interpolation_csv(
    dataset: Dataset,
    train_data: Trainset,
    test_data: List,
    model_1: CythonSVD,
    model_2: CythonSVD,
    _min: float,
    _max: float,
    path_name: str,
):
    alphas_train, rmse_train = get_interpolated_rmse(
        dataset, train_data, model_1, model_2, _min, _max
    )
    _, rmse_test = get_interpolated_rmse(
        dataset, test_data, model_1, model_2, _min, _max
    )

    df = pd.DataFrame(
        zip(alphas_train, rmse_train, rmse_test),
        columns=["alphas", "rmse_train", "rmse_test"],
    )
    df.to_csv(path_name, index=False)


for folder in io.listdir(f"../models/target_1m", filter_files=True):
    ds_folder = [*folder.split("_"), "from"]
    ds_folder[-1], ds_folder[-2] = ds_folder[-2], ds_folder[-1]
    ds_folder = "_".join(ds_folder)

    for sm in tqdm(
        io.listdir(f"../models/target_1m/{folder}/target/", filter_files=True)
    ):
        _sm = sm.split("_")[-1]
        dataset = ExplicitMLTransfer(**get_kwargs(ds_folder))
        path = f"../models/target_1m/{folder}/target/{sm}"
        baseline_path = get_target_model_path(qp=(("random", {}), 0, 1), path=path)
        full_path = get_target_model_path(qp=(("random", {}), 10, 1), path=path)

        baseline = CythonSVD.load_model(baseline_path)
        full = CythonSVD.load_model(full_path)

        dataset.load()
        trainset, testset = dataset.get_fold(k=1)
        pathname = f"../results/target_1m/{folder}/interpolate_{_sm}.csv"
        interpolation_csv(dataset, trainset, testset, baseline, full, 0, 1, pathname)

