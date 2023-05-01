import logging
import threading
from typing import Any, Callable, List, NewType, Tuple, Union

from src.catalog_util import add_new_to_catalog
from src.config import EMBEDDING_SIZE, EPOCHS, LR, PATIENCE, REG
from src.dataset import Dataset, load_from_name
from src.dataset.ml_transfer import ExplicitMLTransfer
from src.dataset.netflix_transfer import ExplicitNetflixTransfer
from src.models.CythonSVD import CythonSVD
from src.util import (
    build_path_from_source_kwargs,
    build_path_from_target_kwargs,
    get_source_model_path,
    get_target_model_path,
)


SourceQueueParams = NewType("SourceQueue", Tuple[Tuple[str, dict], str, int])
TransferQueueParams = NewType("TransferQueueParams", Tuple[Tuple[str, dict], int, int])


def _load_or_create_source_model(
    path: str,
    source_dataset: Dataset,
    init_method: str,
    init_method_params: dict,
    output_dir: str,
):
    if path != "":
        logging.debug("Loading halted model")
        source_model = CythonSVD.load_model(path)

    else:
        source_model = CythonSVD(
            source_dataset,
            EMBEDDING_SIZE,
            init_method,
            init_method_params,
            REG,
            model_path=output_dir,
        )

    return source_model


def _load_or_create_target_model(
    path: str,
    source_model: CythonSVD,
    source_dataset: Dataset,
    target_dataset: Dataset,
    target_trainset: Any,
    qp: TransferQueueParams,
    output_dir: str,
) -> CythonSVD:
    if path != "":
        logging.debug("Loading halted model")
        target_model = CythonSVD.load_model(path)

    else:
        (inst, inpr), limit, _ = qp
        _limit = limit / 10
        mapping = source_dataset.map_ids(target_dataset, target_trainset)

        target_model = CythonSVD(
            target_dataset, EMBEDDING_SIZE, inst, inpr, REG, model_path=output_dir
        )

        target_model.use_pretrained_items(
            source_model.qi, mapping, source_dataset.name, source_model.method, _limit
        )

    return target_model


def _train_svd_source(
    queue_params: SourceQueueParams, source_kwargs: dict,
):
    source_dataset = ExplicitMLTransfer(**source_kwargs)
    source_dataset.load()

    output_dir = build_path_from_source_kwargs(source_kwargs)
    logging.debug(f"Training {len(queue_params)} models")

    for enum_idx, ((init_method, init_method_params, split), path) in enumerate(
        queue_params
    ):
        source_trainset, source_testset = source_dataset.get_fold(k=split)
        _path = (
            get_source_model_path(source_kwargs, init_method, split)
            if path != ""
            else ""
        )
        source_model = _load_or_create_source_model(
            _path, source_dataset, init_method, init_method_params, output_dir
        )
        add_new_to_catalog(source_model, is_complete=False)

        source_model.fit(
            source_trainset,
            source_testset,
            EPOCHS,
            LR,
            PATIENCE,
            verbose=False,
            early_stopping=False,
        )
        source_model.save()
        add_new_to_catalog(source_model)
        logging.debug(f"Completed {enum_idx + 1} jobs.")


def _train_svd_target(
    queue_params: Union[List[TransferQueueParams], TransferQueueParams],
    source_model: CythonSVD,
    source_dataset: Dataset,
    target_kwargs: dict,
) -> None:
    params = queue_params if isinstance(queue_params, list) else [queue_params]
    target_dataset = ExplicitMLTransfer(**target_kwargs)
    target_dataset.load()

    output_dir = build_path_from_target_kwargs(
        target_kwargs, source_method=source_model.method
    )

    for enum_idx, (qp, path) in enumerate(params):
        _, _, split = qp
        _path = get_target_model_path(qp, path) if path != "" else ""
        target_trainset, target_testset = target_dataset.get_fold(k=split)
        target_model = _load_or_create_target_model(
            _path,
            source_model,
            source_dataset,
            target_dataset,
            target_trainset,
            qp,
            output_dir,
        )
        add_new_to_catalog(target_model, is_complete=False)

        target_model.fit(
            target_trainset,
            target_testset,
            EPOCHS,
            LR,
            PATIENCE,
            verbose=False,
            early_stopping=False,
        )
        target_model.save()
        add_new_to_catalog(target_model)
        logging.debug(f"Completed {enum_idx + 1} jobs.")


def _train_netflix_source(
    queue_params: SourceQueueParams, source_kwargs: dict,
):
    source_dataset = ExplicitNetflixTransfer(**source_kwargs)
    source_dataset.load()
    source_trainset, source_testset = source_dataset.get_fold(k=1)

    output_dir = build_path_from_source_kwargs(source_kwargs)
    logging.debug(f"Training {len(queue_params)} models")

    for enum_idx, ((init_method, init_method_params), path) in enumerate(queue_params):
        _path = get_source_model_path(source_kwargs, init_method) if path != "" else ""
        source_model = _load_or_create_source_model(
            _path, source_dataset, init_method, init_method_params, output_dir
        )
        add_new_to_catalog(source_model, is_complete=False)

        source_model.fit(
            source_trainset,
            source_testset,
            EPOCHS,
            LR,
            PATIENCE,
            verbose=False,
            early_stopping=False,
        )
        source_model.save()
        add_new_to_catalog(source_model)
        logging.debug(f"Completed {enum_idx + 1} jobs.")


def _train_netflix_target(
    queue_params: Union[List[TransferQueueParams], TransferQueueParams],
    source_model: CythonSVD,
    source_dataset: Dataset,
    target_kwargs: dict,
):
    params = queue_params if isinstance(queue_params, list) else [queue_params]
    target_dataset = load_from_name(**target_kwargs)()
    target_dataset.load()

    output_dir = build_path_from_target_kwargs(
        target_kwargs, source_method=source_model.method
    )

    for enum_idx, (qp, path) in enumerate(params):
        _, _, split = qp
        _path = get_target_model_path(qp, path) if path != "" else ""
        target_trainset, target_testset = target_dataset.get_fold(k=split)
        target_model = _load_or_create_target_model(
            _path,
            source_model,
            source_dataset,
            target_dataset,
            target_trainset,
            qp,
            output_dir,
        )
        add_new_to_catalog(target_model, is_complete=False)

        target_model.fit(
            target_trainset,
            target_testset,
            EPOCHS,
            LR,
            PATIENCE,
            verbose=False,
            early_stopping=False,
        )
        target_model.save()
        add_new_to_catalog(target_model)
        logging.debug(f"Completed {enum_idx + 1} jobs.")


def _wait_for_threads(fn: Callable, queues: List, *args) -> None:
    from src.config import MAX_THREADS

    threads = []

    for i in range(MAX_THREADS):
        if len(queues[i]) == 0:
            continue

        t = threading.Thread(target=fn, args=[queues[i], *args])
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


def train_svd_source(queues: List, source_kwargs: dict,) -> None:
    _wait_for_threads(
        _train_svd_source, queues, source_kwargs,
    )


def train_svd_target(
    queues: List, source_model: CythonSVD, source_dataset: Dataset, target_kwargs: dict,
) -> None:
    _wait_for_threads(
        _train_svd_target, queues, source_model, source_dataset, target_kwargs
    )


def train_netflix_source(queues: List, source_kwargs: dict,) -> None:
    _wait_for_threads(
        _train_netflix_source, queues, source_kwargs,
    )


def train_netflix_target(
    queues: List, source_model: CythonSVD, source_dataset: Dataset, target_kwargs: dict,
) -> None:
    _wait_for_threads(
        _train_netflix_target, queues, source_model, source_dataset, target_kwargs
    )
