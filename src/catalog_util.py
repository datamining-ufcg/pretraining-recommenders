import re
from threading import Lock
from time import sleep
from typing import Any, Callable, List, Optional, Tuple, Union

from src import io_util as io
from src.exception import *
from src.models.CythonModel import CythonModel

lock = Lock()


def open_config(folder: str) -> dict:
    """
    Opens the config file inside the given folder. If such file doesn't
    exist, raises an exception.
    """
    return io.load_json(folder, "config", should_raise=True)


def open_catalog():
    """
    Opens the catalog as a dictionary.
    """
    try:
        return io.load_json("../models", "catalog")
    except:
        raise CatalogNotFoundException()


def create_catalog_key(values: List[Union[str, int, float]]) -> str:
    """
    Given the values of properties that will be used as keys,
    parses and formats them, returning a string.
    """
    vals = [str(v) for v in values]
    key = "\t".join(vals)
    return key


def catalog_keys_from_config(config: dict) -> Tuple:
    """
    Given a config dictionary, returns the properties that will
    be used as keys in the catalog.
    """
    init = config["init"] if "init" in config else None
    method = config["method"] if "method" in config else None
    limit = config["limit"] if "limit" in config else None
    ds_split = config["dataset_split"] if "dataset_split" in config else None
    is_complete = config["is_complete"] if "is_complete" in config else True

    return (init, method, limit, ds_split, is_complete)


def add_to_catalog(path: str, model_folder: str, catalog: dict) -> dict:
    """
    Adds an existing model to the given catalog.
    """
    try:
        config = open_config(f"{path}/{model_folder}")
    except FileNotFoundError as e:
        io.delete_folder(path, model_folder)
        raise e

    values = catalog_keys_from_config(config)
    key = create_catalog_key(values[:-1])

    if path not in catalog:
        catalog[path] = {}

    catalog[path][key] = values[-1]

    return catalog


def add_new_to_catalog(model: Union[CythonModel, Any], is_complete=False) -> None:
    """
    Adds a model to the catalog.
    """
    global lock
    init = model.init_strategy
    method = model.method
    limit = model.limit
    ds_split = model.dataset_split

    key = create_catalog_key([init, method, limit, ds_split])

    while not lock.acquire():
        sleep(0.1)

    catalog = open_catalog()
    path = "/".join(model.model_path.split("/")[:-1])
    if path not in catalog:
        catalog[path] = {}

    catalog[path][key] = is_complete

    kwargs = {"indent": 4, "sort_keys": True}
    io.update_json("../models", "catalog", catalog, **kwargs)
    lock.release()


def check_source_on_catalog(
    target_dataset: str, dataset_foldername: str, method: str, split: int = 1
) -> Tuple[bool, str]:
    """
    Checks if a source model exists and is trained through the catalog.
    Returns a tuple, where the first element indicates if its fully trained
    and the second indicates the path if the training started but was halted.
    """
    path = f"../models/{target_dataset}/{dataset_foldername}/source"
    path = path.replace("_negative", "")
    catalog = open_catalog()
    if path not in catalog:
        return False, ""

    key = f"None\t{method}\t1\t{split}"
    if key not in catalog[path]:
        return False, ""

    return catalog[path][key], path


def check_target_on_catalog(
    target_dataset: str,
    dataset_foldername: str,
    parent_method: str,
    wanted_config: List[Union[str, int, float]],
    catalog: Optional[Union[dict, None]] = None,
) -> Tuple[bool, str]:
    """
    Checks if a target model exists and is trained through the catalog.
    Returns a tuple, where the first element indicates if its fully trained
    and the second indicates the path if the training started but was halted.
    """
    path = f"../models/{target_dataset}/{dataset_foldername}/target"
    foldername = [
        f for f in io.listdir(path, filter_files=True) if f.endswith(parent_method)
    ]

    if len(foldername) == 0:
        return False, ""

    path += f"/{foldername[0]}"
    _catalog = open_catalog() if catalog is None else catalog
    if path not in _catalog:
        return False, ""

    key_pattern = f"{wanted_config[0]}\s+(\w|\s)*_{wanted_config[1]}\s+{wanted_config[2]}\s+{wanted_config[3]}"

    for _key in _catalog[path].keys():
        m = re.match(key_pattern, _key)
        if m:
            return _catalog[path][_key], path

    return False, ""


def _generate_empty_paths(*args):
    return False, ""


def _filter_configs(
    target_dataset: str,
    dataset_foldername: str,
    parent_method: str,
    params: List,
    fn: Callable,
):
    filtered_params = []
    paths = []

    for (inst, inpr), limit, split in params:
        wanted = [inst, parent_method, limit / 10, split]
        catalog_response = fn(target_dataset, dataset_foldername, parent_method, wanted)
        if not catalog_response[0]:
            filtered_params.append(((inst, inpr), limit, split))
            paths.append(catalog_response[1])

    return list(zip(filtered_params, paths))


def filter_target_params_with_catalog(
    target_dataset: str, dataset_foldername: str, parent_method: str, params: List,
) -> List[Tuple[Tuple, str]]:
    path = f"../models/{target_dataset}/{dataset_foldername}/target"
    foldername = [
        f for f in io.listdir(path, filter_files=True) if f.endswith(parent_method)
    ]

    if len(foldername) == 0:
        return _filter_configs(
            target_dataset,
            dataset_foldername,
            parent_method,
            params,
            _generate_empty_paths,
        )

    path += f"/{foldername[0]}"
    catalog = open_catalog()
    if path not in catalog:
        return _filter_configs(
            target_dataset,
            dataset_foldername,
            parent_method,
            params,
            _generate_empty_paths,
        )

    return _filter_configs(
        target_dataset,
        dataset_foldername,
        parent_method,
        params,
        check_target_on_catalog,
    )
