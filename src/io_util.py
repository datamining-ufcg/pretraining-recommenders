import json
import os
import shutil

import pandas as pd

from src.exception import *


def load_json(filepath, filename, should_raise=False):
    """
    Loads JSON file as a dictionary.
    """
    fn = filename.replace(".json", "")
    try:
        with open(f"{filepath}/{fn}.json", "r") as jsf:
            data = json.load(jsf)
    except FileNotFoundError as e:
        if not should_raise:
            data = {}
        else:
            raise e

    return data


def save_json(filepath, filename, data, **kwargs):
    """
    Saves dictionary as a JSON file.
    """
    with open(f"{filepath}/{filename}.json", "w") as jsf:
        json.dump(data, jsf, **kwargs)


def update_json(filepath, filename, data, **kwargs):
    """
    Updates the existing json with the provided data. If the
    specified json does not exist, creates it.
    """
    previous_data = load_json(filepath, filename)
    updated_data = {**previous_data, **data}
    save_json(filepath, filename, updated_data, **kwargs)


def show_json(object, **kwargs):
    return json.dumps(object, **kwargs)


def listdir(path, filter_files=False):
    """
    Lists files and directories in the given path.
    """
    try:
        folders = os.listdir(path)
        if filter_files:
            folders = _filter_files(folders)
        folders.sort()

    except FileNotFoundError:
        folders = []

    return folders


def check_file(fname, path):
    """
    Checks if a file exists in the given path.
    """
    return fname in os.listdir(path)


def create_folder(path, folder_name):
    """
    Creates a folder in that path with the given name.
    If the folder already exists, does nothing.
    """

    files = listdir(path)
    if folder_name not in files:
        pw = path[:-1] if path.endswith("/") else path
        try:
            os.mkdir(f"{pw}/{folder_name}/")
        except FileExistsError:
            pass


def delete_folder(path, folder_name):
    """
    Deletes the folder in the given path.
    """
    shutil.rmtree(f"{path}/{folder_name}")


def remove_file(filepath):
    """
    Deletes a file. If the file is not found or the given
    path correponds to a directory, does nothing.
    """

    try:
        os.remove(filepath)
    except FileNotFoundError:
        pass
    except IsADirectoryError:
        pass


def read_ratings_as_list(filepath, filename):
    """
    Read ratings file as a list.
    """
    path = f"{filepath}/{filename}"
    ratingList = []

    with open(path, "r") as f:
        line = f.readline()

        while line != None and line != "":
            arr = line.split("\t")
            user, item = arr[0], arr[1]
            ratingList.append([user, item])
            line = f.readline()

    return ratingList


def read_negative_file(filepath, filename):
    path = f"{filepath}/{filename}"
    negativeList = []

    with open(path, "r") as f:
        line = f.readline()

        while line != None and line != "":
            arr = line[:-1].split("\t")
            negatives = arr[1:]
            negativeList.append(negatives)
            line = f.readline()

    return negativeList


def save_triples(path, filename, triples):
    """
    """
    df = pd.DataFrame.from_records(triples)
    df.to_csv(f"{path}/{filename}", sep="\t", header=None, index=False)


def save_negatives(path, filename, negatives):
    """
    """
    with open(f"{path}/{filename}", "w") as f_out:
        for negative in negatives:
            f_out.write("\t".join(negative))
            f_out.write("\n")


def _filter_files(files):
    """
    Discards files when listing folders.
    """
    filtered = []
    for f in files:
        if f == ".gitkeep" or f.endswith(".zip") or f.endswith(".json"):
            continue

        filtered.append(f)

    return filtered
