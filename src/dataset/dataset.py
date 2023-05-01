from abc import ABC, abstractmethod
import numpy as np
import logging
from surprise import Dataset as SurpriseDataset, Trainset
from typing import Union

from src.dataset.raw_dataset import RawDataset
from src.embeddings import EmbeddingGenerator
from src.exception import WrongParamsMappingException
from src.matching import match_dataframes
import src.io_util as io


class Dataset(RawDataset, ABC):
    def __init__(self, name, **kwargs) -> None:
        self.data_path = "../data"
        super(Dataset, self).__init__(name, **kwargs)

    @property
    def n_users(self):
        return self.trainset.n_users

    @property
    def n_items(self):
        return self.trainset.n_items

    @property
    def n_ratings(self):
        return self.trainset.n_ratings

    @property
    @abstractmethod
    def is_loaded(self):
        pass

    @property
    @abstractmethod
    def ds_path(self):
        """
        Path to dataset folder.
        """
        pass

    @property
    @abstractmethod
    def sep(self):
        """
        Separator character for the ratings file.
        """
        pass

    @property
    @abstractmethod
    def reader(self):
        return None

    @property
    @abstractmethod
    def ratings_file(self):
        """
        Ratings' file name.
        """
        pass

    @property
    @abstractmethod
    def ratings_intermediate_file(self):
        """
        Intermediate file name for implicit ratings. This file is
        generated when you are applying transformations from explicit
        to implicit feedback.
        """
        pass

    @property
    @abstractmethod
    def items_file(self):
        """
        Item description file name.
        """
        pass

    @property
    @abstractmethod
    def items_intermediate_file(self):
        """
        Intermediate file for item description. This file is
        generated when you are applying transformations from explicit
        to implicit feedback.
        """
        pass

    @property
    def filepath(self):
        """
        self.ds_path/self.ratings_file
        """
        return f"{self.ds_path}/{self.ratings_file}"

    def load(self, filepath=None):
        """
        Loads data from the ratings file.
        """
        self.check_and_preprocess()
        fpath = filepath if filepath is not None else self.filepath
        self.data = SurpriseDataset.load_from_file(fpath, self.reader)
        return self.data

    @abstractmethod
    def load_as_dataframe(self):
        pass

    def load_inner_mappings(self):
        filename = f"raw2inner_uids_{self.current_fold}.json"
        if not self.check_file(filename):
            raise Exception("Please call `load` to generate the inner mappings files.")

        raw2inner_uids = io.load_json(
            self.ds_path, f"raw2inner_uids_{self.current_fold}"
        )
        raw2inner_iids = io.load_json(
            self.ds_path, f"raw2inner_iids_{self.current_fold}"
        )

        self.user_inner_mapping = raw2inner_uids
        self.item_inner_mapping = raw2inner_iids

    @abstractmethod
    def load_item_description(self, path, read_params, adjust_cols=True):
        """
        Loads item description file as a Pandas dataframe.
        """
        pass

    @abstractmethod
    def preprocess(self):
        """
        Preprocess ratings file if necessary.
        """
        pass

    @abstractmethod
    def preprocess_items(self):
        """
        Preprocess items file if necessary.
        """
        pass

    @abstractmethod
    def preprocess_min_ratings(self):
        """
        Preprocess ratings file for keeping only core
        users and items, i.e., users and items with at least
        20 ratings.
        """
        pass

    @abstractmethod
    def preprocess_items_min_ratings(self):
        """
        Preprocess items file for keeping only core items, i.e.,
        items with at least 20 ratings.
        """
        pass

    def check_file(self, fname):
        return super().check_file(fname, self.ds_path)

    def check_and_retrieve_mapping(self, other):
        """
        Checks if a mapping between datasets exists.
        If so, retrieves the existing mapping.
        Otherwise, returns None.
        """

        if not self.check_file("mapping.json"):
            return None

        mapping = io.load_json(self.ds_path, "mapping")

        if other.name in mapping:
            return mapping[other.name]

        return None

    def save_mapping(self, other, mapping):
        """
        Saves a mapping between datasets. If this dataset has other mappings,
        saved, updates the file. Otherwise, creates a new file.
        """

        previous_mapping = io.load_json(self.ds_path, "mapping")
        mapping_str = {str(k): str(v) for k, v in mapping.items()}
        updated_mapping = {**previous_mapping, other.name: mapping_str}
        io.save_json(self.ds_path, "mapping", updated_mapping)

    def to_inner_uid(self, ext_id):
        """
        Maps the user id of the raw dataset to the user id used internally by Surprise.
        """
        _ext_id = str(ext_id) if (not isinstance(ext_id, str)) else ext_id
        uid = self.user_inner_mapping[_ext_id]

        return uid

    def to_inner_iid(self, ext_id):
        """
        Maps the item id of the raw dataset to the item id used internally by Surprise.
        """

        _ext_id = str(ext_id) if (not isinstance(ext_id, str)) else ext_id
        iid = self.item_inner_mapping[_ext_id]

        return iid

    def _solve_mapping(self, mapping, other_trainset):
        """
        Maps the item id of the current trainset with the item id of another trainset.
        """
        inner_mapping = {}
        for k, v in mapping.items():
            try:
                key = self.to_inner_iid(k)
                value = other_trainset.to_inner_iid(str(v))
                inner_mapping[key] = value
            except KeyError:
                pass
            except ValueError:
                pass

        return inner_mapping

    def map_ids(self, other, other_trainset=None, previous_mapping={}, raw=False):
        """

        """
        if not raw and other_trainset is None:
            raise WrongParamsMappingException()

        mapping = self.check_and_retrieve_mapping(other)

        if not hasattr(self, "item_data"):
            self.load_item_description()
        other_data = other.load_item_description()

        if mapping is None:
            mapping = match_dataframes(self.item_data, other_data, previous_mapping)
            self.save_mapping(other, mapping)

        if raw:
            return mapping

        inner_mapping = self._solve_mapping(mapping, other_trainset)

        hits = round(len(set(inner_mapping.values())) / other_trainset.n_items, 2) * 100
        logging.debug(f"Target is {hits}% mapped.")

        return inner_mapping

    def get_fold(self, n_folds=5, k=1):
        trainset, testset = super().get_fold(n_folds, k)
        self.trainset = trainset
        self.testset = testset

        self._save_inner_mapping()
        return trainset, testset

    def _save_inner_mapping(self):
        """
        Saves the mapping to the inner id of a Surprise Trainset.
        """
        template = f"raw2inner_%sids_{self.current_fold}"

        raw2inner_uids = self.trainset._raw2inner_id_users
        io.save_json(self.ds_path, template % "u", raw2inner_uids)
        self.user_inner_mapping = raw2inner_uids

        raw2inner_iids = self.trainset._raw2inner_id_items
        io.save_json(self.ds_path, template % "i", raw2inner_iids)
        self.item_inner_mapping = raw2inner_iids

    def check_and_load(self):
        if not self.is_loaded:
            self.load()

    @abstractmethod
    def check_ratings_intermediate_file():
        pass

    @abstractmethod
    def check_preprocessed():
        pass

    def check_and_preprocess(self, remove=True):
        if self.check_ratings_intermediate_file():
            logging.debug(
                "Preprocessing ratings intermediate file. This only happens once."
            )
            self.preprocess_min_ratings()

        elif not self.check_preprocessed():
            logging.debug("Preprocessing files. This only happens once.")
            self.preprocess()

        if remove:
            io.remove_file(f"{self.ds_path}/{self.ratings_intermediate_file}")

    def check_preprocessed_items(self):
        return self.check_file(self.items_file)

    def check_items_intermediate_file(self):
        return self.check_file(self.items_intermediate_file)

    def check_and_preprocess_items(self):
        if self.check_items_intermediate_file():
            logging.debug(
                "Preprocessing items intermediate file. This only happens once."
            )
            self.preprocess_items_min_ratings()

        elif not self.check_preprocessed_items():
            logging.debug("Preprocessing item files. This only happens once.")
            self.preprocess_items()

        else:
            return

        io.remove_file(f"{self.ds_path}/{self.items_intermediate_file}")

    def get_embeddings(
        self, implicit, embedding_dim, method="random", user=False, **kwargs
    ):
        """
        Retrieves embeddings given the configuration parameters.
        """

        ext = "implicit" if implicit else "explicit"
        emb_filename = f"{ext}_{method}_{self.current_fold}"

        if method != "random" and self.check_file(f"{emb_filename}.npy"):
            embeddings = np.load(f"{self.ds_path}/{emb_filename}.npy")
        else:
            embeddings = super().get_embeddings(embedding_dim, method, user, **kwargs)

        if method != "random":
            np.save(f"{self.ds_path}/{emb_filename}", embeddings)

        return embeddings

    def _create_ratings_mask(self, ratings: np.array):
        return np.where(ratings > 0, 1, 0)
