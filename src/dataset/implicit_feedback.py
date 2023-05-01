import itertools
from collections import defaultdict
import numpy as np
from scipy.sparse import dok_matrix
from surprise import Reader
from surprise.model_selection import LeaveOneOut

import src.io_util as io
from src.exception import NotLoadedException
from src.dataset.dataset import Dataset


class ImplicitFeedback(Dataset):
    """
    Dataset inspired in the one from https://github.com/hexiangnan/neural_collaborative_filtering,
    also used in https://github.com/google-research/google-research/blob/master/ials. However,
    we inherit from RawDataset to use the initialization methods we are proposing to evaluate.
    """
    def __init__(self, name) -> None:
        super(ImplicitFeedback, self).__init__(name)

    # @property
    # def n_users(self):
    #     return self.get_current_metadata()['n_users']

    # @property
    # def n_items(self):
    #     return self.get_current_metadata()['n_items']

    # @property
    # def n_ratings(self):
    #     return self.get_current_metadata()['n_ratings']

    @property
    def reader(self):
        return Reader(sep='\t')

    @property
    def train_template(self):
        return 'train_{}.rating'

    @property
    def test_template(self):
        return 'test_{}.rating'

    @property
    def negatives_template(self):
        return 'test_{}.negatives'

    @property
    def is_loaded(self):
        return hasattr(self, 'metadata') and hasattr(self, 'train_ratings') and \
            hasattr(self, 'test_ratings') and hasattr(self, 'test_negatives')

    @property
    def has_metadata(self):
        return hasattr(self, 'metadata')

    def load(self, load_data=True):
        """
        Unlike expected, this function does not load the data. Instead, it calls
        the preprocessing functions that allow calling `get_fold`, which does load
        the given partition of the data.
        """
        if (not hasattr(self, 'data') and load_data):
            super().load()
        
        self.check_implicit_and_preprocess()
        self.load_metadata()

    def get_fold(self, k=1):
        """
        Gets train ratings, test ratings and test negatives for the k-th fold.
        In the case of Implicit Feedback, the parameter n_folds is ignored and
        the number of folds defaults to 5. To change this setting, please modify
        the preprocess method.
        """
        self.current_fold = k
        self.train_ratings = self.load_train_ratings()
        self.test_ratings = self.load_test_ratings()
        self.test_negatives = self.load_test_negatives()

        return self.train_ratings, self.test_ratings, self.test_negatives

    def get_train_batches(self, num_batches=1):
        if (not self.is_loaded):
            raise NotLoadedException()

        tbu = defaultdict(list)
        tbi = defaultdict(list)

        for u, i, _ in self.trainset.all_ratings():
            tbu[u].append(i)
            tbi[i].append(u)

        tbu = list(tbu.items())
        tbi = list(tbi.items())

        tbu_batched = self.batch(tbu, num_batches)
        tbi_batched = self.batch(tbi, num_batches)

        return tbu_batched, tbi_batched

    def check_implicit_and_preprocess(self):
        """
        Checks if the data is already preprocessed to the implicit predefined
        structure. If not, preprocesses it.
        """
        if (not self.check_implicit_preprocessed()):
            self.implicit_preprocess()

    def check_implicit_preprocessed(self):
        """
        Checks if there are ratings and negatives files for each split.
        """
        files = [self.train_template, self.test_template, self.negatives_template]
        splits = list(range(1, 6))
        return all([
            self.check_file(fname.format(split)) for \
                fname, split in itertools.product(files, splits)
        ])

    def kfold(self, k=5):
        """
        Implicit Feedback uses the Leave One Out strategy along negative sampling
        for model evaluation. We also restrain the test set to have at least one
        rating for each user.
        """
        kfold = LeaveOneOut(k, random_state=self.STATE, min_n_ratings=1)
        return kfold.split(self.data)

    def implicit_preprocess(self):
        """
        Preprocess the ratings data to the implicit format, splitted in folds.
        """
        if (not hasattr(self, 'data')):
            raise NotLoadedException()
        
        for kth in range(1, 6):
            super().get_fold(k=kth)
            self.revert_inner_ids()
            self.update_metadata()
            self.generate_negatives()

    def _revert_inner(self, iterator, filename):
        """
        Reverts Surprise internal mapping and saves ratings.
        """
        triples = []
        for u, i, r in iterator:
            ruid = self.trainset.to_raw_uid(u)
            riid = self.trainset.to_raw_iid(i)

            triples.append((ruid, riid, r))

        io.save_triples(self.ds_path, filename, triples)

    def revert_inner_ids(self):
        """
        Reverts Surprise internal mapping and saves train and test ratings.
        """
        self._revert_inner(
            self.trainset.all_ratings(),
            self.train_template.format(self.current_fold)
        )

        self._revert_inner(
            self.testset,
            self.test_template.format(self.current_fold)
        )

    def generate_negatives(self):
        """
        Generates 99 negative samples for each user in the test set.
        """
        negative_samples = []
        for u, i, r in self.testset:
            positives = set([ k for k, v in self.trainset.ur[u] ])
            negatives = set(list(range(self.trainset.n_items)))
            negatives = negatives - positives
            negatives = list(negatives - { i })
            test_positive = str((self.trainset.to_raw_uid(u), self.trainset.to_raw_iid(i)))
            sample = np.random.choice(negatives, size=(99,))
            sample = [self.trainset.to_raw_iid(s) for s in sample]
            sample = [test_positive, *sample]
            negative_samples.append(sample)

        io.save_negatives(
            self.ds_path,
            self.negatives_template.format(self.current_fold),
            negative_samples
        )

    def update_metadata(self):
        """
        Updates metadata file with current trainset dimensions.
        """
        split = {
            self.current_fold: {
                'n_users': self.trainset.n_users,
                'n_items': self.trainset.n_items,
                'n_ratings': self.trainset.n_ratings
            }
        }
        io.update_json(self.ds_path, 'implicit_metadata', split)

    def load_metadata(self):
        """
        Load user and item dimensions from metadata.
        """
        self.metadata = io.load_json(self.ds_path, 'implicit_metadata')
        return self.metadata

    def get_current_metadata(self):
        """
        Retrieves metadata for the current fold.
        """
        if (not hasattr(self, 'current_fold')):
            raise NotLoadedException()
        
        return self.metadata[str(self.current_fold)]

    def load_train_ratings(self, filepath=None):
        """
        Reads .rating file as a dok matrix.
        """
        super().load(filepath)
        self.trainset = self.data.build_full_trainset()

        mat = dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        for u, i, r in self.trainset.all_ratings():
            if (r > 0):
                mat[u, i] = 1.0
        
        return mat

    def load_test_ratings(self):
        """
        Reads test rating file as a list and transforms the user and item ids into
        the ids used internally.
        """
        ratings_list = io.read_ratings_as_list(
            self.ds_path, self.test_template.format(self.current_fold)
        )
        ratings_list = [
            [self.trainset.to_inner_uid(u), self.trainset.to_inner_iid(i)] \
                for u, i in ratings_list
        ]

        return ratings_list
    
    def load_test_negatives(self):
        """
        Reads the test negatives file as a list and transforms the negative item
        ids into the ids used internally.
        """
        negative_list = io.read_negative_file(
            self.ds_path, self.negatives_template.format(self.current_fold)
        )
        negative_list = [
            [self.trainset.to_inner_iid(i) for i in negative_items] \
                for negative_items in negative_list
        ]
        return negative_list

    def get_embeddings(self, embedding_dim, method='random', user=False, **kwargs):
        """
        Retrieves the embeddings, given the parameters. Also, activates sigma.
        """
        kwargs['use_sigma'] = True
        return super().get_embeddings(True, embedding_dim, method, user, **kwargs)

    def cleanup(self):
        """
        Deles all files created, keeping only the files native to the dataset.
        """
        for fname in io.listdir(self.ds_path):
            if (fname.startswith('raw2inner') or \
                fname.endswith('.negatives') or \
                fname.endswith('.rating') or \
                fname.endswith('.npy') or \
                fname == 'implicit_metadata.json' or \
                fname == 'mapping.json'):
                    io.remove_file(f'{self.ds_path}/{fname}')
