import pandas as pd

from surprise import Reader
from surprise.dataset import Dataset as SurpriseDataset
from surprise.model_selection import KFold

from src.embeddings import EmbeddingGenerator
from src.config import RANDOM_STATE
import src.io_util as io


class RawDataset(object):
    def __init__(self, name, **kwargs) -> None:
        self.name = name
        self.STATE = RANDOM_STATE

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
    def is_loaded(self):
        return hasattr(self, 'data')

    @classmethod
    def build_testset(cls, trainset, testset):
        raw_users = set()
        raw_items = set()

        for u in trainset.all_users():
            raw_users.add(trainset.to_raw_uid(u))

        for i in trainset.all_items():
            raw_items.add(trainset.to_raw_iid(i))

        testset_mapped_ids = []
        for u, i, r in testset:
            if (u in raw_users) and (i in raw_items):
                testset_mapped_ids.append((
                    trainset.to_inner_uid(
                        u), trainset.to_inner_iid(i), float(r)
                ))

        return testset_mapped_ids

    def load_from_triples(self, triples):
        df = pd.DataFrame(triples, columns=['userId', 'itemId', 'ratings'])
        self.load_from_df(df)

    def load_from_df(self, df, scale=(1, 5)):
        sds = SurpriseDataset.load_from_df(df, Reader(rating_scale=scale))
        self.data = sds

    def sparsity(self):
        return 1 - (self.n_ratings / (self.n_items * self.n_users))

    def kfold(self, k=5):
        """
        Generates k random splits of `self.data`.
        """
        kfold = KFold(k, random_state=self.STATE)
        return kfold.split(self.data)

    def get_fold(self, n_folds=5, k=1):
        self.current_fold = k
        i = 1
        it = self.kfold(n_folds)

        while i <= k:
            trainset, testset = next(it)
            i += 1

        testset = self.build_testset(trainset, testset)
        self.trainset = trainset
        self.testset = testset

        return trainset, testset

    def get_embeddings(self, embedding_dim, method='random', user=False, **kwargs):
        return EmbeddingGenerator.get_embeddings(
            self.trainset, embedding_dim, method, user, **kwargs
        )

    def batch(self, xs, num_batches):
        batches = [[] for _ in range(num_batches)]
        for i, x in enumerate(xs):
            batches[i % num_batches].append(x)
        
        return batches
    
    def check_file(self, fname, filepath):
        """
        Checks if a file exists in the given filepath.
        """
        return io.check_file(fname, filepath)