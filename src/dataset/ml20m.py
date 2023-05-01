from surprise import Reader
from src.dataset.movielens import ExplicitMovielens, ImplicitMovielens, Movielens


class ML20M(Movielens):
    """
    Implementation of ML20M with Surprise.
    """

    @property
    def ds_path(self):
        return f'{self.data_path}/ml20m'

    @property
    def sep(self):
        return ','

    @property
    def reader(self):
        rdr = Reader(
            line_format='user item rating timestamp',
            sep=self.sep, rating_scale=(1, 5)
        )
        return rdr

    @property
    def ratings_file(self):
        return 'ratings.csv'

    @property
    def items_file(self):
        return 'movies.csv'

    @property
    def skip_rows(self):
        return 1

    def load_item_description(self):
        self.check_and_preprocess_items()
        path = f'{self.ds_path}/{self.items_file}'
        return super().load_item_description(path, {}, adjust_cols=False)


class ExplicitML20M(ML20M, ExplicitMovielens):
    """

    """

    def __init__(self, **kwargs) -> None:
        self.name = 'ExplicitML20M'
        super(ExplicitML20M, self).__init__(self.name, **kwargs)


class ImplicitML20M(ML20M, ImplicitMovielens):
    """

    """

    def __init__(self, **kwargs) -> None:
        self.name = 'ImplicitML20M'
        super(ExplicitML20M, self).__init__(self.name, **kwargs)
