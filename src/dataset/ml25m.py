from surprise import Reader
from src.dataset.movielens import ExplicitMovielens, ImplicitMovielens, Movielens


class ML25M(Movielens):
    """
    Implementation of ML25M with Surprise.
    """

    @property
    def sep(self):
        return ','

    @property
    def reader(self):
        rdr = Reader(
            line_format='user item rating timestamp',
            sep=self.sep, skip_lines=1, rating_scale=(1, 5)
        )
        return rdr

    @property
    def ds_path(self):
        return f'{self.data_path}/ml25m'

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


class ExplicitML25M(ML25M, ExplicitMovielens):
    """

    """

    def __init__(self, **kwargs) -> None:
        self.name = 'ExplicitML25M'
        super(ExplicitML25M, self).__init__(self.name, **kwargs)


class ImplicitML25M(ML25M, ImplicitMovielens):
    """

    """

    def __init__(self, **kwargs) -> None:
        self.name = 'ImplicitML25M'
        super(ImplicitML25M, self).__init__(self.name, **kwargs)
