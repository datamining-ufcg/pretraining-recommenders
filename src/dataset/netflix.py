from datetime import datetime

import pandas as pd
from tqdm import tqdm

from surprise import Reader

from src.dataset.dataset import Dataset


class Netflix(Dataset):
    def __init__(self, item_prediction=False, **kwargs):
        self.name = "Netflix"
        super(Netflix, self).__init__(
            self.name, item_prediction=item_prediction, **kwargs)

    @property
    def ds_path(self):
        return f'{self.data_path}/netflix'

    @property
    def sep(self):
        return ','

    @property
    def reader(self):
        return Reader(line_format='user item rating timestamp', sep=self.sep)

    @property
    def ratings_file(self):
        return 'netflix.csv'

    @property
    def ratings_intermediate_file(self):
        return 'netflix_complete.csv'

    @property
    def items_file(self):
        return 'netflix_titles.tsv'

    @property
    def items_intermediate_file(self):
        return 'netflix_titles_complete.tsv'

    def load_item_description(self, path=None, read_params=None, adjust_cols=True):
        self.check_and_preprocess_items()
        cols = ['movieId', 'year', 'title']
        df = pd.read_csv(
            f'{self.ds_path}/{self.items_file}', sep='\t', header=None,
            names=cols, engine='python'
        )
        df['year'] = df['year'].fillna(0)
        df['year'] = df['year'].astype('int')

        present_movies = [
            int(self.trainset.to_raw_iid(iid)) for iid in self.trainset.all_items()
        ]
        df = df[df['movieId'].isin(present_movies)]

        self.item_data = df

        return self.item_data

    def _preprocess_file(self, n_split):
        result = []
        with open(f'{self.ds_path}/combined_data_{n_split}.txt', 'r') as data_file:
            for line in tqdm(list(data_file)):
                split = line.split(',')
                if len(split) == 1:
                    movieId = int(split[0][:-2])
                else:
                    userId, rating, date = split
                    date_iso = datetime.strptime(
                        date[:-1], "%Y-%m-%d").isoformat()
                    result.append(f'{userId},{movieId},{rating},{date_iso}\n')

        return result

    def preprocess(self):
        for i in range(1, 5):
            if (i == 1):
                continue

            result = self._preprocess_file(i)

            with open(f'{self.ds_path}/{self.ratings_intermediate_file}', 'a') as nff:
                for l in result:
                    nff.write(l)

            break

        self.preprocess_min_ratings()

    def preprocess_min_ratings(self):
        cols = ['user', 'item', 'rating', 'timestamp']
        df = pd.read_csv(f'{self.ds_path}/{self.ratings_intermediate_file}',
                         engine='python', header=None, names=cols)

        gpu = df.groupby('user').count()
        gpu = gpu[gpu.item >= 20].reset_index()
        users = gpu['user'].tolist()

        gpi = df.groupby('item').count().reset_index()
        # gpi = gpi[gpi.user >= 20].reset_index()
        items = gpi['item'].tolist()

        core = df[(df['user'].isin(users)) & (
            df['item'].isin(items))].reset_index(drop=True)
        core.to_csv(f'{self.ds_path}/{self.ratings_file}',
                    index=False, header=False)

    def preprocess_items_min_ratings(self):
        item_ids = set()
        with open(f'{self.ds_path}/{self.ratings_file}', 'r') as csf:
            lines = csf.readlines()
            for i, line in tqdm(enumerate(lines), total=len(lines)):
                if (i == 0):
                    continue

                splitted = line.split(',')
                item_ids.add(int(splitted[0]))
                line = csf.readline()

        item_ids = list(item_ids)
        items_cols = ['movieId', 'year', 'title']
        items = pd.read_csv(f'{self.ds_path}/{self.items_intermediate_file}',
                            header=None, names=items_cols, sep='\t', engine='python')
        core_items = items[items['movieId'].isin(
            item_ids)].reset_index(drop=True)
        core_items['year'] = core_items['year'].astype('int')
        core_items.to_csv(f'{self.ds_path}/{self.items_file}',
                          sep='\t', index=False, header=False)

    def check_preprocessed_items(self):
        return self.check_file(self.items_file)

    def preprocess_items(self):
        with open(f'{self.ds_path}/movie_titles.csv', 'r', encoding='latin1') as mv:
            for line in tqdm(mv.readlines()):
                l2 = line.replace(',', '\t', 2)
                with open(f'{self.ds_path}/{self.items_intermediate_file}', 'a') as mv2:
                    mv2.write(l2)

        self.preprocess_items_min_ratings()

    def to_implicit(self, files, is_test=False, **kwargs):
        pass
