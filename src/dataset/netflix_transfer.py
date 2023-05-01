from collections import defaultdict
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from surprise import Reader

from src.dataset.explicit_feedback import ExplicitFeedback
from src.dataset.implicit_feedback import ImplicitFeedback


class NetflixTransfer():
    def __init__(self, ratings_filename: str = 'netflix.csv', **kwargs):
        self.ratings_filename = ratings_filename
        self.current_fold = 1
        super().__init__(**kwargs)

    @property
    def ratings_intermediate_file(self):
        return 'netflix_complete.csv'

    @property
    def items_intermediate_file(self):
        return 'netflix_titles_complete.tsv'

    @property
    def ds_path(self):
        return f'{self.data_path}/netflix_transfer'

    @property
    def sep(self):
        return ','

    @property
    def reader(self):
        return Reader(line_format='user item rating timestamp', rating_scale=(1, 5), sep=self.sep)

    @property
    def ratings_file(self):
        if (self.ratings_filename.endswith('.csv')):
            return self.ratings_filename
        return f'{self.ratings_filename}.csv'

    @property
    def items_file(self):
        return 'netflix_titles.tsv'

    @property
    def mapped_ratings(self):
        return 'netflix_mapped.csv'

    def load_item_description(self, path=None, read_params=None, adjust_cols=True):
        self.check_and_preprocess_items()
        cols = ['movieId', 'year', 'title']
        df = pd.read_csv(
            f'{self.ds_path}/{self.items_file}', sep='\t', header=None,
            names=cols, engine='python'
        )
        df['year'] = df['year'].fillna(0)
        df['year'] = df['year'].astype('int')

        self.item_data = df

        return self.item_data

    def load_as_dataframe(self):
        cols = ['userId', 'movieId', 'rating', 'timestamp']
        eng = 'python' if len(self.sep) > 1 else None
        df = pd.read_csv(
            self.filepath, header=None, names=cols,
            sep=self.sep, engine=eng
        )
        return df

    def _preprocess_file(self, n_split):
        result = []
        with open(f'{self.ds_path}/combined_data_{n_split}.txt', 'r') as data_file:
            for line in tqdm(data_file):
                split = line.split(',')
                if len(split) == 1:
                    movieId = int(split[0][:-2])
                else:
                    userId, rating, date = split
                    date_iso = datetime.strptime(
                        date[:-1], "%Y-%m-%d").isoformat()
                    result.append(f'{userId},{movieId},{rating},{date_iso}\n')

        return result

    def preprocess(self, use_pandas=False):
        for i in range(1, 5):
            result = self._preprocess_file(i)

            with open(f'{self.ds_path}/{self.ratings_intermediate_file}', 'a') as nff:
                for l in result:
                    nff.write(l)

        self.preprocess_min_ratings()

    def preprocess_min_ratings(self, use_pandas=False):
        if (use_pandas):
            self._preprocess_min_ratings_pandas()

        else:
            self._preprocess_min_ratings_base()

    def _preprocess_min_ratings_pandas(self):
        cols = ['user', 'item', 'rating', 'timestamp']
        df = pd.read_csv(
            f'{self.ds_path}/{self.ratings_intermediate_file}', header=None, names=cols)

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

    def _preprocess_min_ratings_base(self):
        counts = defaultdict(int)
        with open(f'{self.ds_path}/{self.ratings_intermediate_file}', 'r') as rifl:
            for line in rifl:
                iid = int(line.split(',')[1])
                counts[iid] += 1

        useable = {k: v for k, v in counts.items() if v >= 20}
        with open(f'{self.ds_path}/{self.ratings_intermediate_file}', 'r') as rifl:
            with open(f'{self.ds_path}/{self.ratings_file}', 'a') as rfl:
                for line in rifl:
                    iid = int(line.split(',')[1])
                    if (iid not in useable):
                        continue

                    rfl.write(line)

    def preprocess_items_min_ratings(self):
        item_ids = set()
        with open(f'{self.ds_path}/{self.ratings_file}', 'r') as csf:
            for line in tqdm(csf):
                splitted = line.split(',')
                item_ids.add(int(splitted[1]))
                line = csf.readline()

        item_ids = list(item_ids)
        items_cols = ['movieId', 'year', 'title']
        items = pd.read_csv(f'{self.ds_path}/{self.items_intermediate_file}',
                            header=None, names=items_cols, sep='\t')
        print(items.head())
        core_items = items[items['movieId'].isin(
            item_ids)].reset_index(drop=True)
        core_items['year'] = core_items['year'].fillna(0).astype('int')
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

    def process_mapped_ids(self, mapping: dict, target_name: str):
        print('Netflix - Mapping ids')
        with open(f'{self.ds_path}/{self.ratings_file}', 'r') as rfl:
            with open(f'{self.ds_path}/{target_name}_{self.mapped_ratings}', 'a') as mfl:
                for line in tqdm(rfl):
                    iid = line.split(',')[1]
                    if iid in mapping:
                        mfl.write(line)

    def load_mapped(self):
        return super().load(filepath=f'{self.data_path}/{self.mapped_ratings}')


class ExplicitNetflixTransfer(NetflixTransfer, ExplicitFeedback):
    """
    Netflix superclass for Explicit Feedback.
    """

    def __init__(self, **kwargs) -> None:
        self.name = 'ExplicitNetflixTransfer'
        kwargs['name'] = self.name
        super(ExplicitNetflixTransfer, self).__init__(**kwargs)


class ImplicitNetflixTransfer(NetflixTransfer, ImplicitFeedback):
    """
    Netflix superclass for Implicit Feedback.
    """

    def __init__(self, **kwargs) -> None:
        self.name = 'ImplicitNetflixTransfer'
        kwargs['name'] = self.name
        super(ImplicitNetflixTransfer, self).__init__(**kwargs)
