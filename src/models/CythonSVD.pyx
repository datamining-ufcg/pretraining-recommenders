# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
cimport numpy as np  # noqa
import numpy as np
import logging

from src.models.CythonModel import CythonModel


class CythonSVD(CythonModel):
    """
    A matrix factorization model trained using SGD and negative sampling.
    """

    def __init__(
        self,
        dataset,
        embedding_dim,
        method,
        method_args,
        reg,
        cold_start=True,
        model_path='',
        skip_folder_creation=False
    ):
        """
        Initializes CythonSVD Model.
        """
        super(CythonSVD, self).__init__(
            dataset,
            embedding_dim,
            method,
            method_args,
            reg,
            cold_start,
            model_path,
            skip_folder_creation
        )

    def predict(self, u, i):
        r_hat = self.pu[u] @ self.qi[i]
        r_hat += self.global_bias + self.bu[u] + self.bi[i]
        return r_hat

    def _predict(self, u, i, global_bias, bu, bi, pu, qi):
        pred = (global_bias + bu[u] + bi[i] + np.dot(pu[u], qi[i]))
        return pred

    def fit(self, trainset, testset, epochs, learning_rate, patience, verbose=True, early_stopping=True):
        self.epochs = epochs
        self.lr = learning_rate
        self.patience = patience

        cdef np.ndarray[np.double_t] bu
        # item biases
        cdef np.ndarray[np.double_t] bi
        # user factors
        cdef np.ndarray[np.double_t, ndim = 2] pu
        # item factors
        cdef np.ndarray[np.double_t, ndim = 2] qi

        cdef int u, i, f
        cdef double r, err, dot, puf, qif, mse
        cdef double global_bias = self.global_bias

        cdef double lr = learning_rate
        cdef double reg = self.reg
        cdef double best_rmse = float('inf')
        cdef int current_patience = patience

        if (hasattr(self, 'last_epoch')):
            pu = np.array(self.pu, dtype='float')
            bu = np.array(self.bu, dtype=np.double)
            qi = np.array(self.qi, dtype='float')
            bi = np.array(self.bi, dtype=np.double)
            epoch_start = self.last_epoch + 1

        else:
            bu = np.zeros(trainset.n_users, np.double)
            bi = np.zeros(trainset.n_items, np.double)
            pu = np.array(self.pu, dtype='float')
            qi = np.array(self.qi, dtype='float')
            epoch_start = 0

        for current_epoch in range(epoch_start, epochs):
            mse = 0
            for u, i, r in trainset.all_ratings():
                r_hat = self._predict(u, i, global_bias, bu, bi, pu, qi)
                err = r - r_hat

                # update biases
                bu[u] += lr * (err - reg * bu[u])
                bi[i] += lr * (err - reg * bi[i])

                # update factors
                for f in range(self.n_factors):
                    puf = pu[u, f]
                    qif = qi[i, f]
                    pu[u, f] += lr * (err * qif - reg * puf)
                    qi[i, f] += lr * (err * puf - reg * qif)

                r_hat = self._predict(u, i, global_bias, bu, bi, pu, qi)
                mse += np.square(np.subtract(r, r_hat))

            train_rmse = np.sqrt(mse / trainset.n_ratings)

            test_predictions = [
                (r, self._predict(u, i, global_bias, bu, bi, pu, qi))
                for (u, i, r) in testset
            ]
            test_rmse = self.rmse(test_predictions)

            self.results['loss'].append(train_rmse)
            self.results['val_loss'].append(test_rmse)

            if (early_stopping):
                if (test_rmse < best_rmse):
                    best_rmse = test_rmse
                    current_patience = patience

                else:
                    current_patience -= 1

            if (current_epoch % 10 == 0):
                self.last_epoch = current_epoch
                self.set_params(pu, bu, qi, bi)
                self.save(is_complete=False)
                logging.debug(f'Saved at epoch {current_epoch}.')

            if (verbose or current_patience < 0 or current_epoch % 10 == 0):
                logging.debug(
                    f'Epoch {current_epoch + 1}. Train: {round(train_rmse, 5)} - Test: {round(test_rmse, 5)}.')

            if (current_patience < 0 and early_stopping):
                self.last_epoch = current_epoch
                logging.debug(
                    f'Method: {self.method}. Split: {self.dataset_split}. Limit: {self.limit}.')
                break

        self.last_epoch = epochs
        self.set_params(pu, bu, qi, bi)

        return self.results

    def predictions(self, ratings):
        pred = [(r, self.predict(u, i)) for (u, i, r) in ratings]
        return pred

    def rmse(self, predictions):
        rmse = np.sqrt(np.mean([np.square(true_r - est)
                       for (true_r, est) in predictions]))
        return rmse

    def set_params(self, pu, bu, qi, bi):
        self.pu = pu
        self.bu = bu
        self.qi = qi
        self.bi = bi

    def _base_config(self, is_complete):
        """
        Gets base config and adds best results for metrics.
        """
        config = super()._base_config(is_complete)
        config['rmse'] = self.results['val_loss'][-1]
        return config

    def _save_embeddings(self, path):
        """
        Saves model embeddings to .npy files.
        """
        np.save(f'{path}/pu', self.pu)
        np.save(f'{path}/bu', self.bu)
        np.save(f'{path}/qi', self.qi)
        np.save(f'{path}/bi', self.bi)

    def load_embeddings(self, path):
        """
        Loads embeddings from .npy files inside the given path.
        """
        self.pu = np.load(f'{path}/pu.npy')
        self.bu = np.load(f'{path}/bu.npy')
        self.qi = np.load(f'{path}/qi.npy')
        self.bi = np.load(f'{path}/bi.npy')

    @staticmethod
    def load_model(path):
        config = CythonSVD._load_config(path)

        model = CythonSVD(
            None, config['embedding_size'], config['method'],
            config['method_args'], config['reg_factor'], cold_start=False,
            model_path=path, skip_folder_creation=True
        )

        model.load_attributes(config)
        model.load_embeddings(path)

        return model
