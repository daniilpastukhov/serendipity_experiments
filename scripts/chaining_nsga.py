import json
import time
import os

import joblib
import numpy as np
import pandas as pd
from joblib import load, dump
from tqdm import tqdm
import logging

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
from pymoo.core.callback import Callback

from models.ease import EASE
from models.primitive_models import PrimitiveModels
from models.knn_popular import KNNpopularity
from models.mf import MatrixFactorization
from models.chaining import Chaining

from utils.metrics import serendipity, unexpectedness, relevance
from utils.helpers import get_movies_by_ids, get_control_items, get_movies_by_profile
from utils.parser import parse_args
from utils.logging import init_logging

import warnings

np.random.seed(42)
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MyProblem(Problem):
    def __init__(self, **kwargs):
        self.n_var = 3
        self.n_obj = kwargs['n_obj']
        assert 1 <= self.n_obj <= 3, 'n_obj must be equal to 1, 2 or 3'
        self.xl = [1, 10, 500]
        self.xu = [150, 1000, 5000]
        super().__init__(
            n_var=self.n_var,
            n_obj=self.n_obj,
            n_constr=0,
            xl=self.xl,
            xu=self.xu
        )

    def _evaluate(self, X, out, *args, **kwargs):
        ser, rec = [], []
        for i in range(len(X)):
            tmp = X[i, :]
            params = {
                'k': tmp[0],
                'n_components': tmp[1],
                'l': tmp[2]
            }
            chaining = Chaining(train_data, ease_ratings, item_ratings, ease_control_items, **params)
            res = chaining.evaluate(test_data, [{'k': params['k']}, {'n_components': params['n_components']}, {'l': params['l']}],
                                    control_items,
                                    user_embeddings, primitive_recommendations, movies)
            ser_ = (np.mean(res[2]) + np.mean(res[3]) + np.mean(res[4])) / 3
            ser.append(ser_)
            rec.append(res[0])
            logging.debug(f'Params: {params}'
                          f'recall: {res[0]},'
                          f'coverage: {res[1]},'
                          f'serednipity: {(np.mean(res[2]) + np.mean(res[3]) + np.mean(res[4])) / 3}',
                          f'novelty: {np.mean(res[2])},'
                          f'unexpectedness: {np.mean(res[3])},'
                          f'relevance: {np.mean(res[4])}')

        if self.n_obj == 1:
            out['F'] = -np.array(ser)
        elif self.n_obj == 2:
            out['F'] = -np.column_stack([ser, rec])


class MyCallback(Callback):
    def __init__(self, i=1, path='') -> None:
        super().__init__()
        self.i = i
        self.path = path

    def notify(self, alg, **kwargs):
        np.save(os.path.join(self.path, 'checkpoint' + str(self.i) + '.joblib'), alg)
        self.i += 1

if __name__ == '__main__':
    args = parse_args()

    init_logging('chaining_gs.log', path=args.save_path)

    # Datasets
    DATASET = '1m'
    DATA_PATH = 'data/movielens/' + DATASET + '/clean/'

    ratings = pd.read_csv(DATA_PATH + 'ratings.csv')
    movies = pd.read_csv(os.path.join(DATA_PATH, 'movies.csv'))
    item_ratings = load(DATA_PATH + 'item_sum_dif_rating.pickle')

    train_data = pd.read_csv(DATA_PATH + 'train_data.csv', index_col='userId')
    test_df = pd.read_csv(DATA_PATH + 'test_data.csv', index_col='userId')
    test_data, control_items = get_control_items(ratings, test_df)

    ease_ratings = pd.read_csv(DATA_PATH + 'ratings.csv')
    ease_test_df = pd.read_csv(DATA_PATH + 'test_data.csv', index_col='userId')
    ease_test_data, _ = get_control_items(ease_ratings, user_profiles=ease_test_df)
    ease_ratings, ease_control_items = get_control_items(ease_ratings, user_ids=ease_test_df.index.values)

    user_embeddings = {}
    for user_id, user_profile in test_data.iterrows():
        user_embeddings[user_id] = get_movies_by_profile(movies, user_profile)

    logging.info('Dataset: ' + DATASET)

    n = 10  # number of recommendations for each user

    primitive_recommendations = None

    if not os.path.isfile('cache/primitive_recommendations.joblib'):
        primitive_models = PrimitiveModels(train_data, ease_ratings, item_ratings, ease_control_items)
        primitive_recommendations = primitive_models.make_recommendations(test_data, n)
        dump(primitive_recommendations, 'cache/primitive_recommendations.joblib')
    else:
        primitive_recommendations = load('cache/primitive_recommendations.joblib')

    kwargs = {
        'n_obj': args.n_obj
    }

    problem = MyProblem(**kwargs)

    mask = ['int', 'int', 'int']
    sampling = MixedVariableSampling(mask, {'int': get_sampling('int_random')})
    crossover = MixedVariableCrossover(mask, {'int': get_crossover('int_sbx', prob=1.0, eta=3.0)})
    mutation = MixedVariableMutation(mask, {'int': get_mutation('int_pm', eta=3.0)})

    if args.checkpoint_path is None:
        algorithm = NSGA2(
            pop_size=10,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=True
        )
    else:
        algorithm, = np.load(args.checkpoint_path, allow_pickle=True).flatten()
        algorithm.has_terminated = False

    res = minimize(
        problem,
        algorithm,
        ('n_gen', args.n_gen),
        seed=42,
        verbose=True,
        save_history=True,
        callback=MyCallback(args.i, path=args.save_path)
    )

    dump(algorithm, os.path.join(args.save_path, 'algorithm.joblib'))
    dump(res, os.path.join(args.save_path, 'res.joblib'))
