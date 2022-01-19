import json
import time
import os

import numpy as np
import pandas as pd
from joblib import load, dump
from tqdm import tqdm
import logging

from sklearn.model_selection import ParameterGrid

from models.ease import EASE
from models.primitive_models import PrimitiveModels
from models.knn_popular import KNNpopularity
from models.mf import MatrixFactorization

from utils.metrics import serendipity
from utils.helpers import get_movies_by_ids, get_control_items, get_movies_by_profile
from utils.parser import parse_args
from utils.logging import init_logging

import warnings

np.random.seed(42)
warnings.filterwarnings("ignore", category=RuntimeWarning)

models = {
    'KNNpopularity': KNNpopularity,
    'MatrixFactorization': MatrixFactorization,
    'EASE': EASE,
}

args = parse_args()

init_logging('gs.log', path=args.save_path)

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


primitive_models = PrimitiveModels(train_data, ratings, item_ratings)
n = 10  # number of recommendations for each user

primitive_recommendations = None

if not os.path.isfile('cache/primitive_recommendations.joblib'):
    primitive_models = PrimitiveModels(train_data, ease_ratings, item_ratings, control_items)
    primitive_recommendations = primitive_models.make_recommendations(test_data, n)
    dump(primitive_recommendations, 'cache/primitive_recommendations.joblib')
else:
    primitive_recommendations = load('cache/primitive_recommendations.joblib')


def grid_search(model_name, grid, data):
    history = []

    for params in ParameterGrid(grid):
        if model_name == 'EASE':
            model = models[model_name](ease_ratings, params['l'], n)
        else:
            model = models[model_name](model_name, data, item_ratings)
            model.fit(params)

        start_time = time.time()
        logging.debug(f'Started {model_name}, {params}')

        if model_name == 'EASE':
            recall, coverage, nov, unexp, rel = model.evaluate(
                ease_test_data,
                params,
                ease_control_items,
                movies,
                user_embeddings,
                primitive_recommendations
            )
        else:
            recall, coverage, nov, unexp, rel = model.evaluate(
                test_data,
                params,
                control_items,
                user_embeddings,
                primitive_recommendations,
                movies,
                n
            )

        logging.debug(f'Recall: {recall},'
                      f'coverage: {coverage},'
                      f'novelty: {np.mean(nov)},'
                      f'unexpectedness: {np.mean(unexp)},'
                      f'relevance: {np.mean(rel)}')
        logging.debug(f'Finished {model_name}, {params}')
        time_elapsed = time.time() - start_time
        history.append((params, recall, coverage, nov, unexp, rel, time_elapsed))

    return history


params_grid = {
    'KNNpopularity': {'k': [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 300, 500, 700, 1000]},
    'MatrixFactorization': {'n_components': [5, 10, 20, 50, 100, 200, 500, 1000, 2000], 'random_state': [42]},
    'EASE': {'l': [1.0, 10.0, 50.0, 100.0, 250.0, 500.0, 750.0, 1000.0, 2000.0, 5000.0, 10000.0]}
}

for model_name in models.keys():
    logging.info('Grid search started: {}'.format(model_name))
    result = grid_search(model_name, params_grid[model_name], train_data)
    logging.info('Grid search ended: {}'.format(model_name))
    logging.info('Saved {} as {}'.format(model_name, os.path.join(args.save_path, model_name + '.joblib')))
    dump(result, os.path.join(args.save_path, model_name + '.joblib'))
