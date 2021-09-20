import os
import numpy as np
import pandas as pd
from joblib import load, dump
import logging
from collections import defaultdict

from pymoo.model.problem import Problem
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
from pymoo.model.callback import Callback

from models.simple_knn import KNN
from models.knn_popular import KNNpopularity
from models.mf import MatrixFactorization
from models.autorec import AutoRec

from utils.metrics import serendipity
from utils.parser import parse_args
from utils.logging import init_logging

import warnings
import tensorflow as tf

warnings.filterwarnings("ignore", category=RuntimeWarning)

args = parse_args()
logger = logging.getLogger(__name__)

X_train, X_test_prepared, beta, ratings_cleaned_df, items_ratings_df, movies_extended_df, primitive_model = [None] * 7


class MyProblem(Problem):
    def __init__(self, **kwargs):
        self.n_var = 3
        self.n_obj = kwargs['n_obj']
        assert 1 <= self.n_obj <= 3, "n_obj must be equal to 1, 2 or 3"
        self.xl = [1, 1, 1]
        self.xu = [80, 100, 64]
        super().__init__(
            n_var=self.n_var,
            n_obj=self.n_obj,
            n_constr=0,
            xl=self.xl,
            xu=self.xu
        )

    def _evaluate(self, X, out, *args, **kwargs):
        ser, rec, cov = [], [], []
        for i in range(len(X)):
            tmp = X[i, :]
            params = {
                'K': tmp[0],
                'n_components': tmp[1],
                'hide_layer': tmp[2] * 8,
                'random_state': 42,
                'corr': 1
            }
            serendipity_, recall, coverage = loss_ensemble(params)
            ser.append(serendipity_)
            rec.append(recall)
            cov.append(coverage)
            logger.debug('{}: {}/{}/{}'.format(params, serendipity_, recall, coverage))

        if self.n_obj == 1:
            out['F'] = -np.array(ser)
        elif self.n_obj == 2:
            out['F'] = -np.column_stack([ser, rec])
        elif self.n_obj == 3:
            out['F'] = -np.column_stack([ser, rec, cov])


class MyCallback(Callback):
    def __init__(self, i=1, path='') -> None:
        super().__init__()
        self.i = i
        self.path = path

    def notify(self, algorithm, **kwargs):
        np.save(os.path.join(self.path, 'checkpoint' + str(self.i) + '.joblib'), algorithm)
        self.i += 1


def gpu_fix():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def loss_ensemble(params):
    logger.debug('Calulating loss for params: {}'.format(params))
    knn, mf, autorec = prepare_models(params)
    logger.debug('Preparing KNN recommendations')
    knn_recommendations, knn_primitive_recommendations, knn_recall, knn_coverage = recommend(knn)
    logger.debug('Preparing MF recommendations')
    mf_recommendations, mf_primitive_recommendations, mf_recall, mf_coverage = recommend(mf)
    logger.debug('Preparing AutoRec recommendations')
    autorec_recommendations, autorec_primitive_recommendations, autorec_recall, autorec_coverage = recommend(autorec)
    recommendations = merge_dict(knn_recommendations, mf_recommendations, autorec_recommendations)
    primitive_recommendations = merge_dict(knn_primitive_recommendations, mf_primitive_recommendations,
                                           autorec_primitive_recommendations)

    # logger.debug('Calculating serendipity')
    # serendipity_, indices = get_serendipity(recommendations, primitive_recommendations)
    total_recommendations = 0
    all_recommendations = []
    recommedation_serendipity = []

    hit = 0
    for user_id, user_profile in X_test_prepared.iterrows():
        if user_id not in recommendations.keys():
            continue
        # subset_ind = indices[user_id]
        tmp = np.array(recommendations[user_id])
        np.random.shuffle(tmp)
        recommendation = list(tmp[:10])
        # ser_recommendation = np.array(recommendations[user_id])[subset_ind][:3]
        # recommedation_serendipity.extend(serendipity_[subset_ind][:3])

        # recommendation.extend(list(ser_recommendation))
        recommendation = np.unique(recommendation)

        recommended_items = movies_extended_df[movies_extended_df.index.isin(recommendation)].values

        user = movies_extended_df[movies_extended_df.index.isin(
            X_test_prepared[X_test_prepared.index == user_id]['user_keep_movies'].values[0]
        )].values
        if len(user) == 0 or len(recommendations[user_id]) == 0:
            continue
        ser_ = serendipity(recommended_items, recommendations[user_id],
                           primitive_recommendations[user_id], user,
                           gamma=0.3, alpha=0.4, beta=0.3)

        if user_profile[2] in recommendation:  # if prediction contains control item increase hit counter
            hit += 1

        recommedation_serendipity.append(ser_)
        all_recommendations.extend(list(recommendation))
        total_recommendations += 1

    ser = np.mean(recommedation_serendipity)
    recall = hit / total_recommendations
    coverage = np.unique(all_recommendations).shape[0] / X_train.shape[0]
    return ser, recall, coverage


def get_serendipity(recommendations, primitive_recommendations):
    p = 10
    serendipity_results = []
    indices = {}
    for user_id in X_test_prepared.index:
        if user_id in recommendations:
            tmp = recommendations[user_id]
            recommended_items = movies_extended_df[movies_extended_df.index.isin(tmp)].values
            user_profile = movies_extended_df[movies_extended_df.index.isin(
                X_test_prepared[X_test_prepared.index == user_id]['user_keep_movies'].values[0]
            )].values
            if len(user_profile) == 0 or len(recommendations[user_id]) == 0:
                continue

            recommendation_serendipity = serendipity(recommended_items, recommendations[user_id],
                                                     primitive_recommendations[user_id], user_profile,
                                                     gamma=0.3, alpha=0.4, beta=0.3, keepdims=True)
            ind = np.argsort(recommendation_serendipity)[::-1][:p]
            serendipity_results.append(np.sum(recommendation_serendipity[ind]))
            indices[user_id] = ind

    return np.array(serendipity_results), indices


def recommend(model):
    """
    Generate n recommendations.

    :param model: recommendation model
    :return: tuple(recommendations made by model, recommendations made by primitive model, recall, coverage)
    """
    n = 10
    hit = 0  # used for recall calculation
    total_recommendations = 0
    all_recommendations = []  # used for coverage calculation
    recommendations = {}
    primitive_recommendations = {}

    for user_id, user_profile in X_test_prepared.iterrows():  # iterate over test users, user_profile is a tuple
        prediction = model.predictItemByUser(user_profile[1], user_profile[0], n)
        primitive_prediction = primitive_model.predictItemByUser(None, user_profile[0], n, ratings_cleaned_df)
        # primitive_predictions = primitive_model.test()
        if prediction is None or prediction.ndim == 0:
            continue
        if user_profile[2] in prediction:  # if prediction contains control item increase hit counter
            hit += 1
        recommendations[user_id] = prediction
        primitive_recommendations[user_id] = primitive_prediction
        all_recommendations.extend(list(prediction))
        total_recommendations += 1
    if total_recommendations > 0:
        recall = hit / total_recommendations
    else:
        recall = 0
    coverage = np.unique(all_recommendations).shape[0] / model.train_data.shape[1]
    return recommendations, primitive_recommendations, recall, coverage


def prepare_models(params):
    knn = KNNpopularity('userBasedKNNpopularity', 'config/config.ini', X_train, 0, items_ratings_df)
    mf = MatrixFactorization('userBasedMF', 'config/config.ini', X_train, items_ratings_df)
    autorec = AutoRec('userBasedAutoRec', 'config/config.ini', X_train, items_ratings_df)
    preprocess_fit(knn, params)
    preprocess_fit(mf, params)
    preprocess_fit(autorec, params['hide_layer'])
    return knn, mf, autorec


def preprocess_fit(model, params):
    model.preprocess()
    model.fit(params)


def merge_dict(*args):
    result = defaultdict(list)
    for d in args:
        for key, value in d.items():
            result[key] += list(value)
            result[key] = list(set(result[key]))
    return result


def train_ensemble(init_params=None):
    gpu_fix()
    global X_train, X_test_prepared, beta, ratings_cleaned_df, items_ratings_df, movies_extended_df, primitive_model

    if not os.path.exists(args.save_path):  # create save directory if it doesn't exist
        os.makedirs(args.save_path)

    init_logging('3models.log', path=args.save_path)

    if args.algorithm == 'ga':
        assert args.n_obj == 1, "Number of objectives must be 1 for GA"

    dataset = '1m'
    data_folder = 'data/movielens/' + dataset + '/clean/'
    X_train = load(data_folder + '/train_data.pickle')
    X_test = load(data_folder + '/test_data.pickle')
    X_test_prepared = load(data_folder + '/test_data_prepare_for_recom.pickle')

    X_test_items = []  # calculation of beta for KNN
    for k, v in X_test.iterrows():
        f = v[v > 0.0]
        X_test_items = np.concatenate((X_test_items, np.array(f.index)), axis=0)

    alfa = len(np.unique(X_test_items))
    gamma = len(X_test.columns)
    beta = gamma / alfa

    logger.info('Dataset: ' + dataset + ' (MovieLens)')
    logger.info('Algorithm: {}, n_obj: {}, n_gen: {}, pop_size: {}'.format(
        args.algorithm, args.n_obj, args.n_gen, args.pop_size)
    )

    ratings_cleaned_df = pd.read_csv(data_folder + 'ratings.csv')
    ratings_cleaned_df.columns = ['userId', 'movieId', 'rating', 'timestamp']
    ratings_cleaned_df = ratings_cleaned_df.sort_values(['userId', 'timestamp'])

    items_ratings_df = load(data_folder + 'item_sum_dif_rating.pickle')
    print(os.listdir('.'))
    movies_extended_df = load('t_film_profile_sem_0_and_com_001.pickle')

    primitive_model = KNN('userBasedKNN', 'config/config.ini', X_train)
    primitive_model.preprocess()
    primitive_model.fit({'K': 3})

    kwargs = {
        'n_obj': args.n_obj
    }

    problem = MyProblem(**kwargs)

    mask = ['int', 'int', 'int']
    if init_params is None:
        sampling = MixedVariableSampling(mask, {'int': get_sampling('int_random')})
    else:
        k_sampling = np.random.randint(problem.xl[0], problem.xu[0], size=(6, 1))
        mf_sampling = np.random.randint(problem.xl[1], problem.xu[1], size=(6, 1))
        autorec_sampling = np.random.randint(problem.xl[2], problem.xu[2], size=(6, 1))
        random_sampling = np.hstack((k_sampling, mf_sampling, autorec_sampling))
        sampling = np.vstack((np.array(init_params).T, random_sampling))
        logger.debug('Using sampling from meta-storage: {}'.format(sampling))

    crossover = MixedVariableCrossover(mask, {'int': get_crossover('int_sbx', prob=1.0, eta=3.0)})
    mutation = MixedVariableMutation(mask, {'int': get_mutation('int_pm', eta=3.0)})

    if args.checkpoint_path is None:
        if args.algorithm == 'nsga2':
            algorithm = NSGA2(
                pop_size=args.pop_size,
                sampling=sampling,
                crossover=crossover,
                mutation=mutation,
                eliminate_duplicates=True
            )
        elif args.algorithm == 'ga':
            algorithm = GA(
                pop_size=args.pop_size,
                sampling=sampling,
                crossover=crossover,
                mutation=mutation,
                eliminate_duplicates=True
            )
        else:
            raise Exception('Unknown algorithm')
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
    return res


if __name__ == '__main__':
    res = train_ensemble()
