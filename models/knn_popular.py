# ref: Stanislav Kuznetsov
import configparser
import pandas as pd
# from pandarallel import pandarallel
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


class KNNpopularity:
    def __init__(self, name, configPath, train_data, item_global_rating):
        self.name = name

        config = configparser.ConfigParser()
        config.read(configPath)

        self.folder = config['DATAFILES']['folder']
        self.train_data = train_data
        self.train_data_item_count = len(train_data)

        self.item_global_rating = item_global_rating
        self.beta = 0.0

    def preprocess(self):
        try:
            self.user_movie_mat_sparse = csr_matrix(self.train_data.values)
            self.train_data_by_row_number = self.train_data.T.reset_index()

        except Exception as e:
            print(self.name + ' proprocess ' + str(e))

    def fit(self, params):
        try:
            self.K = params['K']
            if 'beta' in params:
                self.beta = params['beta']
            self.model_knn = NearestNeighbors(metric='cosine', algorithm='auto', n_neighbors=self.K, n_jobs=-1)
            self.model_knn.fit(self.user_movie_mat_sparse)

        except Exception as e:
            print(self.name + ' fit ' + str(e))

    def predict(self, user_profile, n):
        try:
            # konvertuju profil uzivatele na crc matrix
            user_crc_profile = csr_matrix(user_profile)
            # ziskavat indexy jeho filmy
            rows, cals_user = user_crc_profile.nonzero()

            # vytahnu sousedy
            distances, indices = self.model_knn.kneighbors(user_crc_profile)
            # vytahnu profily sousedu
            my_neighbors = indices.squeeze().tolist()
            my_neighbors_distance = distances.squeeze().tolist()

            # fce
            # prumerny rating
            def f_avg_rating(row):
                tmp = self.user_movie_mat_sparse.getrow(row['my_neighbors_index'])
                return tmp.sum() / tmp.count_nonzero()

            # ziskame nadprumerne popularni itemy
            def f_popular_item(row):
                tmp = self.user_movie_mat_sparse.getrow(row['my_neighbors_index'])
                rows, cals = tmp.nonzero()
                return pd.DataFrame(
                    [[i, j, row['my_neighbors_distance'], row['avg_rating']] for (i, j) in zip(cals, tmp.data) if
                     j > row['avg_rating']],
                    columns=['movieIndex', 'rating', 'my_neighbors_distance', 'avg_rating'])

            def f_rating_diff(row):
                tmp = row['rating'] - row['avg_rating']
                return tmp

            # vypocet vazeneho score
            def f_score(row):
                tmp = row['rating_diff'] * (1 - row['my_neighbors_distance'])
                return tmp

            def f_beta_score(row):
                item_glob_popularity = \
                    self.item_global_rating[self.item_global_rating['index'] == row['index']][['sum_rating']].values[0][
                        0]
                bottom_score = item_glob_popularity ** self.beta
                return row['upper_score'] / bottom_score

            # vytahnu svoje sousedy
            k_users_orig_items = pd.DataFrame(my_neighbors, columns=['my_neighbors_index'])
            # pridam jejich cosine similaritu (distance)
            k_users_orig_items['my_neighbors_distance'] = my_neighbors_distance
            # vypoctu prumerny rating uzivatele
            k_users_orig_items['avg_rating'] = k_users_orig_items.apply(f_avg_rating, axis=1)

            # nechame jen nadprumerny itemy
            k_users_orig_items['popular_item'] = k_users_orig_items.apply(f_popular_item, axis=1)
            # prevedu itemy do samostatneho dataframe
            frame = k_users_orig_items['popular_item'].values
            recomItems = pd.concat(frame)
            # vypocet rozdilu ratingu
            recomItems['rating_diff'] = recomItems.apply(f_rating_diff, axis=1)
            # vypocitam score
            recomItems['upper_score'] = recomItems.apply(f_score, axis=1)
            # provedu summu score a ratingu
            recomItems = recomItems.groupby(['movieIndex']).sum()
            # vratim zpet index
            recomItems['index'] = recomItems.index
            # vypocitam final beta score
            recomItems['final_score'] = recomItems.apply(f_beta_score, axis=1)

            # vyhodim itemy, ktery muj user uz videl
            recomItems = recomItems.loc[~recomItems.index.isin(cals_user)]
            # vyhodim nan
            recomItems = recomItems.dropna()
            # vse seradim
            recomItems = recomItems.sort_values(by=['final_score'], ascending=False)
            # riznu pocet rekomentaco
            recomItems = recomItems[:n]
            # !!!! moje rekomendace jsou indexy ve sparce matrix, ted potrebuje je prevest na movieId
            tmp = list(recomItems.index)

            # todle je kvuli onto divers
            #####
            # _itemIdArr = []
            # for i in tmp:
            #     _itemId = self.train_data_by_row_number[self.train_data_by_row_number.index.isin([i])]['index'].values
            #     _itemIdArr.append(_itemId[0])
            #
            # recomItems['movieId'] = _itemIdArr
            # self.recomItemsCache = recomItems
            ######

            recoms_movie = self.train_data_by_row_number[self.train_data_by_row_number.index.isin(tmp)]

            return_recoms = np.array(recoms_movie['index'].values).astype(int)
            # #vytahnu indexy item
            # recoms = np.array(recomItems.index)
            #
            # #vrati rekomendaci - jedna se o indexy filmu ve sparce matrix!
            # recoms = recoms[:num_of_recom]
            #
            # #potrebujem recomendaci prevest na movieId
            #
            # return_recoms =  np.array(recoms_movie['movieId'].values)

            return return_recoms

        except Exception as e:
            # print(self.name + ' predictItemByUser ' + str(e))
            return None
