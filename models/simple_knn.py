# ref: Stanislav Kuznetsov
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


class KNN:
    def __init__(self, name, train_data, ratings):
        self.name = name

        self.train_data = train_data
        self.ratings = ratings

        self.k = 0
        self.model_knn = None

    def fit(self, params):
        try:
            self.k = params['k']
            self.model_knn = NearestNeighbors(metric='cosine', algorithm='auto', n_neighbors=self.k, n_jobs=-1)
            self.model_knn.fit(csr_matrix(self.train_data.values))

        except Exception as e:
            print(self.name + ' fit ' + str(e))

    def predict(self, user_movies_profile, n):
        # prevedu profil na matici
        crc_profile = csr_matrix(user_movies_profile)
        # spocitam sousedy
        distances, indices = self.model_knn.kneighbors(crc_profile, n_neighbors=self.k)
        # vytahnu profily sousedu
        tmp = indices[0]
        # seradim jejich filmy podle ratingu
        k_users_orig_items = self.ratings[self.ratings['userId'].isin(tmp)].sort_values(by=['rating'], ascending=False)
        # vytahnu serazeny unikatni filmy
        recoms = k_users_orig_items.movieId.unique()
        # vrati rekomendaci
        recoms = recoms[:n]

        return recoms
