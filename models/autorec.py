# ref: Stanislav Kuznetsov
import numpy as np
import tensorflow as tf


class AutoRec:
    def __init__(self, name, train_data, item_ratings):
        self.name = name

        self.train_data = train_data
        self.train_data_item_count = len(train_data)
        self.item_ratings = item_ratings
        self.train_data_t = self.train_data.sort_index().T.reset_index(drop=True).T

        self.model = None

        melt_data = self.train_data_t.reset_index('userId')
        melt_data = melt_data.drop(columns=['userId'])

        self.train_interaction_matrix = np.zeros((self.train_data.shape[1], self.train_data.shape[0])).astype(
            np.float32)

        for i, v in melt_data.T.iterrows():
            tmp = v.to_numpy(dtype='float32')
            for j in range(0, len(tmp)):
                self.train_interaction_matrix[i, j] = tmp[j]

        self.train_data_reset_index = self.train_data.T.reset_index()

    @staticmethod
    def autoencoder(X, hidden_layer_size):
        input_layer = tf.keras.layers.Input(shape=(X.shape[1],), name='UserScore')

        enc = tf.keras.layers.Dense(512, activation='selu', name='EncLayer1')(input_layer)
        lat_space = tf.keras.layers.Dense(hidden_layer_size, activation='selu', name='LatentSpace')(enc)

        lat_space = tf.keras.layers.Dropout(0.8, name='Dropout')(lat_space)  # Dropout
        dec = tf.keras.layers.Dense(512, activation='selu', name='DecLayer1')(lat_space)

        output_layer = tf.keras.layers.Dense(X.shape[1], activation='linear', name='UserScorePred')(dec)

        # this model maps an input to its reconstruction
        model = tf.keras.models.Model(input_layer, output_layer)
        return model

    def fit(self, params):
        try:
            tf.random.set_seed(params['random_state'])
            self.model = self.autoencoder(self.train_interaction_matrix.T, params['hidden_layer_size'])
            self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='mse')
            self.model.summary()

            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1)

            self.model.fit(
                self.train_interaction_matrix.T, self.train_interaction_matrix.T,
                epochs=50, batch_size=64,
                validation_split=0.2, shuffle=True, callbacks=[es], verbose=False
            )

        except Exception as e:
            print(self.name + ' fit ' + str(e))

    def predict(self, user_profile, n):
        try:
            prediction = np.array(self.model.predict(user_profile.values.reshape(1, -1)))
            candidates = prediction[0].argsort()[-n:]
            # moje rekomendace jsou indexy v matici, potebuju je prevest na filmy

            recommendations = []
            for r in candidates:
                movie_id = self.train_data_reset_index[self.train_data_reset_index.index.isin([r])]['index'].values[0]
                recommendations.append(movie_id)

            recommendations = np.array(recommendations).astype(int)
            return recommendations

        except Exception as e:
            print(self.name + ' predictItemByUser ' + str(e))
            return None
