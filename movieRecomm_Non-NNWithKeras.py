%matplotlib inline
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading ratings file
ratings = pd.read_csv('C:\\Users\\Sreecharan\\Downloads\\movielens-master\\movielens-master\\ratings.csv', sep='\t', encoding='latin-1', 
                      usecols=['user_id', 'movie_id', 'user_emb_id', 'movie_emb_id', 'rating'])
max_userid = ratings['user_id'].drop_duplicates().max()
max_movieid = ratings['movie_id'].drop_duplicates().max()

# Reading ratings file
users = pd.read_csv('C:\\Users\\Sreecharan\\Downloads\\movielens-master\\movielens-master\\users.csv', sep='\t', encoding='latin-1', 
                    usecols=['user_id', 'gender', 'zipcode', 'age_desc', 'occ_desc'])

# Reading ratings file
movies = pd.read_csv('C:\\Users\\Sreecharan\\Downloads\\movielens-master\\movielens-master\\movies.csv', sep='\t', encoding='latin-1', 
                     usecols=['movie_id', 'title', 'genres'])

# Create training set
shuffled_ratings = ratings.sample(frac=1., random_state=2)
# Shuffling users
Users = shuffled_ratings['user_emb_id'].values
print('Users:', Users, ', shape =', Users.shape,' of length: ', Users.size)

# Shuffling movies
Movies = shuffled_ratings['movie_emb_id'].values
print('Movies:', Movies, ', shape =', Movies.shape)

# Shuffling ratings
Ratings = shuffled_ratings['rating'].values
print ('Ratings:', Ratings, ', shape =', Ratings.shape)

# Import Keras libraries
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

# Define constants
K_FACTORS = 100 # The number of dimensional embeddings for movies and users
TEST_USER = 2033 # A random test user (user_id = 2033)

from keras.models import Model
from keras.layers import Input, Reshape, Dot
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.regularizers import l2

# P is the embedding layer that creates an User by latent factors matrix.
# If the intput is a user_id, P returns the latent factor vector for that user.
user = Input(shape=(1,))
u = Embedding(max_userid, K_FACTORS, embeddings_initializer='he_normal',
          embeddings_regularizer=l2(1e-6))(user)
u = Reshape((K_FACTORS,))(u)

# Q is the embedding layer that creates a Movie by latent factors matrix.
# If the input is a movie_id, Q returns the latent factor vector for that movie.
movie = Input(shape=(1,)) #initiate another tensor 
m = Embedding(max_movieid, K_FACTORS, embeddings_initializer='he_normal',
          embeddings_regularizer=l2(1e-6))(movie)
m = Reshape((K_FACTORS,))(m)

#perform dot product
x = Dot(axes=1)([u, m])
model = Model(inputs=[user, movie], outputs=x)
opt = Adam(lr=0.001)
model.compile(loss='mean_squared_error', optimizer=opt)
print('Model Built')

# Callbacks monitor the validation loss
# Save the model weights each time the validation loss has improved
callbacks = [EarlyStopping('val_loss', patience=2), 
             ModelCheckpoint('weights.h5', save_best_only=True)]

# Use 30 epochs, 90% training data, 10% validation data 
history = model.fit([Users, Movies], Ratings, epochs=10, validation_split=.1, verbose=2, callbacks=callbacks)

min_val_loss, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
print ('Minimum RMSE at epoch', '{:d}'.format(idx+1), '=', '{:.4f}'.format(math.sqrt(min_val_loss)))
# Load weights
model.load_weights('weights.h5')


def rate(model, user_id, item_id):
    return model.predict([np.array([user_id]), np.array([item_id])])[0][0]

# Function to predict the ratings given User ID and Movie ID
def predict_rating(user_id, movie_id):
    return rate(model,user_id - 1, movie_id - 1)

users[users['user_id'] == TEST_USER]



user_ratings = ratings[ratings['user_id'] == TEST_USER][['user_id', 'movie_id', 'rating']]
user_ratings['prediction'] = user_ratings.apply(lambda x: predict_rating(TEST_USER, int(x['movie_id'])), axis=1)
user_ratings.sort_values(by='rating', 
                         ascending=False).merge(movies, 
                                                on='movie_id', 
                                                how='inner', 
                                                suffixes=['_u', '_m']).head(20)



#Here I make a recommendation list of unrated 20 movies sorted by prediction value for user in TEST_USER constant. Let's see it.
recommendations = ratings[ratings['movie_id'].isin(user_ratings['movie_id']) == False][['movie_id']].drop_duplicates()
recommendations['prediction'] = recommendations.apply(lambda x: predict_rating(TEST_USER, int(x['movie_id'])), axis=1)
recommendations.sort_values(by='prediction',
                          ascending=False).merge(movies,
                                                 on='movie_id',
                                                 how='inner',
                                                 suffixes=['_u', '_m']).head(20)

