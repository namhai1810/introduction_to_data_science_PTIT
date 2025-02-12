import numpy as np
import pandas as pd
import torch
from sklearn import model_selection, metrics, preprocessing

def map_data(data):
    """
    Map data to proper indices in case they are not in a continues [0, N) range
    Parameters
    ----------
    data : np.int32 arrays
    Returns
    -------
    mapped_data : np.int32 arrays
    n : length of mapped_data
    """
    uniq = list(set(data))

    id_dict = {old: new for new, old in enumerate(sorted(uniq))}
    data = np.array(list(map(lambda x: id_dict[x], data)))
    n = len(uniq)

    return data, id_dict, n


# UserID::Gender::Age::Occupation::Zip-code
# MovieID::Title::Genres
# UserID::MovieID::Rating::Timestamp (5-star scale)

# Importing the dataset
#movies = pd.read_csv('./data/ml_1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
#users = pd.read_csv('./data/ml_1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

###############################
ratings = pd.read_csv('data/ml_1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

total_length = len(ratings)
ratings = ratings.sample(frac=1)

len_train = int(total_length*0.9)

rating_train = ratings[:len_train]
rating_val   = ratings[len_train:]


num_users  = 6040
num_items = 3706
rating_cnt = 5
# process new data
lbl_movie = preprocessing.LabelEncoder()
ratings[1] = lbl_movie.fit_transform(ratings[1])

for i, ratings in enumerate([rating_train, rating_val]):
    rating_mtx = torch.zeros(rating_cnt, num_users, num_items)
    
    for index, row in ratings.iterrows():
        u = row[0]-1
        v = row[1]-1
        r = row[2]-1
        
        rating_mtx[r, u, v] = 1
    print(rating_mtx.shape)
    torch.save(rating_mtx, './data/rating_%d.pkl'%i)


# users_headers = ['user id', 'gender', 'age', 'occupation', 'zip code']
# users_df = pd.read_csv('data/ml_1m/users.dat', sep = '::', header = None, names = users_headers, engine = 'python', encoding = 'latin-1')
# movie_headers = ['movie id', 'movie title', 'genre']
# movie_df = pd.read_csv('data/ml_1m/movies.dat', sep = '::', header = None, names = movie_headers, engine = 'python', encoding = 'latin-1')


# ####################################################
# occupation = set(users_df['occupation'].values.tolist())
# age_dict = {1:0., 18:1., 25:2., 35:3., 45:4., 50:5., 56:6.}
# gender_dict = {'M': 0., 'F': 1.}
# occupation_dict = {f: i for i, f in enumerate(occupation, start=2)}

# num_feats = 2 + len(occupation_dict)

# u_features = np.zeros((num_users, num_feats), dtype=np.float32)
# for _, row in users_df.iterrows():
#     u_id = row['user id']-1
#     # age
#     u_features[u_id, 0] = age_dict[row['age']]
#     # gender
#     u_features[u_id, 1] = gender_dict[row['gender']]
#     # occupation
#     u_features[u_id, occupation_dict[row['occupation']]] = 1.
# torch.save(torch.from_numpy(u_features), './data/ml_1m/u_features.pkl')

# ###################################################################3
# genre_dict = {'Action':0, 'Adventure':1, 'Animation':2, "Children's":3, 'Comedy':4,
#               'Crime':5, 'Documentary':6, 'Drama':7, 'Fantasy':8, 'Film-Noir':9, 'Horror':10,
#               'Musical':11, 'Mystery':12, 'Romance':13, 'Sci-Fi':14, 'Thriller':15,
#               'War':16, 'Western':17}
# num_genres = len(genre_dict)

# v_features = np.zeros((num_items, num_genres), dtype=np.float32)
# for movie_id, g_vec in zip(movie_df['movie id'].values.tolist(), movie_df['genre'].values.tolist()):
#     # check if movie_id was listed in ratings file and therefore in mapping dictionary
#     for j in [genre_dict[g] for g in g_vec.split('|')]:
#         v_features[movie_id-1][j] = 1

# torch.save(torch.from_numpy(v_features), './data/ml_1m/v_features.pkl')