import pandas as pd

# # Load the dataset
# ratings = pd.read_csv(
#     'data/ml_1m/ratings.dat',
#     sep='::',
#     names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'],
#     engine='python',
#     encoding='latin-1'
# )

# # Convert only numeric columns to int32
# numeric_cols = ['u_nodes', 'v_nodes', 'ratings']
# ratings[numeric_cols] = ratings[numeric_cols].astype('int32')

# print(ratings.describe())
# print(len(ratings['v_nodes'].unique()))

import numpy as np
data = np.loadtxt('/home/iec/hainn/rcm/implement_global_K/data/MovieLens_1M/movielens_1m_dataset.dat', skiprows=0, delimiter='::').astype('int32')


n_u = np.unique(data[:,0]).size  # num of users
n_m = np.unique(data[:,1]).size  # num of movies
n_r = data.shape[0]  # num of ratings

print(n_u, n_m)