import numpy as np
import pandas as pd

#Solution from:
#https://towardsdatascience.com/beginners-guide-to-creating-an-svd-recommender-system-1fd7326d1f65

#Funk SVD
n_latent_factors = 20
learning_rate = 0.01
regularizer = 0.02
max_epochs = 100
stop_threshold = 0.005

def get_triples(from_set):
    triples = list()

    for user, movie_ratings in from_set.items():
        for movie, rating in movie_ratings.items():
            triples.append((user, movie, rating))
    return triples


def get_movies(from_set):
    movies = set()

    for user, movie_ratings in from_set.items():
        movies.update(movie_ratings.keys())


def initialize_missing_values(num_movies, num_users):
    rand_movie_values = np.random.rand(num_movies, latent_factors)
    rand_user_values = np.random.rand(latent_factors, num_users)

def MAE(train_set, movie_val, user_val):
    # Compute the predicted rating matrix
    predicted = np.matmul(movie_val, user_val)

    n_instances = 0
    mean_absolute_errors = 0
    for user, movie_rating in on_set.items():
        n_instances += len(movie_rating.keys())

        for movie, rating in movie_rating.items():
            sum_absolute_errors += np.absolute(predicted[movie][user] - rating)

    return np.sum(mean_absolute_errors)/ n_instances

def RMSE(train_set, movie_val, user_val):
    # Compute the predicted rating matrix
    predicted = np.matmul(movie_val, user_val)

    n_instances = 0
    sum_squared_errors = 0
    for user, movie_rating in on_set.items():
        n_instances += len(movie_rating.keys())

        for movie, rating in movie_rating.items():
            sum_squared_errors += (predicted[movie][user] - rating) ** 2

    return np.sqrt(sum_squared_errors / n_instances)

def SVD_and_SGD(train):
    movie_set = get_movies(train)
    movie_val, user_val = initialize_missing_values(max(movie_set) + 1, max(train.keys()) + 1)

    # Training instances are represented as a list of triples
    triples = get_triples(train)
    last_rmse = None

    for epoch in range(max_epochs):
        # At the start of every epoch, we shuffle the dataset
        # Shuffling may not be strictly necessary, but is an attempt to avoid overfitting
        random.shuffle(triples)

        # Calculate RMSE for training set, stop if change is below threshold
        rmse = calculate_rmse(train, movie_val, user_val)
        logger.info(f'Epoch {epoch}, RMSE: {rmse}')
        if last_rmse and abs(rmse - last_rmse) < stop_threshold:
            break
        last_rmse = rmse

        for user, movie, rating in triples:
            # Update values in vector movie_values
            for k in range(n_latent_factors):
                error = sum(movie_values[movie][i] * user_values[i][user] for i in range(n_latent_factors)) - rating

                # Compute the movie gradient
                # Update the kth movie factor for the current movie
                movie_gradient = error * user_values[k][user]
                movie_values[movie][k] -= learning_rate * (movie_gradient - regularizer * movie_values[movie][k])

                # Compute the user gradient
                # Update the kth user factor the the current user
                user_gradient = error * movie_values[movie][k]
                user_values[k][user] -= learning_rate * (user_gradient - regularizer * user_values[k][user])

    return movie_values, user_values


#Cosine Similarity
#This will not handle a sparse matrix
def cos_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.sqrt(vec1.dot(vec1))) * (np.sqrt(vec2.dot(vec2)))

#sklearn's function accounts for sparse input and supports sparse output
from sklearn.metrics.pairwise import cosine_similarity

cos_sim = cosine_similarity(sparse_mat, dense_output=False)
print('Sparse Output:\n {}\n'.format(cos_sim))