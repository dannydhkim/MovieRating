import pandas as pd
import numpy as np
from surprise import KNNBaseline, accuracy, Reader, Dataset
from surprise.model_selection import train_test_split, cross_validate, KFold, GridSearchCV

users = pd.read_csv('users.csv')
movies = pd.read_csv('movies.tsv', sep='\t')
ratings = pd.read_csv('ratings.csv')
alldata = pd.read_csv('allData.tsv', sep='\t')
predict = pd.read_csv('predict.csv')

#Preprocessing Data- Trainset/Testset
trainset, testset = train_test_split(alldata, test_size= 0.2)
reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(ratings[['userID', 'movieID', 'rating']], reader)

#Memory-based Collaborative Filtering (User) Evaluation
param_grid = {'k': [10, 20, 30, 40, 50, 60, 70],
              'sim_options': {'name': ['msd', 'cosine'],
                              'min_support': [1, 3, 5],
                              'user_based': [True]},
              'bsl_options': {'reg_i': [5, 10, 15, 20],
                             'reg_u': [10, 15, 20, 25],
                             'n_epochs': [10, 15]}
              }

gs = GridSearchCV(KNNBaseline, param_grid, measures=['rmse','mae'], cv=3, refit=True)
gs.fit(data)

print(gs.best_score['rmse'], gs.best_score['mae'])
print(gs.best_params['rmse'], gs.best_params['mae'])

knn_algo = gs.best_estimator['mae']
cross_validate(knn_algo, data, measures=['rmse', 'mae'], cv=5, verbose =True)

KNN_preds = []
for i in range(len(predict)):
    _, _, _, rating_pred, _ = gs.predict(predict.userID[i], predict.movieID[i])
    KNN_preds.append(rating_pred)

predict['KNNbase_preds'] = KNN_preds

KNN_df = predict.drop(['rating'], axis=1)
KNN_df.rename(columns={'KNNbase_preds':'rating'}, inplace=True)
KNN_df.to_csv('KNNBase_rating_v1.csv')
KNN_df.round().to_csv('KNNBase_rating_v2.csv')

