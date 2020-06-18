import pandas as pd
import numpy as np
from surprise.model_selection import train_test_split, cross_validate, KFold, GridSearchCV
from surprise import SVD, accuracy, Reader, Dataset

users = pd.read_csv('users.csv')
movies = pd.read_csv('movies.tsv', sep='\t')
ratings = pd.read_csv('ratings.csv')
alldata = pd.read_csv('allData.tsv', sep='\t')
predict = pd.read_csv('predict.csv')

#Preprocessing Data- Trainset/Testset
trainset, testset = train_test_split(alldata, test_size= 0.2)
reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(ratings[['userID', 'movieID', 'rating']], reader)

SVD_params = {'n_factors': [80, 90, 100, 110, 120, 140], 'n_epochs': [10, 15, 20, 25, 30], 
              'lr_all': [0.001, 0.003, 0.005, 0.008], 'reg_all': [0.01, 0.02, 0.04, 0.8]}

gs_svd = GridSearchCV(SVD, SVD_params, measures=['rmse','mae'], cv=3, refit=True)
gs_svd.fit(data)

print(gs_svd.best_score['rmse'], gs_svd.best_score['mae'])
print(gs_svd.best_params['rmse'], gs_svd.best_params['mae'])

rmse_svd_algo = gs_svd.best_estimator['rmse']
cross_validate(rmse_svd_algo, data, measures=['rmse', 'mae'], cv = 5, verbose =True)

mae_svd_algo = gs_svd.best_estimator['mae']
cross_validate(mae_svd_algo, data, measures=['rmse', 'mae'], cv = 5, verbose =True)

SVD_preds = []
for i in range(len(predict)):
    _, _, _, svd_rating_pred, _ = best_svd_algo.predict(predict.userID[i], predict.movieID[i])
    SVD_preds.append(svd_rating_pred)

predict['SVD_preds'] = SVD_preds
SVD_Prediction = predict.drop(['rating','KNN_preds'], axis=1)
SVD_Prediction.rename(columns={'SVD_preds':'rating'}, inplace =True)
SVD_Prediction.to_csv('svd_pred2.csv')
SVD_Prediction.round().to_csv('svd_pred_2v2.csv')