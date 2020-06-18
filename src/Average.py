import pandas as pd
import numpy as np

users = pd.read_csv('users.csv')
movies = pd.read_csv('movies.tsv', sep='\t')
ratings = pd.read_csv('ratings.csv')
alldata = pd.read_csv('allData.tsv', sep='\t')
predict = pd.read_csv('predict.csv')

def Eval(test, actual):
    merged = test.merge(actual, how='inner', on ='movieID')
    distance = abs(merged['rating_x'] - merged['rating_y'])
    average_dist = sum(distance)/len(distance)
    number_correct = len(merged[merged.rating_x == merged.rating_y])
    percentage_correct = number_correct/len(merged)
    return (average_dist, number_correct, percentage_correct)


simplified_ratings = ratings[['movieID', 'rating']]

#Average
average = pd.DataFrame(ratings.groupby(['movieID']).mean()['rating'])
average_dist, number_correct, percentage_correct = Eval(average.round(), simplified_ratings)
print('Rounded Avg Fractional Rating:', average_dist)
print('Rounded Avg Integer Rating:', number_correct)
print('Rounded Avg Percentage Rating:', percentage_correct)

#Weighted Average
average = pd.DataFrame(ratings.groupby(['movieID']).mean()['rating'])
average_dist, number_correct, percentage_correct = Eval(average, simplified_ratings)
print('Fractional Rating:', average_dist)
print('Integer Rating:', number_correct)
print('Percentage Rating:', percentage_correct)

average_df = predict.merge(average, how='left', on='movieID')
average_df.drop(['rating_x','KNN_preds'], axis=1, inplace=True)
average_df.rename(columns={'rating_y':'rating'}, inplace=True)
average_df.to_csv('average_df.csv')
average_df.round().to_csv('average_df_v2.csv')

# Rating = Weight * Individual Rating + (1- Weight) * Global Rating
weight = 260*(ratings.groupby(['movieID']).count()['rating']/sum(ratings.groupby(['movieID']).count()['rating']))
ind_rating = ratings.groupby(['movieID']).mean()['rating']
global_rating = ratings.mean()['rating']
weighted_avg_rating = pd.DataFrame(weight*ind_rating + (1-weight) * global_rating)

average_dist, number_correct, percentage_correct = Eval(weighted_avg_rating, simplified_ratings)
print('Weighted Avg Fractional Rating:', average_dist)
print('Weighted Avg Integer Rating:', number_correct)
print('Weighted Avg Percentage Rating:', percentage_correct)

average_dist, number_correct, percentage_correct = Eval(weighted_avg_rating.round(), simplified_ratings)
print('Rounded Weighted Avg Fractional Rating:', average_dist)
print('Rounded Weighted Avg Integer Rating:', number_correct)
print('Rounded Weighted Avg Percentage Rating:', percentage_correct)

weighted_average_df = predict.merge(weighted_avg_rating, how='left', on='movieID')
weighted_average_df.drop(['rating_x'], axis=1, inplace=True)
weighted_average_df.rename(columns={'rating_y':'rating'}, inplace=True)
weighted_average_df.to_csv('weighted_average_df2.csv')
weighted_average_df.round().to_csv('weighted_average_df2_v2.csv')