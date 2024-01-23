#make necesarry imports
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae,precision_score,recall_score
from datetime import datetime
from itertools import combinations

def data_initialization(path, percentage):
    ratings = pd.read_csv(path, usecols = ['userId','movieId','rating'])
    filtered_movies = ratings[ratings['movieId'].map(ratings['movieId'].value_counts()) >= 5]
    filtered_users = filtered_movies[filtered_movies['userId'].map(filtered_movies['userId'].value_counts()) >= 5]
    
    train, test = train_test_split(filtered_users, 
                                   train_size = percentage,
                                   test_size = 0.1,
                                   random_state = 1)
    #precalculate user mean and common users who rated all 2-movie combinations
    users = train.groupby('movieId')['userId'].apply(list).to_dict()
    counts = {(mov1, mov2): len(set(users[mov1]).intersection(users[mov2])) for mov1, mov2 in combinations(set(train['movieId']), 2)}
    mean = train.groupby(by='userId', as_index=True)['rating'].mean() 
    
    train = train.pivot(index='movieId', columns='userId', values='rating')
    test = test.pivot(index='movieId', columns='userId', values='rating')
    predictions = pd.DataFrame(np.nan, index = test.index, columns = test.columns)
    similarities = train.T.corr(method="pearson", min_periods=5)
    similarities.replace(0.0, np.nan, inplace=True) 
    return train, test, similarities, predictions, mean, counts

#This function finds k similar items given the item_id and ratings matrix 
def find_k_similar_items(item_id, similarity_matrix, neighbors):
    similarities = similarity_matrix[item_id].dropna().sort_values(ascending=False)[:neighbors].values
    indices = similarity_matrix[item_id].dropna().sort_values(ascending=False)[:neighbors].index
    return similarities, indices  

#This function finds k++ similar items given the item_id and ratings matrix 
def find_kplus_similar_items(train, item_id, user_id, similarity_matrix, neighbors):
    similarities = []
    indices = []
    temp_s = similarity_matrix[item_id].dropna().sort_values(ascending=False).values 
    temp_i = similarity_matrix[item_id].dropna().sort_values(ascending=False).index
    for i in range(0, len(temp_i)):   
        if not np.isnan(train.loc[temp_i[i]][user_id]) and temp_i[i] != item_id and temp_s[i] >= 0.5:
            similarities.append(temp_s[i])
            indices.append(temp_i[i])
            if len(indices) == neighbors: break     
    return similarities, indices  

#This function predicts the rating for specified user-item combination based on item-based approach
def predict_itembased(train, item_id, user_id, similarity_matrix, neighbors, mode, mean, counts):
    prediction = product = sum_p = sum_c = no_rates = 0
    similarities, indices = find_k_similar_items(item_id, similarity_matrix, neighbors)
    #similarities, indices = find_kplus_similar_items(train, item_id, user_id, similarity_matrix, neighbors)
    sum_s = np.sum(similarities)
    for i in range(0, len(indices)):
        #if the item itself is in the similarities remove it
        if indices[i] == item_id:
            sum_s = sum_s - similarities[i]
            continue
        #we don't count ratings for movies that are NaN(user hasn't rated yet) even they are similiar
        if np.isnan(train.loc[indices[i], user_id]):
            no_rates += 1
            sum_s = sum_s - similarities[i]
            continue
        if mode == 0:
            product = train.loc[indices[i], user_id]
        elif mode == 1:
            product = train.loc[indices[i], user_id] * similarities[i] 
        elif mode == 2:
            #dictionary store keys in form (A,B),use try to catch the exception and reverse A,B
            try:
                c = counts[(item_id, indices[i])]
            except: 
                c = counts[(indices[i], item_id,)]
            product = train.loc[indices[i], user_id] * c
            sum_c += c  
        sum_p += product                                      
    #in this case user has not rated any similiar movie, prediction becomes the mean of user's rating
    if sum_p == 0 or sum_s == 0:
        return round(mean[user_id], 1)
    #prediction as average 
    if mode == 0:
        prediction = round((sum_p/(neighbors-no_rates)), 1)
    #prediction as weighted average
    elif mode == 1:
        prediction = round((sum_p/sum_s), 1)
    #prediction as weighted average and weights as user counts
    elif mode == 2:
        prediction = round((sum_p/sum_c), 1)
    #normalize values
    if prediction < 0.5: prediction = 0.5
    elif prediction > 5: prediction = 5.0
    return prediction
            
#Calculate mean absolute error and precision,recall 
def calculate_metrics(true, pred):
   print("Calculating metrics...") 
   ratings_true = []
   ratings_pred = []
   for row,k in true.iterrows(): 
       for column in true.columns: 
           if not np.isnan(true.loc[row,column]) and not np.isnan(pred.loc[row,column]):
               ratings_true.append(true.loc[row,column])
               ratings_pred.append(pred.loc[row,column])
           
   mae_score = mae(ratings_true, ratings_pred)
   print('Mae: %.3f' % mae_score)

   ratings_pred = [int(i>=3) for i in ratings_pred]
   ratings_true = [int(i>=3) for i in ratings_true]
   print('Precision: %.3f' % precision_score(ratings_true, ratings_pred))
   print('Recall: %.3f' % recall_score(ratings_true, ratings_pred))

if __name__ == "__main__":
    if len(sys.argv) != 4:
            print("Usage: <source> <path> <neighbors> <train-percentage>", file=sys.stderr)
            sys.exit(-1)
        
    path = sys.argv[1]
    neighbors = int(sys.argv[2])
    percentage = float(sys.argv[3])
    train, test,similarities, predictions, mean, counts = data_initialization(path, percentage)
    print('Neighbors: ' + str(neighbors) + '\n' + 'Train split percentage: ' + str(percentage))
    for mode in range(3):
        print('Calculating prediction function ' + str(mode+1))
        start_time = datetime.now()
        for row,i in test.iterrows(): 
            for column in test.columns:
                try:         
                    if not np.isnan(test.loc[row,column]) and column in train:
                        predictions.at[row,column] = predict_itembased(train, row, column, similarities, neighbors, mode, mean, counts)     
                except:
                        continue; 
        calculate_metrics(test, predictions)
        end_time = datetime.now()
        print('Duration: {}\n'.format(end_time - start_time))



