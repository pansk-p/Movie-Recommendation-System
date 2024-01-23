# In[1]:
#make necesarry imports
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import squareform,pdist
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae,precision_score,recall_score
from scipy.sparse import csr_matrix

global k, percent
k = 20
percent = 0.9   # percentage of train-test split

#Read the data,preprocess it and split to train and test sets
def data_init():
    ratings = pd.read_csv("C:/Users/Panos/Desktop/data/ratings.csv", usecols = ['userId','movieId','rating'])
    filtered_movies = ratings.groupby('movieId').filter(lambda x: x['movieId'].count()>5)
    
    
    train, test = train_test_split(filtered_movies, 
                                   train_size = percent,
                                   test_size = 0.1,
                                   random_state = 2,)

    train = train.pivot(index='movieId',columns='userId',values='rating')
    train.fillna(0,inplace=True)
   
    test = test.pivot(index='movieId',columns='userId',values='rating')
    test.fillna(0,inplace=True)
    
    return train, test

#This functions calls the chosen method to calculate the item similarity
def calculate_simMatrix(matrix, mode):
    if(mode == 1):
        sim_matrix = jaccard(matrix)
    elif(mode == 2):
        sim_matrix = adjcosine(matrix)
    else:
        print("Type: 1 for jaccard , 2 for adjusted cosine!")   
    return  matchIndicies(matrix, sim_matrix)

#This function is used to compute jaccard similarity matrix for items
def jaccard(matrix):
      sim_matrix = 1 - pairwise_distances(matrix.T, metric = "hamming")
      sim_matrix = pd.DataFrame(sim_matrix)
      return sim_matrix

#This function is used to compute adjusted cosine similarity matrix for items
def adjcosine(matrix):
    M_u = matrix.mean(axis=1)
    item_mean_subtracted = matrix - M_u[:, None]
    similarity_matrix = 1 - squareform(pdist(item_mean_subtracted.T, 'cosine'))
    similarity_matrix = pd.DataFrame(similarity_matrix)
    return similarity_matrix

#This function match the indicies(row & column)from one matrix to another
def matchIndicies(fromMatrix, toMatrix):
    fromMatrix = fromMatrix.T
    toMatrix.index = fromMatrix.index.copy()
    sim_matrix = toMatrix.T
    sim_matrix.index = fromMatrix.index.copy()
    print("Similarity Matrix Calculated!\n")
    return sim_matrix

#This function finds k similar items given the item_id and ratings matrix 
def find_k_similar_items(item_id, simMatrix):
    similarities = simMatrix[item_id].sort_values(ascending=False)[:k+1].values
    indices = simMatrix[item_id].sort_values(ascending=False)[:k+1].index
    
    #print ('For user {0}, {1} most similar items for item {2}:\n'.format(user_id,k,item_id))
    #for i in range(0, len(indices)):
    #        if indices[i] == item_id:
    #            continue;
    #        else:
               #print ('{0}: Item {1} :, with similarity of {2}'
                #.format(i,indices[i], similarities[i]))
    #            continue;
    return similarities ,indices  



#item-item collaborative filtering
def S1(train, test, mode):
    itemFilter = test.copy()
    sim_matrix = calculate_simMatrix(train, mode)
    for row,k in test.iterrows(): 
       for column in test.columns: 
            try:
               #find movies that are similar and the prediction rating is above 2,5 and store them             
                if(test.loc[row,column] != 0):  
                    prediction = predict_itembased(row, column, train, sim_matrix)
                    if(prediction < 2.5):    
                        itemFilter.at[row,column] = -1 # this item is irrelevant
                        test.at[row,column] = 0
                    else:
                        print('Prediction',prediction)
                        continue;       
            except:
                continue;      
    print("\n\nSuccess S1!\n\n")         
    return itemFilter

#user-item collaborative filtering          
def S2(train, itemFilter, test, mode):
    final_data = test.copy()
    sim_matrix = calculate_simMatrix(train.T, mode)
    for row,k in test.iterrows(): 
       for column in test.columns: 
            try: 
               #predict only for filtered movies from S1()
               if(itemFilter.loc[row,column] != -1):
                    #predict only for movies the user has not seen
                    if(test.loc[row,column] != 0):
                        prediction = predict_userbased(row, column, train, sim_matrix)
                        if(prediction > 1):
                            print('Prediction',prediction) 
                        final_data.at[row,column] = prediction
                
            except:
                continue;              
    print("\n\nSuccess S2!\n\n")                   
    return final_data

#Calculate mean absolute error and precision,recall 
def calculateMetrics(true, pred):
   ratings_true = []
   ratings_pred = []
   for row,k in true.iterrows(): 
       for column in true.columns: 
           if(true.loc[row,column] != 0):
               ratings_true.append(true.loc[row,column])
               ratings_pred.append(pred.loc[row,column])

   mae_score = mae(ratings_true, ratings_pred, multioutput='raw_values')
   print('Mae: %.3f' % mae_score)

   for i in range(len(ratings_pred)):
      if ratings_pred[i] > 3 :
        ratings_pred[i] = 1
      else:
        ratings_pred[i] = 0
      if ratings_true[i] > 3 :
        ratings_true[i] = 1
      else:
        ratings_true[i] = 0     

   precision = precision_score(ratings_true, ratings_pred, average='binary')
   print('Precision: %.3f' % precision)
 
   recall = recall_score(ratings_true, ratings_pred, average='binary')
   print('Recall: %.3f' % recall)


# %%
train, test = data_init()
train_filtered = S1(train, test, 2)
final = S2(train, train_filtered, test, 2)
calculateMetrics(test, final)

#test.to_csv("C:/Users/Panos/Desktop/test.csv", sep='\t')
#final.to_csv("C:/Users/Panos/Desktop/final.csv", sep='\t')


# %%

#This function predicts rating for specified user-item combination based on user-based approach
def predict_userbased(user_id, item_id, ratingMatrix, simMatrix):
    prediction = wtd_sum =0
    similarities, indices=find_k_similar_items(user_id, simMatrix) #similar users based on cosine similarity
    sum_wt = np.sum(similarities)-1
    product = 1 
    for i in range(0, len(indices)):
        if indices[i] == user_id:
            continue;
        else:
           #print( ratingMatrix.loc[indices[i],item_id] , similarities[i])
            product = ratingMatrix.loc[indices[i],item_id] * similarities[i]
            wtd_sum = wtd_sum + product
    prediction = round((wtd_sum/sum_wt),1)
    #print ('\nPredicted rating for user {0} -> item {1}: {2}\n'.format(user_id,item_id,prediction))
    if prediction < 0:
        prediction = 0
    elif prediction > 5:
        prediction = 5
    return prediction


#This function predicts the rating for specified user-item combination based on item-based approach
def predict_itembased(user_id, item_id, ratingMatrix,simMatrix):
    prediction = wtd_sum =0
    similarities, indices=find_k_similar_items(item_id, simMatrix) #similar users based on correlation coefficients
    sum_wt = np.sum(similarities)-1
    product = 1
    for i in range(0, len(indices)):
        if indices[i] == item_id:
            continue;
        else:
            #print( ratingMatrix.loc[user_id,indices[i]], similarities[i])
            product = ratingMatrix.loc[user_id,indices[i]] * (similarities[i])
            wtd_sum = wtd_sum + product                                      
    prediction = round((wtd_sum/sum_wt),1)
    if prediction < 0:
        prediction = 0
    elif prediction >5:
        prediction = 5
    #print ('\nPredicted rating for user {0} -> item {1}: {2}'
    #.format(user_id,item_id,prediction))     
    return prediction

# %%
