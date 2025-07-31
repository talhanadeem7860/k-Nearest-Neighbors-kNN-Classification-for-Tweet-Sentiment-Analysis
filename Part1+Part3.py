#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from collections import Counter
import pandas as pd
import re as regex
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cdist 
import matplotlib.pyplot as plt
import plotly
import re
from gensim.models import KeyedVectors


# In[4]:


#reading training and testing data
training_data = pd.read_csv(r'C:\Users\Talha\Desktop\train.csv')
testing_data = pd.read_csv(r'C:\Users\Talha\Desktop\test.csv')

#reading stopwords
my_file =open("C:\\Users\\Talha\\Desktop\\stop_words.txt")
content = my_file.read()
stop_words = content.split("\n")
my_file.close()


# In[5]:


def clean_tweets(tweet):
    
    # Remove usernames
    tweet = re.sub(r"@[^\s]+[\s]?",'',tweet)

    
    # remove URL
    tweet = re.sub(r"http\S+", "", tweet)
    
        
    # remove special characters 
    tweet = re.sub('[^ a-zA-Z0-9]', '', tweet)
    
    # remove Numbers
    tweet = re.sub('[0-9]', '', tweet)
   
    #changes to tweet to lowercase
    tweet = tweet.lower()
   
    return tweet


# In[6]:


#applying function clean_tweet to training data
training_data['Tweet'] = training_data['Tweet'].apply(clean_tweets)


# In[7]:


training_data.head()


# In[8]:


# Function to tokenize the data
def tweet_tokenize(tokenized_text):
    return tokenized_text.split()


# In[9]:


training_data['Tokenized Text']=training_data['Tweet'].apply(tweet_tokenize)


# In[10]:


training_data.head()


# In[11]:


words = Counter() 
for serial_no in training_data.index: #loop for all the values in the training data
    words.update(training_data.loc[serial_no, "Tokenized Text"])#counts the number of words in the Tokenized Text column
#words.most_common(11679)
#print(serial_no)


# In[12]:


#delete the stopwords from text
for serial_no, word in enumerate(stop_words):
        del words[word]
#print(serial_no)        
#words.most_common(179)


# In[13]:


#makes the list of words appearing in training data
def dictionary(iterated_dataa):
    
    min_freq=1 
    max_freq=3150 
    dict_df = pd.DataFrame(data={"Word": [k for k, v in words.most_common() if min_freq < v < max_freq],
                                 "Frequency": [v for k, v in words.most_common() if min_freq < v < max_freq]},
                           columns=["Word", "Frequency"]) #converts data into table format
    #print(word_df)
    vocab =dict_df.to_csv(r'C:\Users\Talha\Desktop\Dictionary.csv', index_label="serial no")
    dictionary = [k for k, v in words.most_common() if min_freq < v < max_freq]
    


# In[14]:


dictionary(training_data) #applies the dictionary function on training data


# In[15]:


#makes bag of word
dict_= []
dict_df = pd.read_csv(r'C:\Users\Talha\Desktop\Dictionary.csv')
dict_df = dict_df[dict_df["Frequency"] > 1]
dict_ = list(dict_df.loc[:, "Word"])

label_column = ["label"]
columns = label_column + list(map(lambda w: w + '',dict_))
labels = []
rows = []
for serial_no in training_data.index:
    current_row = []
    
    # add label to 
    current_label = training_data.loc[serial_no, "Sentiment"]
    labels.append(current_label)
    current_row.append(current_label)

    # add bag-of-words
    ref = set(training_data.loc[serial_no, "Tokenized Text"])
    for _, word in enumerate(dict_):
        current_row.append(1 if word in ref else 0) #makes the 0,1 vector

    rows.append(current_row)

data_vect = pd.DataFrame(rows, columns=columns) #makes the whole dataframe
data_labels = pd.Series(labels)


# In[ ]:


file =data_vect.to_csv(r'C:\Users\Talha\Desktop\BOW.csv', index_label="serial no") #creates the bow file


# In[ ]:


#preprocessing of testing data
def clean_testtweets(test_tweets):
    
    
    # Remove usernames
    test_tweet = re.sub(r"@[^\s]+[\s]?",'',test_tweet)
    
    
    # remove URL
    test_tweet = re.sub(r"http\S+", "", test_tweet)
    
    # remove special characters 
    test_tweet = re.sub('[^ a-zA-Z0-9]', '', test_tweet)
    
    # remove Numbers
    test_tweet = re.sub('[0-9]', '', test_tweet)
    test_tweet = test_tweet.lower()
    
    return test_tweet


# In[ ]:


testing_data['Tweet'] = testing_data['Tweet'].apply(clean_tweets)


# In[ ]:


testing_data.head()


# In[ ]:


testing_data['Tokenized Text']=testing_data['Tweet'].apply(tweet_tokenize)


# In[ ]:


testing_data.head()


# In[ ]:


words = Counter()
for serial_no in testing_data.index:
    words.update(testing_data.loc[serial_no, "Tokenized Text"])
#words.most_common(4)


# In[ ]:


for serial_no, word in enumerate(stop_words):
        del words[word]
#print(serial_no)        
#words.most_common(4)


# In[ ]:


#makes bag of word
dict_= []
dict_df = pd.read_csv(r'C:\Users\Talha\Desktop\Dictionary.csv')
dict_df = dict_df[dict_df["Frequency"] > 1]
dict_ = list(dict_df.loc[:, "Word"])

label_column = ["label"]
columns = label_column + list(map(lambda w: w + '',dict_))
labels = []
rows = []
for serial_no in testing_data.index:
    current_row = []
    
    # add label to 
    current_label = testing_data.loc[serial_no, "Sentiment"]
    labels.append(current_label)
    current_row.append(current_label)

    # add bag-of-words
    ref = set(testing_data.loc[serial_no, "Tokenized Text"])
    for _, word in enumerate(dict_):
        current_row.append(1 if word in ref else 0) #makes the 0,1 vector

    rows.append(current_row)

test_data_vect = pd.DataFrame(rows, columns=columns) #makes the whole dataframe
data_labels = pd.Series(labels)


# In[ ]:


file =test_data_vect.to_csv(r'C:\Users\Talha\Desktop\BOW_test.csv')


# In[ ]:


train_data_vect = pd.read_csv(r'C:\Users\Talha\Desktop\BOW.csv')


# In[ ]:


test_data_vect = pd.read_csv(r'C:\Users\Talha\Desktop\BOW_test.csv')


# In[ ]:


#converts the data into numpy arrays
#Also seperated the labels from features
X_train = train_data_vect.to_numpy()
X_train = X_train[:,2:2912]

Y_train = train_data_vect.to_numpy()
Y_train = Y_train[:,1]

X_test = test_data_vect.to_numpy()
X_test = X_test[:,2:2912]

Y_test = test_data_vect.to_numpy()
Y_test = Y_test[:,1]


# In[ ]:


#Calculate Euclidean Distances between testing set and training set
distance = cdist(X_test, X_train, 'euclidean')
#Sorts array indices
sorted_distance = np.argsort(distance)


# In[ ]:


K= [1, 3, 5, 7, 10]
accuracy=[]
precision=[]
recall=[]
F1=[]
#predicts label
predicted_label= []
for i in range(len(sorted_distance)):
        freq = []
        for j in range(K):
            freq.append(X_test.item(sorted_distance.item((i,j))))
            predicted_label.append(collections.Counter(freq).most_common(1)[0][0])
#finds macrovaerages and plots             
for k in K:
    
    print('k = ', k)
    
    precision_p =0
    precision_neg =0
    precision_neu = 0
    precision_ = 0
    
    recall_p = 0
    recall_neg = 0               
    recall_neu = 0
    recall_ = 0
    
    F1_=0
    
    accuracy_ = 0 
    

    for i in range(len(predicted_label)):
        
        df_confusion_matrix = pd.crosstab(Y_test[i],predicted_label[i])
        arr = np.df_confusion_matrix
                   
        
        precision_p = (arr[0,0])/(arr[0,0]+arr[0,1]+ar[0,2])
        precision_neg = (arr[1,1])/(arr[1,1]+arr[1,1]+arr[1,2])
        precision_neu = (arr[2,2])/(arr[2,2]+arr[2,1],arr[2,0])
        precision_ = (precision_p+precision_neg+precision_neu)/3
        print('Macroaverage Precision: ',macroaverage_precision)

        recall_p = (arr[0,0])/(arr[1,0]+arr[2,1]+arr[0,0])
        recall_neg = (arr[1,1])/(arr[1,1]+arr[2,1]+arr[0,1])               
        recall_neu = (arr[2,2])/(arr[2,2]+arr[1,2]+arr[0,2])
        recall_ = (recall_p+recall_neg+recall_neu)/3
        print('Macroaverage recall: ',macroaverage_recall)

        F1_=2/(1/macroaverage_precision+1/macroaverage_recall)
        print('F1: ',f1)

        accuracy = (arr[0,0]+arr[1,1]+arr[2,2])/(arr[0,0]+arr[1,1]+arr[2,2]+arr[0,1]+arr[1,0]+arr[1,2]+arr[2,1]) 
                   
        accuracy.append(accuracy_)
        precision.append(precision_)
        recall.append(recall_)
        F1.append(F1_)

plt.plot(k,accuracy_)
plt.plot(k,recall_)
plt.plot(k,precision_)
plt.plot(k,F1_)
plt.legend(('Accuracy','Recall','Precision','F1'))
plt.xlabel('K')
plt.ylabel('Results')

    



# In[ ]:


def extract_features(data):
    words = [word for word in data.split() if word in word2vec.vocab]
    result = np.mean(word2vec[words], axis=0)
    return result

training_word2vec = X_train.apply(lambda x: extract_features(x))
testing_word2vec = X_test.apply(lambda x: extract_features(x))


print('\nPart 3 for Part 1:\n')
    #Calculate Euclidean Distances between testing set and training set
    distance = cdist(testing_word2vec, training_word2vec, 'euclidean')
    #Sorts array indices
    sorted_distance = np.argsort(distance)
    

    K= [1, 3, 5, 7, 10]
    accuracy=[]
    precision=[]
    recall=[]
    F1=[]
    #predicts label
    predicted_label= []
        for i in range(len(sorted_distance)):
            freq = []
            for j in range(K):
                freq.append(labels.item(sorted_distance.item((i,j))))
                predicted_label.append(collections.Counter(votes).most_common(1)[0][0])
    #finds macrovaerages and plots             
    for k in K:
        
        print('k = ', k)
        
        precision_p =0
        precision_neg =0
        precision_neu = 0
        precision_ = 0
        
        recall_p = 0
        recall_neg = 0               
        recall_neu = 0
        recall_ = 0
        
        F1_=0
        
        accuracy_ = 0 
        
        for i in range(len(predicted_label):
            
            df_confusion_matrix = pd.crosstab(testing_word2vec[i],predicted_label[i])
            arr = np.df_confusion_matrix
                       
            
            precision_p = (arr[0,0])/(arr[0,0]+arr[0,1]+ar[0,2])
            precision_neg = (arr[1,1])/(arr[1,1]+arr[1,1]+arr[1,2])
            precision_neu = (arr[2,2])/(arr[2,2]+arr[2,1],arr[2,0])
            precision_ = (precision_p+precision_neg+precision_neu)/3
            print('Macroaverage Precision: ',macroaverage_precision)

            recall_p = (arr[0,0])/(arr[1,0]+arr[2,1]+arr[0,0])
            recall_neg = (arr[1,1])/(arr[1,1]+arr[2,1]+arr[0,1])               
            recall_neu = (arr[2,2])/(arr[2,2]+arr[1,2]+arr[0,2])
            recall_ = (recall_p+recall_neg+recall_neu)/3
            print('Macroaverage recall: ',macroaverage_recall)

            F1_=2/(1/macroaverage_precision+1/macroaverage_recall)
            print('F1: ',f1)

            accuracy = (arr[0,0]+arr[1,1]+arr[2,2])/(arr[0,0]+arr[1,1]+arr[2,2]+arr[0,1]+arr[1,0]+arr[1,2]+arr[2,1]) 
                       
            accuracy.append(accuracy_)
            precision.append(precision_)
            recall.append(recall_)
            F1.append(F1_)

    plt.plot(k,accuracy)
    plt.plot(k,recall)
    plt.plot(k,precision)
    plt.plot(k,F1)
    plt.legend(('Accuracy','Recall','Precision','F1'))
    plt.xlabel('K')
    plt.ylabel('Results')
    
        

