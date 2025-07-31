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
    vocaba =dict_df.to_csv(r'C:\Users\Talha\Desktop\Dictionary.csv', index_label="serial no")
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


# In[16]:


file =data_vect.to_csv(r'C:\Users\Talha\Desktop\BOW.csv', index_label="serial no") #creates the bow file


# In[17]:


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


# In[18]:


testing_data['Tweet'] = testing_data['Tweet'].apply(clean_tweets)


# In[19]:


testing_data.head()


# In[20]:


testing_data['Tokenized Text']=testing_data['Tweet'].apply(tweet_tokenize)


# In[21]:


testing_data.head()


# In[22]:


words = Counter()
for serial_no in testing_data.index:
    words.update(testing_data.loc[serial_no, "Tokenized Text"])
#words.most_common(4)


# In[23]:


for serial_no, word in enumerate(stop_words):
        del words[word]
#print(serial_no)        
#words.most_common(4)


# In[24]:


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


# In[25]:


file =test_data_vect.to_csv(r'C:\Users\Talha\Desktop\BOW_test.csv')


# In[26]:


train_data_vect = pd.read_csv(r'C:\Users\Talha\Desktop\BOW.csv')


# In[27]:


test_data_vect = pd.read_csv(r'C:\Users\Talha\Desktop\BOW_test.csv')


# In[28]:


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


K= [1, 3, 5, 7, 10]
accuracy=[]
precision=[]
recall=[]
F1=[]

for k in K:
    classifier = KNeighborsClassifier(n_neighbors = k,metric = 'minkowski',p=2)
    classifier.fit(X_train,Y_train)
    Y_predict = classifier.predict(X_test)
    print('k = ',k)
    
    accuracy_ = accuracy_score(Y_test,Y_predict)
    print('Accuracy= ', accuracy_)
    
    precision_ = precision_score(Y_test,Y_predict, average='macro')
    print('Precision= ', precision_)
    
    recall_ = recall_score(Y_test,Y_predict, average='macro')
    print('Recall= ', recall_) 
    
    F1_ = f1_score(Y_test,Y_predict, average='macro')
    print('F1= ', F1_)
    
    cmt = confusion_matrix(Y_test,Y_predict)
    print('Confusion Matrix= ', cmt)

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


# In[ ]:


def extract_features(data):
    words = [word for word in data.split() if word in word2vec.vocab]
    result = np.mean(word2vec[words], axis=0)
    return result

training_word2vec = X_train.apply(lambda x: extract_features(x))
testing_word2vec = X_test.apply(lambda x: extract_features(x))

print('\nPart 3 for Part 2:\n')
  K= [1, 3, 5, 7, 10]
    accuracy=[]
    precision=[]
    recall=[]
    F1=[]

    for k in K:
        classifier = KNeighborsClassifier(n_neighbors = k,metric = 'minkowski',p=2)
        classifier.fit(training_word2vec,Y_train)
        Y_predict = classifier.predict(testing_word2vec)
        print('k = ',k)
        
        accuracy_ = accuracy_score(Y_test,Y_predict)
        print('Accuracy= ', accuracy_)
        
        precision_ = precision_score(Y_test,Y_predict, average='macro')
        print('Precision= ', precision_)
        
        recall_ = recall_score(Y_test,Y_predict, average='macro')
        print('Recall= ', recall_) 
        
        F1_ = f1_score(Y_test,Y_predict, average='macro')
        print('F1= ', F1_)
        
        cmt = confusion_matrix(Y_test,Y_predict)
        print('Confusion Matrix= ', cmt)

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
    


