import numpy as np 
import pandas as pd
import os
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from nltk.tokenize import sent_tokenize, word_tokenize
from scipy import stats
import seaborn as sns; sns.set()
import warnings
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.utils import shuffle
warnings.filterwarnings(action = 'ignore') 
import gensim 
from gensim.models import Word2Vec



data=pd.read_csv('RNA_Train.csv')
X,y=data['Sequence'],data['label']
X=np.array(X)
y=np.array(y)
#X, y = shuffle(X, y)

#find all the unique characters in sequence strings of RNA_Train.csv
set1=set()
for i in range(len(X)):
  for ch in X[i]:
    set1.add(ch)
unique_characters=[]
for ch in set1:
  unique_characters.append(ch)
unique_characters.sort()          #yeh h list of unique protiens

listt = []
for i in unique_characters:
  temp_list = []
  for j in range(len(unique_characters)):
    if (j == unique_characters.index(i)):
      temp_list.append(1)
    else:
      temp_list.append(0)
  listt.append(temp_list)

#converting each sequence string to numerical values (binary)
processed_data=[]
for st in X:
  temp_data=[]
  for i in range(len(st)):
    temp_data.append(listt[unique_characters.index(st[i])])
  processed_data.append(temp_data)

processed_data=np.array(processed_data)
processed_data = np.reshape(processed_data, (330862, 357))        #binary conversion done


#preprocess test.csv
test_data=pd.read_csv('test.csv')
X=test_data['Sequence']
X=np.array(X)
processed_test_data=[]
for st in X:
  temp_test_data=[]
  for i in range(len(st)):
    temp_test_data.append(listt[unique_characters.index(st[i])])
  processed_test_data.append(temp_test_data)

processed_test_data=np.array(processed_test_data)
processed_test_data=np.reshape(processed_test_data, (6276, 357))


from imblearn.over_sampling import RandomOverSampler as ROS 
X_resample, y_resample = ROS(sampling_strategy='minority').fit_resample(processed_data, y)

#finally predict on processed_test_data
clf = RandomForestClassifier(n_estimators=450)
print("hello")
clf.fit(X_resample, y_resample)
test_prediction=clf.predict(processed_test_data)


out_data=pd.DataFrame(data=test_prediction,index=test_data.iloc[:,0])
out_data.rename(columns={0:'label'},inplace=True)
out_data.to_csv('output.csv')

