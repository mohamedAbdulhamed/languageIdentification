# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import re
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from kmeans import Kmeans
# Data Set Path

path = "dataset.csv"
languages = pd.read_csv(path)

print(languages.head(20))



lang_labels = languages['language']
lang_labels[0:10]
lang_labels2 = ['English' ,'Arabic','Hindi','Russian','Persian', 'Thai'] # 6 lang

# ['Estonian' 'Swedish' 'Thai' 'Tamil' 'Dutch' 'Japanese' 'Turkish' 'Latin'
#  'Urdu' 'Indonesian' 'Portugese' 'French' 'Chinese' 'Korean' 'Hindi'
#  'Spanish' 'Pushto' 'Persian' 'Romanian' 'Russian' 'English' 'Arabic']


languages=languages.iloc[:6000,:]
Labels = np.array(languages.iloc[:, 1:]) # 22 Languages
Sequences = np.array(languages.iloc[:, 0:1]) # Sentences
X = languages["Text"]
y = languages["language"]
print (len(Labels))
print (len(Sequences))

        
  
languagesToRead = []
for text in X:
       
        text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
        text = re.sub(r'[[]]', ' ', text)
        text = text.lower().split()
        languagesToRead.append(text)


valArray=[]
for i in range(len(languagesToRead)) :            
    le =len(languagesToRead[i])
    valArray.append(le)
A = np.array(valArray)

languagesToRead[1046]  
            
# temp=[]
# tem=[]
# for i in range(len(languagesToRead)):
#     for j in range(len(languagesToRead[i])):
#         word=languagesToRead[i][j]
#         if len(word)>1:
#             word=(ord(*word[0]) + ord(*word[1])) 
#             tem.append(word)
#         else:
#             word=(ord(*word[0]) + ord(*word[0])) 
#             tem.append(word)  
#     temp.append(tem)
#     tem=[]

temp=[]
tem=[]
maximum=50
word1 =1
c=0
word=''
for i in range(len(languagesToRead)):
    c=0
    for j in range(maximum):
        
        if c < len(languagesToRead[i]):
            word=languagesToRead[i][j]
            c+=1
        else:
            word=''
        if len(word)>1:
            word=(ord(*word[0]) + ord(*word[1])) 
            word1=word
            tem.append(word)
        elif len(word)==1 :
            word=(ord(*word[0]) + ord(*word[0])) 
            tem.append(word)
        elif len(word)==0 :
            tem.append(word1)                    
    temp.append(tem)
    tem=[]
    
word=(ord('ø') + ord("ù"))     
len(languagesToRead[1] )   
len(temp[1000])
temp[10]
temp1=[]
word1 =1
c1=0
word1=''
ar='مصدر تنفعل المتحكم يبحث'.split()    
for j in range(maximum):
    if c1 < len(ar):
        word=ar[c1]
        c1+=1
    else:
        word=''
    if len(word)>1:
        word=(ord(*word[0]) + ord(*word[1])) 
        word1=word
        temp1.append(word)
    elif len(word)==1 :
        word=(ord(*word[0]) + ord(*word[0])) 
        temp1.append(word)
    elif len(word)==0 :
        temp1.append(word1)
    
      

      
df=pd.DataFrame(temp)
df.isnull()
#df.drop(axis=0,index=dropedCol)           
ms = MinMaxScaler()
X = ms.fit_transform(df)
    
df1=pd.DataFrame(temp1)
df1.isnull()
#df.drop(axis=0,index=dropedCol)           
ms = MinMaxScaler()
X2 = ms.fit_transform(df1)



n_clusters = 4
sklearn_pca = PCA(n_components = 1)
Y_sklearn = sklearn_pca.fit_transform(X)
Y_sklearn[:20]
test1_sklearn = sklearn_pca.fit_transform(X2)



km = Kmeans(k=n_clusters,max_iter=400)
# kmeans = KMeans(n_clusters= n_clusters, max_iter=100,init='k-means++', algorithm = 'auto')
fitted = km.fit_kmeans(Y_sklearn)
#number of each language
prediction = km.predict(Y_sklearn)
prediction[:15]

Y_sklearn[0:10]

plt.scatter(Y_sklearn[:,], Y_sklearn[:, ],c=prediction ,s=50, cmap='viridis')





