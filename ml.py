import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df=pd.read_csv(r"C:\Users\aksmk\OneDrive\Desktop\DSA\Datasets\Social_Network_Ads.csv")
df.drop('User ID',axis=1,inplace=True)

le=LabelEncoder()
df.Gender=le.fit_transform(df.Gender)

x=df.drop('Purchased',axis=1)
y=df.Purchased

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
knn=KNeighborsClassifier(n_neighbors=5,metric='euclidean')
knn.fit(x_train,y_train)

import pickle
with open('model.pkl','wb') as model_file:
  pickle.dump(knn,model_file)