# -*- coding: utf-8 -*-
"""
Created on Thu May 12 15:31:51 2022

@author: Sai pranay
"""
#-----------------------IMPORTING_THE_DATA_SET---------------------------------

import pandas as pd
dt_cd = pd.read_csv("E:\\DATA_SCIENCE_ASS\\DECISION TREE\\Company_Data.csv")
print(dt_cd)
list(dt_cd)
dt_cd.shape
dt_cd.info()
dt_cd.describe()

#-----------------working_on_label_encoding------------------------------------

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
dt_cd['ShelveLoc_1']=LE.fit_transform(dt_cd['ShelveLoc'])
dt_cd['Urban_1']=LE.fit_transform(dt_cd['Urban'])
dt_cd['US_1']=LE.fit_transform(dt_cd['US'])

print(dt_cd)
list(dt_cd)

#-----------------------seaborn plotting---------------------------------------
import seaborn as sns

sns.distplot(dt_cd['CompPrice'])
sns.distplot(dt_cd['Income'])
sns.countplot(dt_cd['Price'])
sns.countplot(dt_cd['Age'])
sns.countplot(dt_cd['Education'])
sns.countplot(dt_cd['Urban'])
sns.countplot(dt_cd['ShelveLoc'])
sns.countplot(dt_cd['Population'])
sns.countplot(dt_cd['Advertising'])


#----------------------------------DROPPING_-----------------------------------

dt_cd_new = dt_cd.drop(['ShelveLoc','Urban','US'],axis=1)
print(dt_cd_new)
dt_cd_new.shape
dt_cd_new.info()
list(dt_cd_new)


dt_cd_new_1 = dt_cd_new.iloc[:,0:8]
dt_cd_new_1


dt_cd_new_le = dt_cd_new.iloc[:,8:11]
dt_cd_new_le




#---------------------------------Standardization------------------------------

from sklearn.preprocessing import StandardScaler
st=StandardScaler()
x1=st.fit_transform(dt_cd_new_1)
x1
x11=pd.DataFrame(x1)
x11
list(x11)



dt_cd_r = pd.concat([x11,dt_cd_new_le],axis = 1)
dt_cd_r




#-----------------------------------splitting----------------------------------

x = dt_cd_r.iloc[:,1:11]
print(x)
x.shape
list(x)


y = dt_cd_r[0]
print(y)
y.shape
list(y)




Y1=[]
for i in range(0,400,1):
    if y.iloc[i,]>=y.mean():
        print('High')
        Y1.append('High')
    else:
        print('Low')
        Y1.append('Low')

Y2 = pd.DataFrame(Y1)
Y2

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, Y2, test_size=0.25,stratify=Y2,random_state=91)

X_train.shape
X_test.shape
Y_train.shape
Y_test.shape



# --------------------------fit the model _via (criterion='entropy')-----------
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion='entropy',max_depth=8)
DT.fit(X_train,Y_train)

Y_pred = DT.predict(X_test)
Y_pred

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y_test,Y_pred)
cm
ac = accuracy_score(Y_test,Y_pred)
ac

#--------------------- fit the model _via (criterion='gini')-------------------
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion='gini')
DT.fit(X_train,Y_train)

Y_pred = DT.predict(X_test)
Y_pred

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y_test,Y_pred)
cm
ac = accuracy_score(Y_test,Y_pred)
ac




from sklearn import tree
import graphviz 
dot_data = tree.export_graphviz(DT, out_file=None, 
                    filled=True, rounded=True,  
                    special_characters=True)  
graph = graphviz.Source(dot_data)  
graph


#Tree
import matplotlib.pyplot as plt
from sklearn import tree
tr=tree.plot_tree(DT,filled=True,fontsize=6)

DT.tree_.node_count
DT.tree_.max_depth
