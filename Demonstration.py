
# coding: utf-8

# # Demonstration Machine Learnin in Python
# 
# Wir nutzen hierfür den [Titanic Daten Satz von Kaggle](https://www.kaggle.com/c/titanic).
# Dies ist ein recht berühmter Datensatz.
# 
# Wir versuchen hier eine Klassifikation der Passagiere der Titanic nach "survival" oder "deceased".
# 
# Wir nutzen hier eine nicht vollständig bereinigte Version, das bedeutet wir müssen die Daten noch vollständig bereinigen.

# # Einbinden der Standardbibliotheken für Datenvisualisierung

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[2]:

from matplotlib import rcParams
rcParams['patch.force_edgecolor'] = True
rcParams['patch.facecolor'] = 'b'


# In[3]:

df_train = pd.read_csv('titanic_train.csv')
df_train.head()


# In[4]:

df_train.info()


# # Datenvisualisierung

# In[5]:

sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In dem Bild oben sehen wir fehlende Angaben. Da die Spalte 'Cabin' nur sehr sporadisch gefüllt ist werden wir diese vor dem Nutzen entfernen oder eine andere Spalte daraus erzeugen : 'Cabin known' mit den Werten 0 oder 1

# In[6]:

sns.set_style('whitegrid')
sns.countplot(x='Survived',data=df_train,palette='RdBu_r')


# In[7]:

sns.countplot(x='Survived',hue='Sex',data=df_train,palette='RdBu_r')


# In[8]:

sns.countplot(x='Survived',hue='Pclass',data=df_train,palette='rainbow')


# In[9]:

sns.distplot(df_train['Age'].dropna(),kde=False,color='darkred',bins=30)


# # Bereinigung der Daten

# Wir könnten die 'Age' Spalte raus nehmen oder den mittelwert des Alters der Passagiere an den fehlenden stellen eintragen. Es geht aber noch besser. Wir setzen den mittelwert der entsprechenden 'PClass' Passagierklasse ein. Ob und wie sich das Alter abhängig von der Passagierklasse unterscheidet sehen wir in der nachfolgenden Grafik.

# In[10]:

plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=df_train,palette='winter')


# In[11]:

def add_average_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


# In[12]:

df_train['Age'] = df_train[['Age','Pclass']].apply(add_average_age,axis=1)


# Anschließend überprüfen wir die Daten erneut

# In[13]:

sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# Nun löschen für die Spalte 'Cabin' und auch den einen Null-Wert der Spalte 'Embarked'

# In[14]:

df_train.drop('Cabin',axis=1,inplace=True)
df_train.dropna(inplace=True)


# In[15]:

sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[16]:

df_train.head()


# # Datenaufbereitung für Machine Learning Algorithmen
# 
# um nun Spalten wie 'Sex' und 'Embarked' nutzbar zu machen müssen die kategorischen Werte zum Beispiel 'male' und 'female' in Zahlen zum Beispiel 0 und 1 konvertiert werden. Python bzw. Pandas bietet hierfür eine Funktion um Dummy Parameter bzw. Spalten aus diesen kategorischen Werten zu erzeugen. 

# In[17]:

sex = pd.get_dummies(df_train['Sex'],drop_first=True)
embark = pd.get_dummies(df_train['Embarked'],drop_first=True)


# In[18]:

df_train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
df_train = pd.concat([df_train,sex,embark],axis=1)


# In[19]:

df_train.head()


# # Model konstruktion und Evaluation
# nun da alle Spalten der Trainingsdaten nur noch Zahlenwerte beinhalten können wir nun mit der Erzeugung verschiedener Modelle beginnen und diese anschließend evaluieren

# # Entscheidungsbaum Model

# In[20]:

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix


# In[21]:

X = df_train.drop('Survived',axis=1)
y = df_train['Survived']


# In[22]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# In[23]:

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


# In[24]:

predictions = dtree.predict(X_test)


# In[25]:

print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))


# # RandomForest Model

# In[26]:

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)


# In[27]:

rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test,rfc_pred))
print('\n')
print(classification_report(y_test,rfc_pred))


# # Support Vektor Machines

# In[28]:

from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)


# In[29]:

predictions = model.predict(X_test)
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))


# # Verbesserter Ansatz für Support Vector Machines
# 
# Hier werden nun für Support Vector Machines einige Parameter durchgespielt um den besten Bias-Variance Trade-off herauszufinden. Dies kann beliebig granular durchgeführt werden, dauert dann aber entsprechend länger.

# In[30]:

from sklearn.model_selection import GridSearchCV


# In[31]:

param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['linear']} 
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)


# In[32]:

# May take awhile!
# grid.fit(X_train,y_train)


# In[33]:

# grid.best_params_


# In[34]:

# grid_predictions = grid.predict(X_test)
# print(confusion_matrix(y_test,grid_predictions))
# print('\n')
# print(classification_report(y_test,grid_predictions))


# In[ ]:



