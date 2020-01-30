#simple linear regression program


#copy paste the data preprocessing template

#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv('Salary_Data.csv')

#matrix of features(x) and dep vars(y)
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values


#Missing values
''' These are deprecated methods that will be removed soon
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN", strategy="mean", axis=0)
'''
'''Not needed for this dataset
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
'''

'''Not needed for this dataset
#Categorical Labeling

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:,0] = labelEncoder_X.fit_transform(X[:,0])
oneHotEncoder_X = OneHotEncoder(categorical_features=[0])
X = oneHotEncoder_X.fit_transform(X).toarray()

labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)
'''

#train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=0)

'''Not needed for this dataset
#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
'''

#import library
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#fit the model
regressor.fit(X_train, y_train)

#test against new y values
y_pred = regressor.predict(X_test)

#plotting shit
plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Experience vs Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#plotting other stuff
plt.scatter(X_test , y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Experience vs Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()







