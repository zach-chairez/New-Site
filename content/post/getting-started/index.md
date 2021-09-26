---
title: Titanic Predictions
subtitle: The Titanic data set is a well known data set used in Kaggle
  challenges.  This blog post will discuss the tutorial provided by Kaggle for
  the Titanic data set, as well as some additional contributions using Logistic
  Regression.
date: 2020-12-13T00:00:00.000Z
summary: ""
draft: false
featured: false
authors:
  - admin
lastmod: 2020-12-13T00:00:00.000Z
tags: []
categories: []
projects: []
image:
  caption: ""
  focal_point: ""
  placement: 2
  preview_only: false
---
### All of the following code until the section marked "My Contribution" was taken from 
### https://www.kaggle.com/alexisbcook/titanic-tutorial for educational purposes.


```python
import numpy as np 
import pandas as pd 
```

The Titanic Data set is split into two separate data sets:  $\textbf{Training and Testing}$.  
The training data set contains 891 observations while the testing data set contains 418.  

The training data set contains the following information about each passenger: 

$
\textbf{Survived} \ \text{(categorical) - 0 (Did not survive) or 1 (Did survive)} \\
\textbf{Pclass} \ \text{(categorical) - 1 (1st Class), 2 (2nd Class), or 3 (3rd  Class)} \\ 
\textbf{Sex} \ \text{(categorical) -  Male or Female} \\ 
\textbf{Age} \ \text{(continuous) - Age of the passenger} \\ 
\textbf{Sibsp} \ \text{(categorical) - Number of siblings/spouses on board} \\ 
\textbf{Parch} \ \text{(categorical) - Number of parents/children on board} \\ 
\textbf{Ticket} \ \text{(categorical) - Ticket Number} \\ 
\textbf{Fare} \ \text{(continuous) - Price of ticket} \\ 
\textbf{Cabin} \ \text{(categorical) - Cabin Number} \\ 
\textbf{Embarked} \ \text{(categorical) - Port of Embarkation, C = Cherbourg, Q = Queenstown, S = Southampton.} \\ 
$

The testing data set contains all of the same information $\textbf{without the Survived variable}$


```python
# Loading the training and testing data sets.  
train_data = pd.read_csv(r"C:\Users\zachc\OneDrive\Desktop\train.csv")
test_data = pd.read_csv(r"C:\Users\zachc\OneDrive\Desktop\test.csv")
```


```python
# Looking at the percentage of passengers who were women that survived
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)
```

    % of women who survived: 0.7420382165605095
    


```python
# Looking at the percentage of passengers who were men that survived
men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)
```

    % of men who survived: 0.18890814558058924
    

By observation, we see that the rate at which women survived the sinking of the Titanic is much higher than that of the men.  This is a good indicator that sex may be a useful predictor for determining survival of passengers.  We'll construct a Random Forest Classifier utilizing the predictor variables $\textbf{Pclass, Sex, SibSp, and Parch}$


```python
# Using a random forest to create a classifier for survival on the Titanic
from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('Tutorial_Submission.csv', index=False)
# print("Your submission was successfully saved!")
```

Once the predictions were made, they were exported to a csv file named "Tutorial Submission" and they were submitted to the Titanic Kaggle challenge found here:  https://www.kaggle.com/c/titanic.  
The classifier had a 77.511% accuracy.  

# My Contribution
In this section, we'll conduct a Logistic Regression to create a model which will act as our classifier.

## Before we begin, we'll parse through our training data set to see if there exists any missing values and what variables we'll use for training and testing.  


```python
# Importing some necessary packages
import matplotlib.pyplot as plt

# Checking to see what features in the training data set contain missing values.
train_data.isnull().sum()
```




    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64




```python
test_data.isnull().sum()
```




    PassengerId      0
    Pclass           0
    Name             0
    Sex              0
    Age             86
    SibSp            0
    Parch            0
    Ticket           0
    Fare             1
    Cabin          327
    Embarked         0
    dtype: int64



The training and testing data set have missing values.  In the traininig data set, Age, Cabin, and Embarked have missing values.  In the testing data set, Age, Cabin, and Fare having missing values.  We'll now look at the correlation between each variable and most importantly, the passenger survival.


```python
train_data.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PassengerId</th>
      <td>1.000000</td>
      <td>-0.005007</td>
      <td>-0.035144</td>
      <td>0.036847</td>
      <td>-0.057527</td>
      <td>-0.001652</td>
      <td>0.012658</td>
    </tr>
    <tr>
      <th>Survived</th>
      <td>-0.005007</td>
      <td>1.000000</td>
      <td>-0.338481</td>
      <td>-0.077221</td>
      <td>-0.035322</td>
      <td>0.081629</td>
      <td>0.257307</td>
    </tr>
    <tr>
      <th>Pclass</th>
      <td>-0.035144</td>
      <td>-0.338481</td>
      <td>1.000000</td>
      <td>-0.369226</td>
      <td>0.083081</td>
      <td>0.018443</td>
      <td>-0.549500</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>0.036847</td>
      <td>-0.077221</td>
      <td>-0.369226</td>
      <td>1.000000</td>
      <td>-0.308247</td>
      <td>-0.189119</td>
      <td>0.096067</td>
    </tr>
    <tr>
      <th>SibSp</th>
      <td>-0.057527</td>
      <td>-0.035322</td>
      <td>0.083081</td>
      <td>-0.308247</td>
      <td>1.000000</td>
      <td>0.414838</td>
      <td>0.159651</td>
    </tr>
    <tr>
      <th>Parch</th>
      <td>-0.001652</td>
      <td>0.081629</td>
      <td>0.018443</td>
      <td>-0.189119</td>
      <td>0.414838</td>
      <td>1.000000</td>
      <td>0.216225</td>
    </tr>
    <tr>
      <th>Fare</th>
      <td>0.012658</td>
      <td>0.257307</td>
      <td>-0.549500</td>
      <td>0.096067</td>
      <td>0.159651</td>
      <td>0.216225</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



By observation, we see small correlations between Survived and Pclass and Fare.  Of course, Pclass and Fare have a higher correlation.  The higher the class of the passenger, the more they spent on their ticket.  We also see some correlation between the age of the passenger and Pclass.  We can suggest that the older the passenger was, the higher in class they were.  For this first part, we'll simply find the median and mean of Fare for the testing data set and fill in the missing value to the training data set with either the median or mean depending on the variables' distribution.      


```python
# Checking the distribution of Fare.  
fare_test = test_data["Fare"]; 
plt.hist(fare,bins = 15);
```


![png](output_16_0.png)


The distribution of Fare is skewed to the right, so we'll use the median to replace the missing value in the testing data set. 


```python
# # The distribution of Fare is skewed the right so we'll use the median value to replace the 
# missing Fare value for our testing data.
fare_med = fare_test.median(); 
fare_test = fare_test.fillna(fare_med)
```

Now that we've filled in the missing values in the testing data set, we'll begin training our classifer.  We'll use Logistic Regression model as our classifier with the following predictors: $\textbf{Pclass, Sex, Sibsp, Parch, Fare}$.  


```python
from sklearn.linear_model import LogisticRegression 

fare_train = pd.DataFrame(train_data["Fare"])
fare_test = pd.DataFrame(fare_test)

# Adding the variable "Fare" to the set of features for training and testing.
X_train_new = pd.concat([X,fare_train],axis = 1)
X_test_new = pd.concat([X_test,fare_test],axis = 1)

# Creating our model based on training data with and without penalty.
# The penalty term is using l2 regularization.
# The l2 regularizer helps with potential dangers caused by outliers in our predictor variables. 
lr = LogisticRegression()
lrl2 = LogisticRegression(penalty = 'l2') 
trained_model_np = lr.fit(X_train_new,y)
trained_model_l2 = lrl2.fit(X_train_new,y)

# Making predictions on testing data.
test_predict_np = lr.predict(X_test_new)
test_predict_l2 = lrl2.predict(X_test_new)

# Exporting necessary columns to a CSV file to submit to Kaggle.
my_contr_np = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': test_predict_np})
my_contr_l2 = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': test_predict_l2})

# Saving predictions in CSV files (NP = No Penalty, l2 = L2 Regularization Used)
my_contr_np.to_csv('My_Contribution_NP.csv', index = False )
my_contr_np.to_csv('My_Contribution_l2.csv', index = False )
```

Both classifiers ended up performing the same with a prediction accuracy of 76.794%, which was less accurate than the initial classifier.  We'll further investigate and utilize additional variables in our new classifier.  We'll continue to use Logistic Regression, but this time, we'll implement the use of Age in our classifer.  


```python
# Let's try something new
# fare = train_data["Fare"]; surv = train_data["Survived"]; pcl = train_data['Pclass']; 
# sex = train_data["Sex"]; sibsp = train_data["SibSp"]; parch = train_data["Parch"]; 
# emb = train_data["Embarked"]; 

X_train = pd.get_dummies(train_data[["Pclass","Sex"]])
X_test = pd.get_dummies(test_data[["Pclass","Sex"]])

# X_train_new = pd.concat([X_train,fare_train],axis = 1)
# X_test_new = pd.concat([X_test,fare_test],axis = 1)

lr = LogisticRegression()
trained_model_np = lr.fit(X_train,y)
pred = lr.predict(X_test)

my_contr_np = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': pred})
my_contr_np.to_csv('My_Contribution_NP_LessV3.csv', index= False )
```
