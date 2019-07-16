#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''Problem statement :  Design a system that predicts the fare amount for a cab ride in the city.
According to CRISP DM Process this problem statement lies in the category of forecasting which deals with
predicting continuous value for future(in our case the continuous value is the fare amount of the cab ride.)'''


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fancyimpute import KNN
import datetime
import itertools
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from math import sqrt
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from IPython import get_ipython


# In[3]:


cab_df=pd.read_csv("train_cab.csv")



# In[4]:


cab_df.info()


# In[5]:


'''Data Type conversion : So that pickup_datetime and fare_amount get converted to proper datatypes.'''


# In[6]:


cab_df['fare_amount']=pd.to_numeric(cab_df['fare_amount'],errors='coerce')


# In[7]:


cab_df['pickup_datetime']=pd.to_datetime(cab_df['pickup_datetime'],errors='coerce')


# In[8]:


cab_df.dtypes


# In[9]:


cab_df[(cab_df['pickup_datetime'].isna()==True)]


# In[10]:


#As there is only 1 inappropriate format of datetime so deleting it
cab_df = cab_df[cab_df.pickup_datetime.notnull()] 


# In[11]:


missing_val=pd.DataFrame(cab_df.isnull().sum())
missing_val=missing_val.reset_index()
missing_val


# In[12]:


missing_val=missing_val.rename(columns={'index':'variables',0:'missing_percentage'})
missing_val['missing_percentage']=(missing_val['missing_percentage']/len(cab_df))*100
missing_val=missing_val.sort_values(by='missing_percentage',ascending=False).reset_index(drop=True)
missing_val


# In[13]:


cab_df[(cab_df['fare_amount'].isna()==True)]


# In[14]:


cab_df = cab_df[cab_df.fare_amount.notnull()] 


# In[15]:


cab_df.shape


# In[16]:


missing_val=pd.DataFrame(cab_df.isnull().sum())
missing_val


# In[17]:


#Converting pickup_datetime back to numeric as for knn imputation all variables must be of numeric type
cab_df['pickup_datetime']=pd.to_numeric(cab_df['pickup_datetime'])


# In[18]:


'''passenger_count has missing values< 30% so it needs to be imputed.
Actual Value of 70th passenger: 2.0
mean value: 2.62
median value: 1.0
KNN: 2.0
Freeze knn for passenger_count''' 
cab_df['passenger_count'].loc[70] = np.nan


# In[19]:


cab_df.iloc[65:75,:]


# In[20]:


#cab_df['passenger_count']= cab_df['passenger_count'].fillna(cab_df['passenger_count'].mean())
#cab_df['passenger_count'].loc[70]


# In[21]:


#cab_df['passenger_count']= cab_df['passenger_count'].fillna(cab_df['passenger_count'].median())
#cab_df['passenger_count'].loc[70]


# In[22]:


cab_df=pd.DataFrame(KNN(k=3).fit_transform(cab_df),columns=cab_df.columns)
cab_df['passenger_count'].loc[70]


# In[23]:


cab_df.isnull().sum()


# In[24]:


#Converting datetime back to its original datatype
cab_df['pickup_datetime']=pd.to_datetime( pd.to_numeric( pd.to_datetime( cab_df['pickup_datetime'], origin = '1970-01-01' ) ), 
                                     origin = '1970-01-01')


# In[25]:


cab_df.head()


# In[26]:


sns.distplot(cab_df['pickup_longitude'],kde=True)


# In[27]:


sns.distplot(cab_df['pickup_latitude'],kde=True)


# In[28]:


cab_df.describe()


# In[29]:


'''There are few strange values in the data like :
minimum fare_amount is -3, fare can never be negative.
pickup_latitude max value is out of range.
minimum passenger_count is 0 which can't be the case.
So to clean these values we detect and remove the outliers from our data.'''


# In[30]:


num_variables=['fare_amount','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count']
for i in num_variables:
    cab_df.boxplot(column=i)
    plt.show()



# In[33]:


for i in num_variables:
    q75,q25=np.percentile(cab_df.loc[:,i],[75,25])
    iqr=q75-q25
    min=q25 -(1.5*iqr)
    max=q75 +(1.5*iqr)
    
    cab_df=cab_df.drop(cab_df[cab_df.loc[:,i]<min].index)
    cab_df=cab_df.drop(cab_df[cab_df.loc[:,i]>max].index)


# In[34]:


#Latitude : 40.730610, Longitude : -73.935242
cab_df.describe()


# In[35]:


cab_df = cab_df[(cab_df['passenger_count']>= 1)]
cab_df = cab_df[(cab_df['fare_amount']>=1)]
cab_df.describe()


# In[36]:


cab_df['passenger_count'] = round(cab_df['passenger_count'])
cab_df.describe()


# In[37]:


f, ax=plt.subplots(figsize=(7,5))

sns.heatmap(cab_df.corr(),mask=np.zeros_like(cab_df.corr(),dtype=np.bool),
           cmap=sns.diverging_palette(220,10,as_cmap=True),ax=ax,annot = True)


# In[38]:


#Feature Engineering : Now after missing value and outlier analysis, we extract some new features from existing ones, which
#can help further in analysis.


# In[39]:


cab_df['year'] = cab_df.pickup_datetime.dt.year
cab_df['month'] = cab_df.pickup_datetime.dt.month
cab_df['day'] = cab_df.pickup_datetime.dt.day
cab_df['weekday'] = cab_df.pickup_datetime.dt.weekday
cab_df['hour'] = cab_df.pickup_datetime.dt.hour


# In[40]:


del cab_df['pickup_datetime']


# In[41]:


def haversine_distance(lat1, long1, lat2, long2):
    data = [cab_df]
    for i in data:
        R = 6371  #radius of earth in kilometers
        #R = 3959 #radius of earth in miles
        phi1 = np.radians(i[lat1])
        phi2 = np.radians(i[lat2])
    
        delta_phi = np.radians(i[lat2]-i[lat1])
        delta_lambda = np.radians(i[long2]-i[long1])
    
        #a = sin²((φB - φA)/2) + cos φA . cos φB . sin²((λB - λA)/2)
        a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    
        #c = 2 * atan2( √a, √(1−a) )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
        #d = R*c
        d = (R * c) #in kilometers
        i['distance_km'] = d
    return d


# In[42]:


cab_df['distance_km'] = haversine_distance('pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')
cab_df.head()


# In[43]:


del cab_df['pickup_longitude']
del cab_df['pickup_latitude']
del cab_df['dropoff_longitude']
del cab_df['dropoff_latitude']
cab_df.head()


# In[44]:


f, ax=plt.subplots(figsize=(7,5))

sns.heatmap(cab_df.corr(),mask=np.zeros_like(cab_df.corr(),dtype=np.bool),
           cmap=sns.diverging_palette(220,10,as_cmap=True),ax=ax,annot=True)



# In[45]:


#Exploratory Data Analysis : to understand the data and make some inferences out of it.


# In[46]:


#Shows the distribution b/w distance and fare
plt.scatter(cab_df['distance_km'], cab_df['fare_amount'], color='red')

plt.title('distance vs fare_amount', fontsize=14)

plt.xlabel('distance', fontsize=14)

plt.ylabel('fare', fontsize=14)

plt.grid(True)

plt.show()


# In[47]:


#shows how the different days have different fare amount
plt.figure(figsize=(15,7))
plt.scatter(x=cab_df['day'], y=cab_df['fare_amount'], s=1.5)
plt.title('day vs fare_amount', fontsize=14)
plt.xlabel('day')
plt.ylabel('Fare')


# In[48]:


#shows the frequency of hours, (tells us about the most active hour)
plt.figure(figsize=(15,7))
plt.hist(cab_df['hour'], bins=100)
plt.title('Frequency of hours')
plt.xlabel('Hour')
plt.ylabel('Frequency')


# In[49]:


#Hour vs Fare_amount
plt.figure(figsize=(15,7))
plt.scatter(x=cab_df['hour'], y=cab_df['fare_amount'], s=1.5)
plt.xlabel('Hour')
plt.ylabel('Fare')


# In[50]:


#shows the frequency of number of passengers
plt.figure(figsize=(15,7))
plt.hist(cab_df['passenger_count'], bins=15)
plt.title('Passenger Count')
plt.xlabel('No. of Passengers')
plt.ylabel('Frequency')


# In[51]:


#Number of Passengers vs Fare_amount
plt.figure(figsize=(15,7))
plt.scatter(x=cab_df['passenger_count'], y=cab_df['fare_amount'], s=1.5)
plt.xlabel('No. of Passengers')
plt.ylabel('Fare')


# In[52]:


#Shows the average fare_amount by Month
fare_mn = cab_df.groupby("month")["fare_amount"].mean().reset_index()
plt.figure(figsize = (10,5))
sns.barplot("month","fare_amount",
            data = fare_mn,
            linewidth =1)
plt.grid(True)
plt.title("Average fare amount by Month")
plt.show()


# In[53]:


cab_df['year'].unique()


# In[54]:


#shows the trends of trips every month of all years except for 2015 as it doesn't have data for all months

yrs = [i for i in cab_df["year"].unique().tolist() if i not in [2015]]

#subset data without year 2015
complete_dat = cab_df[cab_df["year"].isin(yrs)]


plt.figure(figsize = (13,15))
for i,j in itertools.zip_longest(yrs,range(len(yrs))) :
    plt.subplot(3,2,j+1)
    trip_counts_mn = complete_dat[complete_dat["year"] == i]["month"].value_counts()
    trip_counts_mn = trip_counts_mn.reset_index()
    sns.barplot(trip_counts_mn["index"],trip_counts_mn["month"],
                palette = "rainbow",linewidth = 1,
                edgecolor = "k"*complete_dat["month"].nunique() 
               )
    plt.title(i,color = "b",fontsize = 12)
    plt.grid(True)
    plt.xlabel("")
    plt.ylabel("trips")


# In[55]:

del cab_df['month']

del cab_df['year']



#Train_test Splitting : Simple Random Sampling as we are dealing with continuous variables
X = cab_df.iloc[:,1:].values
Y = cab_df.iloc[:,0].values
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state = 0)


# In[56]:


#Multiple Linear Regression
model = sm.OLS(y_train,X_train).fit()
model.summary()


# In[60]:


predictions_LR = model.predict(X_test)

def MAPE(y_actual,y_pred):
    mape = np.mean(np.abs((y_actual - y_pred)/y_actual))
    return mape

error = MAPE(y_test,predictions_LR)*100


import pickle
pickle.dump(model,open("model.pkl",'wb'))

#Loading model
model = pickle.load(open("model.pkl",'rb'))

#accuracy : 80.17, rsq : 93.4
# In[61]:


error = sqrt(metrics.mean_squared_error(y_test,predictions_LR)) #calculate rmse
print('RMSE value for Multiple Linear Regression is:', error)


# In[62]:


#Decision Tree
train,test = train_test_split(cab_df,test_size = 0.2,random_state=0)
fit = DecisionTreeRegressor(max_depth=5).fit(train.iloc[:,1:],train.iloc[:,0])
predictions_DT = fit.predict(test.iloc[:,1:])
predictions_DT


# In[63]:


error = sqrt(metrics.mean_squared_error(y_test,predictions_DT)) #calculate rmse
print('R square value for Decision Tree is: ',metrics.r2_score(y_test,predictions_DT))
print('RMSE value for Decision Tree is:', error)


# In[64]:


#Random Forest
X = cab_df.iloc[:,1:].values
Y = cab_df.iloc[:,0].values
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test) 


# In[65]:


regressor = RandomForestRegressor(n_estimators=200, random_state=0)  
regressor.fit(X_train, y_train)  
y_pred = regressor.predict(X_test)  
y_pred
print('R square value for Random Forest is: ',metrics.r2_score(y_test,y_pred))
print('RMSE value for Random Forest is :', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 


# In[66]:


#K Nearest Neighbours
train , test = train_test_split(cab_df, test_size = 0.2)

x_train = train.drop('fare_amount', axis=1)
y_train = train['fare_amount']

x_test = test.drop('fare_amount', axis = 1)
y_test = test['fare_amount']


# In[67]:


scaler = MinMaxScaler(feature_range=(0, 1))

x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)

x_test_scaled = scaler.fit_transform(x_test)
x_test = pd.DataFrame(x_test_scaled)


# In[68]:


#import required packages
get_ipython()


# In[69]:


model = neighbors.KNeighborsRegressor(n_neighbors = 10)
model.fit(x_train, y_train)  #fit the model
pred=model.predict(x_test) #make prediction on test set
error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse

print('R square value for K Nearest Neighbours is: ',metrics.r2_score(y_test,pred))
print('RMSE value for K Nearest Neighbours is:', error)


# In[70]:


'''
Error metric used is Root Mean Square Error as this is a Time Series Forecasting Problem. 
It represents the sample standard deviation of the differences between predicted values and observed values (called residuals).
Lower RMSE mean better model performance.
RMSE -->
MLR : 2.16
DT : 2.06
RF : 2.04
KNN : 2.43

Best Linear Regression as highest r2 = 94.3
'''
print('Highest accuracy : 94.3 --> Linear Regression, so using this model to predict')


# In[71]:


#Now that we have chosen the right model, we use it to predict fare_Amount for our test cases.
#For that we first preprocess the test dataset and make it appropriate in such a way that it fits the model i.e the 
#input variables be same as the input variables of the algorithm chosen.


# In[72]:


test_df = pd.read_csv('test.csv')
print(test_df.describe())


# In[73]:


#Check missing value
print(test_df.isnull().sum())


# In[74]:


#Data type conversion
test_df['pickup_datetime']=pd.to_datetime(test_df['pickup_datetime'])
print(test_df.dtypes)


# In[75]:


#Outlier Analysis
num_variables=['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count']
for i in num_variables:
    q75,q25=np.percentile(test_df.loc[:,i],[75,25])
    iqr=q75-q25
    min=q25 -(1.5*iqr)
    max=q75 +(1.5*iqr)
    
    test_df=test_df.drop(test_df[test_df.loc[:,i]<min].index)
    test_df=test_df.drop(test_df[test_df.loc[:,i]>max].index)


# In[76]:


test_df.describe()


# In[77]:


#Feature Engineering
test_df['year'] = test_df.pickup_datetime.dt.year
test_df['month'] = test_df.pickup_datetime.dt.month
test_df['day'] = test_df.pickup_datetime.dt.day
test_df['weekday'] = test_df.pickup_datetime.dt.weekday
test_df['hour'] = test_df.pickup_datetime.dt.hour
del test_df['pickup_datetime']


# In[78]:


def haversine_distance(lat1, long1, lat2, long2):
    data = [test_df]
    for i in data:
        R = 6371  #radius of earth in kilometers
        #R = 3959 #radius of earth in miles
        phi1 = np.radians(i[lat1])
        phi2 = np.radians(i[lat2])
    
        delta_phi = np.radians(i[lat2]-i[lat1])
        delta_lambda = np.radians(i[long2]-i[long1])
    
        #a = sin²((φB - φA)/2) + cos φA . cos φB . sin²((λB - λA)/2)
        a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    
        #c = 2 * atan2( √a, √(1−a) )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
        #d = R*c
        d = (R * c) #in kilometers
        i['distance_km'] = d
    return d


# In[79]:


test_df['distance_km'] = haversine_distance('pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')
test_df.head()


# In[80]:


del test_df['pickup_longitude']
del test_df['pickup_latitude']
del test_df['dropoff_longitude']
del test_df['dropoff_latitude']
test_df.describe()


# In[81]:


#Feature scaling for all values to lie under 1 range and then predicting
y = sc.transform(test_df) 
predicted_fare = model.predict(y)
predicted_fare


# In[82]:


test_df['predicted_fare'] = predicted_fare


# In[83]:


test_df.head()


# In[84]:


test_df = test_df[test_df['distance_km']>=1]
test_df.describe()


# In[85]:


plt.scatter(test_df['distance_km'], test_df['predicted_fare'], color='red')

plt.title('distance vs fare_amount', fontsize=14)

plt.xlabel('distance', fontsize=14)

plt.ylabel('fare', fontsize=14)

plt.grid(True)

plt.show()


# In[86]:


plt.scatter(test_df['hour'],test_df['predicted_fare'], color='red')

plt.title('hour vs fare_amount', fontsize=14)

plt.xlabel('hour', fontsize=14)

plt.ylabel('fare', fontsize=14)

plt.grid(True)

plt.show()


# In[87]:


import itertools
yrs = [i for i in test_df["year"].unique().tolist() if i not in [2015]]

#subset data without year 2015
complete_dat = test_df[test_df["year"].isin(yrs)]


plt.figure(figsize = (13,15))
for i,j in itertools.zip_longest(yrs,range(len(yrs))) :
    plt.subplot(3,2,j+1)
    trip_counts_mn = complete_dat[complete_dat["year"] == i]["month"].value_counts()
    trip_counts_mn = trip_counts_mn.reset_index()
    sns.barplot(trip_counts_mn["index"],trip_counts_mn["month"],
                palette = "rainbow",linewidth = 1,
                edgecolor = "k"*complete_dat["month"].nunique() 
               )
    plt.title(i,color = "b",fontsize = 12)
    plt.grid(True)
    plt.xlabel("")
    plt.ylabel("trips")


# In[88]:


fare_mn = test_df.groupby("month")["predicted_fare"].mean().reset_index()


plt.figure(figsize = (10,5))
sns.barplot("month","predicted_fare",
            data = fare_mn,
            linewidth =1)
plt.grid(True)
plt.title("Average fare amount by Month")
plt.show()


# In[89]:


import pickle


# In[90]:


pickle.dump(model, open("predict_fare.pkl","wb"))


# In[ ]:




