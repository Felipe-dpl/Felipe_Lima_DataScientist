# Felipe_Lima_DataScientist
```python
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from statsmodels.tools.eval_measures import rmse
from statsmodels.tsa.arima_model import ARIMA

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
import random
import math

import warnings

warnings.filterwarnings("ignore")

!pip install squarify
import squarify
```
```python
data = pd.read_csv("/content/TFP.csv")
data.head() # Using (head) to get the first 5 values of our dataset to check it

data.info # Using (info) to see a little more of our dataset, here we can get its size too, for example, this one have 186 rows and 3 columns

data.describe() # With (descibre) we can get some informations regarding the max and min values, the mean of each column, each quartile, so we can have a better understanding of it

data.nunique() # We can check the uniqueness of our columns using (nunique), we can see that we have only 3 countries(isocode), have 62 years represented (from 1950 to 2011) and have 184 values for RTFPNA

data.isnull().sum() #Finally, we can check if we have any null value in the dataset
```

## Data Visualization

Now, we'll plot some views of this dataset, so we can get a look on it and we can discover patterns, spot anomalies, frame our hypothesis and check our assumptions

#Plot using the 3 Countries (isocode) by Year, using Seaborn
lm = sns.lmplot(x='year', y='rtfpna', data=data,fit_reg=False, hue='isocode')
fig = lm.fig 
fig.suptitle(x=0.5, y=1, t='Distribution of RTFPNA by Country per Year',va='center',fontsize=10)
plt.show()

# Pivoting the table so that each region is a column and each row is a year.
data1 = data.pivot_table(index='year',columns='isocode',values='rtfpna')

# Ploting again using another format, with lines and in a bigger size
fig, ax = plt.subplots(figsize=(20,10))
plt.title('Distribution of RTFPNA by Country per Year')
plt.xlabel('Years')
plt.ylabel('RFPNA')
plt.plot(data1)
ax.plot(data1.MEX, label='MEX', color='green')
ax.plot(data1.CAN, label='CAN', color='red')
ax.plot(data1.USA, label='USA', color='blue')
leg = ax.legend()
plt.show()

# Ploting the RTPFNA by Country, which allow us to get the distribution of the in a image.
fig, ax =plt.subplots(1,3)
fig.suptitle(x=0.5, y=1, t='Histogram of RTFPNA by Country',va='center',fontsize=14)
sns.distplot(data1.USA, color='blue', ax=ax[0])
sns.distplot(data1.CAN, color='red', ax=ax[1])
sns.distplot(data1.MEX, color='green', ax=ax[2])
fig.show()

# Ploting all of them in one figure.
sns.kdeplot(data1.USA, label="USA")
sns.kdeplot(data1.CAN, label="CAN", color='red')
sns.kdeplot(data1.MEX, label="MEX", color='green')
plt.legend();

"""## Forecasting

Here,  we'll show some tryouts of forecasting, trying to get the best result so we could use it in the future
"""

#Create a new dataframe with only the 'USA' column
data2 = data1.filter(['USA'])

#Converting the dataframe to a numpy array
data2 = data2.values

#Get /Compute the number of rows to train the model on
training_data_len = math.ceil( len(data2) *.8)

#Scale the all of the data to be values between 0 and 1 
scaler = MinMaxScaler(feature_range=(0, 1)) 
scaled_data = scaler.fit_transform(data2)

#Create the scaled training data set 
train_data = scaled_data[0:training_data_len  , : ]

#Split the data into x_train and y_train data sets
x_train=[]
y_train=[]
for i in range(1,len(train_data)):
    x_train.append(train_data[i-1:i,0])
    y_train.append(train_data[i,0])

#Convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the data into the shape accepted by the LSTM
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
x_train.shape

# Now we set how we want our model, set our hidden layers, activation functions and how we want our return, in this case we want 1 value. Setted this one by trial and erro to get the best result.
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25, activation='linear'))
model.add(Dense(1))
#Compile the model
model.compile(optimizer='rmsprop', loss='mean_squared_error')

#Train the model
model.fit(x_train, y_train, batch_size=1, epochs=100)

#Test data set
test_data = scaled_data[training_data_len - 1: , : ]
#Create the x_test and y_test data sets
x_test = []
y_test =  data2[training_data_len : , : ]
for i in range(1,len(test_data)):
    x_test.append(test_data[i-1:i,0])

#Convert x_test to a numpy array 
x_test = np.array(x_test)

#Reshape the data into the shape accepted by the LSTM
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

#Getting the models predicted price values
predictions = model.predict(x_test) 
predictions = scaler.inverse_transform(predictions) #Undo scaling

#Calculate/Get the value of RMSE
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
rmse

#Plot/Create the data for the graph
train = data1[ :training_data_len]
valid = data1[training_data_len: ]
valid['Predictions'] = predictions


#Visualize the data of how our model behave with the already known data
plt.figure(figsize=(16,8))
plt.title('Train & Test Model')
plt.xlabel('Years', fontsize=12)
plt.ylabel('RFTPNA', fontsize=12)
plt.plot(train['USA'])
plt.plot(valid[['USA', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

#Creating an copy of our used dataset, to try with LSTM to predict the 10 years
data3 = data2.copy()

#Always getting the last line of our dataset to try to predict the next year, which we use a loop to get 10 years.
for i in range(10):
  x = model.predict(np.array([[[data3[-1][0]]]]))
  data3 = np.append(data3, x, 0)

#Visualize the data of our model
plt.figure(figsize=(16,8))
plt.title('Prediction Model')
plt.xlabel('Years', fontsize=12)
plt.ylabel('RFTPNA', fontsize=12)
plt.plot(data3)
plt.legend(['Predictions'], loc='lower right')
plt.show()

"""Okay, now, let's try using some Decision Tree"""

usa = data1[['USA']]
usa.head()

#Variavel de predição futura - 10 anos
#future_year=10

#Criar coluna nova com o target
usa['Predict'] = usa[['USA']]

#Criar uma feature no dataset e converter para array numpy e remover os "X" anos
x = np.array(usa.drop(['Predict'],1))

#Criar o dataset que queremos (y) e conveter para numpy e pegar todos os dados exceto os anos X
y= np.array(usa['Predict'])

#Criar os modelos
#Decision Tree de regressão
decision_tree = DecisionTreeRegressor().fit(x, y)

x_forecast = np.array([[x] for x in data1['USA']])

for i in range(10):
  x = decision_tree.predict(np.array([[x_forecast[-1][0]]]))
  #print(x_forecast[-1][0])
  x_forecast = np.append(x_forecast,[x],0)

plt.figure(figsize=(18,9))
plt.title('Teste')
plt.xlabel('Years')
plt.ylabel('Rfpna')
plt.plot(x_forecast)
plt.show()

"""Decision tree tende a repetir os valores, o modelo está defasado e não é recomendado.

Now, let's try using ARIMA to forecast
"""

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return np.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

#Creat a Pivot Table to use
series1 = data.pivot_table(index='year',columns='isocode',values='rtfpna')

# Using only USA column to start our forecast
usa = series1[['USA']]
X = usa.values
years_time = 1
differenced = difference(X, years_time)

# Fitting the model
model = ARIMA(differenced, order=(10,0,1))
model_fit = model.fit(disp=0)

# Multi-step out-of-sample forecast, trying for 10 steps, which represent our years
forecast = model_fit.forecast(steps=10)[0]

# Invert the differenced forecast for our use
history = [x for x in X]
year = 2012
x_future=[]

for yhat in forecast:
	inverted = inverse_difference(history, yhat, years_time)
	print('Day %d: %f' % (year, inverted))
	history.append(inverted)
	year += 1

#Importing to a new dataframe to plot it all
dates = pd.date_range(start='1950', periods=72, freq='Y')
df = pd.DataFrame(history,index=dates, columns=['USA'])

fig, ax1 = plt.subplots(figsize =(30,10))
ax1.grid() # turn on grid #2
ax1.set_title('TESTE')
ax1.set_xlabel('Years')
ax1.set_ylabel('Rfpna')
ax1.plot(df)

"""By analyzing all 3 forecasts, the recommended one should be the ARIMA Mode.

*Can you think about another feature that could be helpful in explaining TFP series?*

Yes, by analyzing the displayed variables in the PWT8, we should use this variables help analyze and understand the TFP:

* CTFP = Because with it we can see the actual PPP value, the Purchasing Power Parity, which affects directly the RTFPNA growth.

* csh_r = Shows the discrepancy of GDP values and the residual trade, showing the country growth.

# Case 2
"""

comex = pd.read_csv("/content/data_comexstat.csv", encoding = 'latin-1', sep = ',', parse_dates=['date'], index_col=['date'])

comex.head()

comex.shape

comex.describe

comex.info()

comex.nunique()

comex.isnull().sum()

dates = mpl.dates.date2num(comex.loc[comex['product'] == 'soybeans', 'usd'].index.values)
plt.plot_date(dates, comex.loc[comex['product'] == 'soybeans', 'usd'])

"""## Question 1

###Show the evolution of total monthly and total annual exports from Brazil (all states and to everywhere) of ‘soybeans’, ‘soybean oil’ and ‘soybean meal’.
"""

soybean = comex.loc[(comex['type']=='Export') & (comex['product'].isin(['soybeans','soybean_oil','soybean_meal']))]
soybean

group_soy = soybean.groupby(soybean.index).sum()

fig, ax1 = plt.subplots(figsize =(30,10))
ax2 = ax1.twinx()  # set up the 2nd axis
ax1.plot(group_soy['usd'], lw = 4) #plot the Revenue on axis #1
#ax1.xaxis_date()

ax1.grid() # turn on grid #2

ax2.bar(group_soy.index, group_soy['tons'], width=10, color = 'green')
#ax2.xaxis_date()

ax1.set_title('Monthly Exportation of Soybeans/Soybean Derivates and Volume of Exported Goods over the year', fontsize = 18, fontweight="bold")
ax1.set_ylabel('Export Monthly Values in USD', fontsize = 14)
ax2.set_ylabel('Export in Tons per Month', fontsize = 14)

"""## Question 2

###What are the 3 most important products exported by Brazil in the last 5 years?
"""

important = comex.loc[(comex['type']=='Export') & (comex.index >= '2015-01-01')]
important

products = important.groupby(important['product']).sum().sort_values(by='usd',ascending=False)[0:3]
products

dataProd = products[products['usd']>0]

norm = mpl.colors.Normalize(vmin=min(dataProd.usd), vmax=max(dataProd.usd))
colors = [mpl.cm.Greens(norm(value)) for value in dataProd.usd]

#Create our plot and resize it.
fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(16, 4.5)

#Use squarify to plot our data, label it and add colours. We add an alpha layer to ensure black labels show through
squarify.plot(label = products.index, sizes=dataProd['usd'], color = colors, alpha=.8)
plt.title("Most Important Products Exported by Brazil",fontsize=18,fontweight="bold")

#Remove our axes and display the plot
plt.axis('off')
plt.show()

"""The 3 most important are:


1.   Soybeans
2.   Sugar
3.   Soybean Meal

## Question 3

### What are the main routes through which Brazil have been exporting ‘corn’ in the last few years? Are there differences in the relative importance of routes depending on the product?
"""

data['route'].value_counts()

exporta = comex.loc[(comex['type']=='Export') & (comex.index >= '2017-01-01')]
exporta

corn = exporta.loc[exporta['product']=='corn'].groupby(['product','route']).sum().sort_values(by='usd',ascending=False)
corn.reset_index(inplace=True)
corn

routes = exporta.groupby(['route','product']).sum().sort_index()
routes.reset_index(inplace=True)
routes

# Create a pieplot
fig, ax1 = plt.subplots(figsize =(30,10))
plt.pie(corn['usd'], labels= corn['route'])

my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
 
plt.show()

pivot_routes = routes.pivot(index='product', columns='route', values='usd')
pivot_routes.fillna(0,inplace=True)
pivot_routes

adjust_route = pivot_routes.div(pivot_routes.sum(axis=1), axis=0)
adjust_route

adjust_route.plot.bar(stacked=True, figsize=(20,10))

"""Clearly we can see that the main exportation route in Brazil is the Sea, but we can take it out and see if the other routes affect the products and how."""

wo_sea = pivot_routes.drop('Sea', axis=1)
adjust_wo_sea= wo_sea.div(wo_sea.sum(axis=1), axis=0)
adjust_wo_sea
adjust_wo_sea.plot.bar(stacked=True, figsize=(20,10))

"""Without the Sea route, sugar, wheat and soybean oil are mostly affected by ground route and the corn and soy are mostly exported by River. With this, we can assume that, depending on the product, the route can affect the exportation, there is relevance in the routes

##Question 4

### Which countries have been the most important trade partners for Brazil in terms of ‘corn’ and ‘sugar’ in the last 3 years?
"""

comex.nunique()

#filtering
trade = comex.loc[(comex['product'].isin(['corn','sugar'])) & (comex.index >= '2017-01-01')]
trade

#groupying by
country_p = trade.groupby(['country','product']).sum().sort_index()
country_p.reset_index(inplace=True)
country_p

sugar = country_p.loc[(country_p['product'].isin(['sugar']))].sort_values(by ='usd', ascending=False)[0:5]
sugar

corn2 = country_p.loc[(country_p['product'].isin(['corn']))].sort_values(by ='usd', ascending=False)[0:5]
corn2

"""## Question 5

###For each of the products in the dataset, show the 5 most important states in terms of exports?
"""

export = comex.loc[(comex['type']== 'Export')]
export

produtos = export.groupby(['product','state']).sum()
produtos.reset_index(inplace=True)
produtos

prod1 = produtos.loc[(produtos['product']=='corn')].sort_values(by ='usd', ascending=False)[0:5]
prod2 = produtos.loc[(produtos['product']=='soybean')].sort_values(by ='usd', ascending=False)[0:5]
prod3 = produtos.loc[(produtos['product']=='soybean_meal')].sort_values(by ='usd', ascending=False)[0:5]
prod4 = produtos.loc[(produtos['product']=='soybean_oil')].sort_values(by ='usd', ascending=False)[0:5]
prod5 = produtos.loc[(produtos['product']=='wheat')].sort_values(by ='usd', ascending=False)[0:5]
prod6 = produtos.loc[(produtos['product']=='sugar')].sort_values(by ='usd', ascending=False)[0:5]

produtos_definitivo = pd.concat([prod1,prod2,prod3,prod4,prod5,prod6])
produtos_definitivo

```
