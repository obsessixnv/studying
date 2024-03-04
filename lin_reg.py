import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model

df = pd.read_csv('static/my2024-fuel-consumption-ratings.csv', encoding='latin1')
df.head()
df.describe()
cdf = df[['Engine size (L)', 'CO2 emissions (g/km)']]
cdf.head(9)


msk = np.random.rand(len(df))< 0.8
train = cdf[msk]
test= cdf[~msk]


regr = linear_model.LinearRegression()
train_x= np.asanyarray(train[['Engine size (L)']])
train_y= np.asanyarray(train[['CO2 emissions (g/km)']])
regr.fit(train_x, train_y)


print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)

test_x= np.asanyarray(test[['Engine size (L)']])
test_y= np.asanyarray(test[['CO2 emissions (g/km)']])
test_y_=regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))

plt.scatter(train['Engine size (L)'], train['CO2 emissions (g/km)'], color = 'cyan')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], 'r')
plt.xlabel('Engine size (L)')
plt.ylabel('Co2 Emission')


plt.show()

