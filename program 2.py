#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler


x = np.array([1, 2, 4, 3, 5])
y = np.array([1, 3, 4, 3, 7])


print(x.shape)
print(y.shape)
print(x)
print(y)


x = x.reshape(-1, 1)
print(x.shape)
print(x)


model = LinearRegression()
model.fit(x, y)
print(model.predict([[8]]))


y_pred = model.predict(x)


plt.scatter(x, y, color='r', marker='o')
plt.plot(x, y_pred, color='b', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.show()


print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)




scaler = StandardScaler()
x_stand = scaler.fit_transform(x.astype(float))


sgdr = SGDRegressor(penalty='l2', alpha=0.15, max_iter=100)


sgdr.fit(x_stand, y)


x_new = scaler.transform([[10]])
print(sgdr.predict(x_new))


y_pred2 = sgdr.predict(x_stand)


plt.scatter(x_stand, y, color='g', marker='o')
plt.plot(x_stand, y_pred2, color='b', linewidth=2)
plt.xlabel('x_normalized')
plt.ylabel('y')
plt.show()


print("SGD Coefficient:", sgdr.coef_)
print("SGD Intercept:", sgdr.intercept_)


# In[ ]:




