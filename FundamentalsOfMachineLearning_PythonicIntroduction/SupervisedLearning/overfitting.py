import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# keeping the seed value constant to observe effect of model comlexity
np.random.seed(1)

# setting sinusoidal distribution for generating synthetic data
def f(x):
    return x * np.sin(x)

# Inputting the degree testue increasing or descreasing order to understand its 
degree = int(input())

# Generating 100 synthetic data points, 
n_total = 100
x_total = np.linspace(0, 10, n_total)
y = f(x_total) + np.random.randn(n_total)
X = x_total[:,np.newaxis]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
idx = np.argsort(X_train[:,0])
X_train[:,0], y_train = X_train[idx,0], y_train[idx]

n_train, n_test = X_train.shape[0], X_test.shape[0]

# Constructing an automated workflow for training the model
model = make_pipeline(PolynomialFeatures(degree), LR())
model.fit(X_train, y_train)

# Computing MSE for training and testing data
train_loss = np.sum((model.predict(X_train)-y_train)**2)/n_train
test_loss = np.sum((model.predict(X_test)-y_test)**2)/n_test 

# Plotting
fig, ax = plt.subplots()
ax.plot(x_total, f(x_total), linewidth=1, label="ground truth", color = "green")
ax.scatter(X_train[:,0], y_train,label="training points", color = '#65b2ff', edgecolors='black')
ax.plot(X_train[:,0], model.predict(X_train), label=f"degree {degree}",linestyle='--', color = "darkorange")
plt.grid(color = 'b', linestyle = '--', linewidth = 0.3)
plt.xlabel(r'$x$',fontsize=20)
plt.ylabel(r'$\hat y$',fontsize=20)
plt.title('MSE(Train) = {:.2f}'.format(train_loss) + ':: MSE(test) = {:.2f}'.format(test_loss),fontsize=20)
ax.legend(loc="lower center")
ax.set_ylim(-20, 10)

