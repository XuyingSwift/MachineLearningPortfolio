import numpy as np

def mse(y, y_hat):
    return np.sum((y - y_hat)**2)/y.shape[0]

num_samples, num_targets = 10, 5

# generating random values for y
y = np.random.rand(num_samples, num_targets)

# generating random values for y_hat
y_hat = np.random.rand(num_samples, num_targets)

# calculating and printing the mse between y and y_hat
print(mse(y, y_hat))
