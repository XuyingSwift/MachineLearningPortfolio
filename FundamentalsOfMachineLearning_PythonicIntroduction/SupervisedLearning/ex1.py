import numpy as np
from matplotlib import pyplot as plt

# define the class line
class Line:
    def __init__(self, m = 1, c = 1) -> None:
        self.m = m
        self.c = c
        
    # Multiplying x-coordinates with ma and add c to return y-coordinates for the line
    def apply_f(self, x):
        return self.m*x + self.c
    
# Defining the mean squared error as a class
class Mean_Squared_Loss:
    def __init__(self, m=1, c=1, x=None, y=None) -> None:
        self.model = Line(m,c)
        self.compute_loss(x,y)
    
    def compute_loss(self, x, y):
        n_points = x.shape[0]
        self.mse = np.linalg.norm(self.model.apply_f(x) - y)/n_points
   
    def __str__(self) -> str:
        return f"MSE = {self.mse:.2f}: for m = {self.model.m} and c = {self.model.c}"
    
l = Line(m = -1, c = 3)
x = np.linspace(-10, 10, 100)
y = l.apply_f(x)

plt.grid()
plt.plot(x, y)

y = 2*x-5 # here m = 2 and c = -5
MSE = Mean_Squared_Loss(m=3,c=2,x=x,y=y)
print(MSE)

MSE = Mean_Squared_Loss(m=2,c=-5,x=x,y=y)
print(MSE)