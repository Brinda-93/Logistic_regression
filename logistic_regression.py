# Importing libraries
import torch
import torch.nn as nn
import toch.optim as optim
import numpy as numpy
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generating synthetic data
np.random_seed(42)
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_informative=2, n_redundant=0, n_repeated=0, random_state=42)
y = y.reshape(-1, 1)

#Converting the data into tensors
X_train = torch.tensor(X, dtype = torch.float32)
y_train = torch.tensor(y, dtype = torch.float32)

#Standard features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Defining the logistic regression model
class LogisticeRegressionModel(nn.module):
    def __init__(self):
        super(LogisticeRegressionModel, self).__init__()
        self.linear = nn.Linear(2, 1) # input features = 2, output feature = 1

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

#Initialize the model, loss function, optimizer
model = LogisticeRegressionModel()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001)

#Training the model
EPOCHS = 100
for epoch in range(EPOCHS):
    optimizer.zero_grad()

    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    loss.backward()

    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch : {epoch}/{EPOCHS}, Loss:{loss}')

#Plot the decision boundary
with torch.no_grad():
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype = torch.float32)
    probs = model(grid).reshape(xx.shape).detach().numpy()

    plt.contourf(xx, yy, probs, levels = [0, 0.5, 1], cmap = 'coolwarm', alpha = 0.6)
    plt.scatter(X[:, 0], X[:, 1], c = y.flatten(), edgecolors = 'k', cmap = 'coolwarm')
    plt.title('Decision Boundary')
    plt.show()