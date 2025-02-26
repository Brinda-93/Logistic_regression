# Logistic Regression using PyTorch

## Overview

This project implements a **Logistic Regression** model using PyTorch to classify synthetic data generated using `sklearn.datasets.make_classification`. The model learns to distinguish between two classes based on two input features.

## Features

- Uses **PyTorch** for model implementation and training.
- Generates synthetic data for classification.
- Implements **gradient descent** for optimization.
- Uses **BCELoss** for binary classification.
- **Plots the decision boundary** after training.

## Dependencies

Ensure you have the following dependencies installed before running the script:

```bash
pip install torch numpy matplotlib scikit-learn
```

## Usage

Run the Python script to train the Logistic Regression model:

```bash
python logistic_regression.py
```

## Implementation Details

1. Generate synthetic classification data using `make_classification`.
2. Normalize the features using `StandardScaler`.
3. Convert the dataset into PyTorch tensors.
4. Define a **Logistic Regression** model with a single linear layer and sigmoid activation.
5. Train the model using **Stochastic Gradient Descent (SGD)**.
6. Display the **decision boundary** after training.

## Visualization

After training, the script generates the following decision boundary graph:



## Example Output

```
Epoch : 9/500, Loss:0.6354324221611023
Epoch : 19/500, Loss:0.6323456168174744
Epoch : 29/500, Loss:0.6292890310287476
...
...
Epoch : 489/500, Loss:0.5162254571914673
Epoch : 499/500, Loss:0.5142671465873718
```

A decision boundary graph is displayed to visualize the classification performance.
![Decision Boundary](graph.png)

## Author

Brinda Navakumar

## License

This project is licensed under the MIT License.

