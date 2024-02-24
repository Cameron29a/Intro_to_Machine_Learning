import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""**Mount Google Drive to Colab.**"""

from google.colab import drive
drive.mount('/content/drive')

"""**Load Data into Data Frame**"""

df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Assignment_1/D3.csv')

"""**Extracting explanatory variables (X) and the output (Y)**"""

X = df[['X1', 'X2', 'X3']].values
Y = df['Y'].values

"""**Display first 5 rows of data frame.**"""

df.head()

"""# **Define Gradient Descent Function**"""

def gradientDescent(X, Y, learningRate, numIterations):
    m = len(Y)
    n = X.shape[1]
    theta = np.zeros(n)  # Initialize parameters to zero
    lossHistory = []

    for i in range(numIterations):
        # Calculate predictions
        predictions = np.dot(X, theta)

        # Calculate error
        error = predictions - Y

        # Update parameters
        theta -= (learningRate / m) * np.dot(X.T, error)

        # Calculate loss (mean squared error)
        loss = np.mean(np.square(error))
        lossHistory.append(loss)

    return theta, lossHistory

"""# **Problem 1**

**Perform linear regression for each explanatory variable**
"""

models = []
losses = []

learningRates = [0.1, 0.05, 0.01]  # Different learning rates
numIterations = 1000

for i in range(3):
    theta, lossHistory = gradientDescent(X[:, i].reshape(-1, 1), Y, learningRates[i], numIterations)
    models.append(theta)
    losses.append(lossHistory)

"""**1. Report linear models for each explanatory variable**"""

for i, theta in enumerate(models):
    print(f"Linear model for X{i+1}: Y = {theta[0]:.2f} * X{i+1}")

"""**2. Plot final regression model and loss over the iteration for each explanatory variable**"""

for i in range(3):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(numIterations), losses[i])
    plt.title(f"Loss over Iterations for X{i+1}")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.scatter(X[:, i], Y, label='Actual')
    plt.plot(X[:, i], X[:, i] * models[i][0], color='red', label='Predicted')
    plt.title(f"Regression Model for X{i+1}")
    plt.xlabel(f"X{i+1}")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

"""**3. Find the explanatory variable with the lowest loss**


"""

minLoss_idx = np.argmin([loss[-1] for loss in losses])
print(f"The explanatory variable X{minLoss_idx + 1} has the lowest final loss.")

"""**4. Based on your training observations, describe the impact of the different learning rates on the final loss and number of training iterations.**

Higher learning rates result in faster convergence at lower iterations, but have greater loss due to overshoot and instability. Lower learning rates require more iterations, but have lower loss due to small steps toward convergence. The decision comes to tade off between speed and stability. For this test, a learning rate in the middle seems to do the best. It offers the lowest loss at the least number of iterations. Every test needed less than 100 runs to reach min loss.

# **Problem 2**

**Perform linear regression using all three explanatory variables**
"""

finalModels = []
finalLosses = []

for lr in learningRates:
    # Perform gradient descent
    theta, lossHistory = gradientDescent(X, Y, lr, numIterations)

    # Save final model and loss
    finalModels.append(theta)
    finalLosses.append(lossHistory[-1])

"""**1. Report final linear models**"""

bestLR_idx = np.argmin(finalLosses)
bestLR = learningRates[bestLR_idx]
bestModel = finalModels[bestLR_idx]
print(f"Best learning rate: {bestLR}")
print(f"Final linear model: Y = {bestModel[0]:.2f} * X1 + {bestModel[1]:.2f} * X2 + {bestModel[2]:.2f} * X3")

"""**2. Plot loss over the iteration**"""

plt.plot(range(numIterations), lossHistory)
plt.title("Loss over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

"""**3. Based on your training observations, describe the impact of the different learning rates on the final loss and number of training iterations.**

In this test it took more iterations before seeing noticable impact to the loss. This is due to more variables being tested at once. With this a learning rate in the middle would be prefered, becase of the multiple variables at play. Min loss was reached before 400 iterations.

**4. Predict the value of Y for new (X1, X2, X3) values**
"""

new_X = np.array([[1, 1, 1], [2, 0, 4], [3, 2, 1]])
predicted_Y = np.dot(new_X, bestModel)
for i, x in enumerate(new_X):
    print(f"For X1={x[0]}, X2={x[1]}, X3={x[2]}, predicted Y={predicted_Y[i]}")
