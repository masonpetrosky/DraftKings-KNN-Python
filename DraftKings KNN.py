# Import necessary libraries
import pandas as pd
import numpy as np

# Load the training data
train = pd.read_csv("RB.csv")

# Scale the "Projection" and "Salary" columns
train['ProjectionScaled'] = train['Projection'] / train['Projection'].mean()
train['SalaryScaled'] = train['Salary'] / train['Salary'].mean()

# Initialize lists to store fitted values and error values
error_values = []

# Loop over potential k values (leave-one-out cross-validation)
for k in range(1, len(train)):
    squared_errors = []

    for i in range(len(train)):
        # Separate the dataset into training and validation
        validation = train.iloc[i]
        training = train.drop(i)

        # Calculate Euclidean distance
        training['Distance'] = np.sqrt((validation['ProjectionScaled'] - training['ProjectionScaled']) ** 2 +
                                       (validation['SalaryScaled'] - training['SalaryScaled']) ** 2)

        # Sort training set by distance and select k nearest neighbors
        nearest_neighbors = training.nsmallest(k, 'Distance')

        # Calculate the mean of the Actual values of the nearest neighbors
        prediction = nearest_neighbors['Actual'].mean()

        # Calculate squared error
        squared_error = (prediction - validation['Actual']) ** 2
        squared_errors.append(squared_error)

    # Calculate the root mean squared error for this k value
    rmse = np.sqrt(np.mean(squared_errors))
    error_values.append(rmse)

# Find the k value with the lowest error
ideal_k = np.argmin(error_values) + 1

# Load the test data
test = pd.read_csv("Test.csv")

# Scale the testing data similarly to the training data
test['ProjectionScaled'] = test['Projection'] / train['Projection'].mean()
test['SalaryScaled'] = test['Salary'] / train['Salary'].mean()

# Initialize list for predictions
fitted_values = []

# Predict for each instance in the test dataset
for i in range(len(test)):
    # Current test instance
    testing = test.iloc[i]

    # Calculate Euclidean distance for each training instance
    train['Distance'] = np.sqrt((testing['ProjectionScaled'] - train['ProjectionScaled']) ** 2 +
                                (testing['SalaryScaled'] - train['SalaryScaled']) ** 2)

    # Sort training set by distance and select ideal_k nearest neighbors
    nearest_neighbors = train.nsmallest(ideal_k, 'Distance')

    # Calculate the mean of the Actual values of the nearest neighbors
    prediction = nearest_neighbors['Actual'].mean()

    # Store the prediction
    fitted_values.append(prediction)

# Append predictions to the test dataset
test['Pred'] = fitted_values

# Write the results to a CSV file
test.to_csv("testResults.csv", index=False)