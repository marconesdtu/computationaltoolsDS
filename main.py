import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the file paths
file1 = '/home/marco/DTU/CTfDS/project/computationaltoolsDS/medicine_details.csv'

# Read the Excel files
data = pd.read_csv(file1)

# Print the dataframes to verify
# print("data from medicine_details.csv:")
# print(data)  


# Print the dataframe to verify the new feature
# print("data with ratings:")
# print(data)

# Step 1: Split the Composition column by " + " to get a list of components for each observation
data['Composition'] = data['Composition'].str.split(' + ')

# Step 2: Apply One-Hot Encoding
# This creates a dataFrame where each unique component is a column with 1 or 0 indicating presence in each row
one_hot_data = data['Composition'].str.join('|').str.get_dummies()

# Step 3: Concatenate the one-hot encoded columns back with the original dataFrame (if needed)
data = pd.concat([data, one_hot_data], axis=1)

# Display the resulting dataFrame
print(data.columns)