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
data['Composition'] = data['Composition'].str.split('+')

# Step 2: Apply One-Hot Encoding
# This creates a dataFrame where each unique component is a column with 1 or 0 indicating presence in each row
one_hot_data = data['Composition'].str.join('|').str.get_dummies()

# Step 3: Concatenate the one-hot encoded columns back with the original dataFrame (if needed)
data2 = pd.concat([one_hot_data], axis=1)

# Display the resulting dataFrame
# print(data2.columns)
#following 2 lines printed the one hot ingredients for the second medicine to see if the one hot encoding of the strings it's correct
# second_row_dict = data2.iloc[1].to_dict()
# print(second_row_dict)
print(data2)

# Define weights for each rating category. Here we're assuming:
#   Good ratings contribute positively to the score (scaled up)
#   Middle ratings contribute a neutral or mid-range value
#   Bad ratings contribute negatively to the score (scaled down)

good_weight = 10  # Max score for good ratings
middle_weight = 5 # Midpoint score for middle ratings
bad_weight = 0    # Min score for bad ratings

# Calculate the weighted score for each observation
data['rating_score'] = (
    data['Excellent Review %'] * good_weight +
    data['Average Review %'] * middle_weight +
    data['Poor Review %'] * bad_weight
) / 100  # Divide by 100 to bring it back to a 0-10 scale

# pd.set_option('display.max_rows', None)  # Show all rows
# pd.set_option('display.max_columns', None)  # Show all columns
# pd.set_option('display.width', 1000)  
# print(data[['rating_score']])