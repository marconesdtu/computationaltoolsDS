import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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


# Load data
uses = data['Uses'].tolist()
ratings = data['rating_score'].values

# Step 1: Compute TF-IDF
def calculate_tfidf(texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix

tfidf_matrix = calculate_tfidf(uses)

# Step 2: Compute similarity matrix
similarity_matrix = cosine_similarity(tfidf_matrix)

# Step 3: Identify similar medicines
similar_meds = {}
threshold = 0.75
for i, row in enumerate(similarity_matrix):
    similar = np.where(row >= threshold)[0]
    similar_meds[i] = similar

# Step 4: Calculate deltas
medicine_scores = []
for i, rating in enumerate(ratings):
    similar_indices = similar_meds[i]
    if len(similar_indices) > 0:
        similar_ratings = ratings[similar_indices]
        avg_rating = np.mean(similar_ratings)
        min_rating = np.min(similar_ratings)
        
        delta_avg = rating - avg_rating
        delta_min = rating - min_rating
        count_similar = len(similar_indices)
        
        medicine_scores.append({
            'medicine_index': i,  # Store index
            'rating': rating,
            'delta_avg': delta_avg,
            'delta_min': delta_min,
            'count_similar': count_similar
        })

# Step 5: Create DataFrame and rank medicines
medicine_df = pd.DataFrame(medicine_scores)


# Filter medicines with at least 3 similar ones or negative delta_avg
filtered_meds = medicine_df[
    (medicine_df['count_similar'] >= 3) | (medicine_df['delta_avg'] < 0)
]

# Adjust obsolescence scoring (low rating is more important now)
filtered_meds['obsolescence_score'] = (
   - 0.5 * filtered_meds['delta_avg'] - 0.3 * filtered_meds['delta_min'] - 0.2 * filtered_meds['rating']
)

# Select the bottom 5%
top_5_percent = filtered_meds.nlargest(
    int(0.05 * len(filtered_meds)), 'obsolescence_score'
)

# Step 6: Output results
top_5_percent = top_5_percent.merge(
    data.reset_index(), left_on='medicine_index', right_index=True
)
print("Top 5% Medicines Likely to Become Obsolete:")
print(top_5_percent[['medicine_index', 'Uses', 'rating', 'obsolescence_score']])

# Load data
data2 = pd.read_csv('/home/marco/DTU/CTfDS/project/computationaltoolsDS/data_with_communities.csv')
data2['obsolescence_score'] = filtered_meds['obsolescence_score']

# Step 1: Extract relevant columns
data2['Community'] = data2.iloc[:, 10]  # 11th column for Community

# # Step 1.5: Select 10 random communities
# random_communities = np.random.choice(data2['Community'].unique(), 50, replace=False)
# filtered_data2 = data2[data2['Community'].isin(random_communities)]

filtered_data2 = data2
random_communities = filtered_data2['Community'].unique()

# Step 2: Group by Community and calculate cluster centers
community_groups = filtered_data2.groupby('Community')
cluster_centers = community_groups['obsolescence_score'].mean()

# Step 3: Calculate distance from cluster center
filtered_data2['distance_from_center'] = filtered_data2.apply(
    lambda row: abs(row['obsolescence_score'] - cluster_centers[row['Community']]),
    axis=1
)


# Marked data: Ensure the 'medicine_index' exists in top_5_percent
filtered_data2['medicine_index'] = filtered_data2.iloc[:, 0]

# Marked data: Ensure the 'medicine_index' exists in top_5_percent
if 'medicine_index' in filtered_data2.columns and 'medicine_index' in top_5_percent.columns:
    marked_data = filtered_data2[filtered_data2['medicine_index'].isin(top_5_percent['medicine_index'])]
else:
    marked_data = pd.DataFrame()

plt.figure(figsize=(15, 10))

# Plot the data for each random community
for community in random_communities:
    community_data = filtered_data2[filtered_data2['Community'] == community]
    
    # Scatter plot for community data
    plt.scatter(
        [community] * len(community_data),  # x-axis: Community ID
        community_data['distance_from_center'],  # y-axis: distance from center
        label=f"Community {community}",
        alpha=0.7,
    )

# Scatter plot for marked data
plt.scatter(
    marked_data['Community'],  # x-axis: Community ID
    marked_data['distance_from_center'],  # y-axis: distance from center
    color='red',
    marker='x',
    s=100,  # Size of the marker
)

# Adding labels and title
plt.xlabel("Community")
plt.ylabel("Distance from Center (Obsolescence Score)")
plt.title("Cluster Analysis for Random Communities")
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.tight_layout()

# Show the plot
plt.show()
