import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Read the dataset with community labels
data = pd.read_csv('data_with_communities.csv')

# Get the top 6 communities by document count
top_communities = data['Community'].value_counts().head(6).index

# Function to calculate TF-IDF for a community
def calculate_tfidf(texts):
    """
    Calculate TF-IDF for a list of texts and return scores.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1
    return dict(zip(feature_names, scores))

# Set up the 2x3 grid for subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# Process each community and plot the word cloud
for i, community in enumerate(top_communities):
    # Filter texts for the current community
    community_texts = data[data['Community'] == community]['Uses'].tolist()
    
    # Calculate TF-IDF scores
    tfidf_scores = calculate_tfidf(community_texts)
    
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(tfidf_scores)
    
    # Plot on the respective subplot
    axes[i].imshow(wordcloud, interpolation='bilinear')
    axes[i].axis('off')
    axes[i].set_title(f'Community {community}', fontsize=16)

# Adjust layout
plt.tight_layout()
plt.show()
