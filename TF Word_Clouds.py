import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

# Read the dataset with community labels
data = pd.read_csv('data_with_communities.csv')

# Function to calculate term frequency for a community
def calculate_tf(texts):
    """
    Calculate term frequency for a list of texts.
    """
    all_words = ' '.join(texts).split()
    return Counter(all_words)

# Plot word cloud for a community in a specific subplot
def plot_wordcloud(ax, word_freq, community_id):
    """
    Generate and display a word cloud from word frequencies in a specific subplot axis.
    """
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Community {community_id}', fontsize=16)

# Get the top 6 communities by document count
top_communities = data['Community'].value_counts().head(6).index

# Create a figure with a 2x3 grid of subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Flatten the axes array for easier indexing
axes = axes.flatten()

# Process each of the top 6 communities
for i, community in enumerate(top_communities):
    # Filter texts for the current community
    community_texts = data[data['Community'] == community]['Uses']
    
    # Calculate term frequency
    word_frequencies = calculate_tf(community_texts)
    
    # Plot word cloud in the corresponding subplot
    plot_wordcloud(axes[i], word_frequencies, community)

# Adjust layout for better spacing
plt.tight_layout()
plt.show()
