import pandas as pd
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.utils import murmurhash3_32
from itertools import combinations
from community import community_louvain  # For Louvain method

# read file
file_path = 'medicine_details.csv'
data = pd.read_csv(file_path)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words.add('treatment')
# delete the stop words and treament
data['Uses'] = data['Uses'].apply(lambda x: ' '.join([word for word in str(x).split() if word.lower() not in stop_words]))


# Function to generate shingles
def create_shingles(text, k=3):
    """
    Split text into k-shingles (n-grams of size k).
    """
    words = text.split()
    if len(words) < k:
        return set(words)
    return set([' '.join(words[i:i+k]) for i in range(len(words) - k + 1)])

# Generate shingles for each document
k = 3  # Size of shingles
data['Shingles'] = data['Uses'].apply(lambda x: create_shingles(str(x)))

# Create the universe of shingles and map them to indices
all_shingles = set().union(*data['Shingles'])
shingle_to_idx = {shingle: idx for idx, shingle in enumerate(all_shingles)}

# Minhashing
n_hashes = 1000  # Number of hash functions
hash_functions = [lambda x, seed=seed: murmurhash3_32(x, seed=seed) for seed in range(n_hashes)]

def minhash_signature(shingles, n_hashes, shingle_to_idx):
    """
    Generate a Minhash signature for a set of shingles.
    """
    signature = np.full(n_hashes, np.inf)
    for shingle in shingles:
        shingle_idx = shingle_to_idx[shingle]
        for i, h in enumerate(hash_functions):
            signature[i] = min(signature[i], h(str(shingle_idx)))
    return signature

data['Minhash'] = data['Shingles'].apply(lambda x: minhash_signature(x, n_hashes, shingle_to_idx))

# Locality Sensitive Hashing (LSH)
def lsh(signatures, bands, rows):
    """
    Perform Locality Sensitive Hashing (LSH) to group similar documents.
    """
    buckets = {}
    for doc_id, signature in enumerate(signatures):
        for band in range(bands):
            band_signature = tuple(signature[band*rows:(band+1)*rows])
            if band_signature not in buckets:
                buckets[band_signature] = []
            buckets[band_signature].append(doc_id)
    return buckets

bands = 20  # Number of bands
rows = n_hashes // bands  # Rows per band
lsh_buckets = lsh(data['Minhash'], bands, rows)

# Find potential similar document pairs
similar_pairs = set()
for bucket_docs in lsh_buckets.values():
    if len(bucket_docs) > 1:
        similar_pairs.update(combinations(bucket_docs, 2))

# Build a graph of similar documents
G = nx.Graph()
G.add_nodes_from(range(len(data)))  # Add nodes for each document

# Add edges based on similar pairs
for doc1, doc2 in similar_pairs:
    shingles1 = data.loc[doc1, 'Shingles']
    shingles2 = data.loc[doc2, 'Shingles']
    # Compute Jaccard similarity
    jaccard_sim = len(shingles1 & shingles2) / len(shingles1 | shingles2)
    if jaccard_sim > 0.3:  # Similarity threshold
        G.add_edge(doc1, doc2, weight=jaccard_sim)

# Apply Louvain community detection
partition = community_louvain.best_partition(G, weight='weight')

# Add partition labels to the graph as node attributes
nx.set_node_attributes(G, partition, 'community')

# Save the graph to a .gexf file
nx.write_gexf(G, 'graph_with_communities.gexf')

# Add community labels to the dataframe
data['Community'] = data.index.map(partition)

# Save the data with community information
data.to_csv('data_with_communities.csv', index=False)

