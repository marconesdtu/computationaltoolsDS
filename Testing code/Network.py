import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.utils import murmurhash3_32
import numpy as np

# Load data
file_path = 'medicine_details.csv'
data = pd.read_csv(file_path)

# Preprocess text
stop_words = set(stopwords.words('english'))
stop_words.add('treatment')

def clean_text(text):
    text = re.sub(r'[()]', '', text)
    words = [word for word in text.split() if word.lower() not in stop_words]
    return ' '.join(words)

data['Uses'] = data['Uses'].apply(lambda x: clean_text(str(x)))

# Generate shingles
def create_shingles(text, k=3):
    words = text.split()
    if len(words) < k:
        return set(words)
    return set([' '.join(words[i:i + k]) for i in range(len(words) - k + 1)])

data['Shingles'] = data['Uses'].apply(lambda x: create_shingles(str(x)))

# Generate MinHash signatures
all_shingles = set().union(*data['Shingles'])
shingle_to_idx = {shingle: idx for idx, shingle in enumerate(all_shingles)}

n_hashes = 100
hash_functions = [lambda x, seed=seed: murmurhash3_32(x, seed=seed) for seed in range(n_hashes)]

def minhash_signature(shingles, n_hashes, shingle_to_idx):
    signature = np.full(n_hashes, np.inf)
    for shingle in shingles:
        shingle_idx = shingle_to_idx[shingle]
        for i, h in enumerate(hash_functions):
            signature[i] = min(signature[i], h(str(shingle_idx)))
    return signature

data['Minhash'] = data['Shingles'].apply(lambda x: minhash_signature(x, n_hashes, shingle_to_idx))

# Locality Sensitive Hashing (LSH)
def lsh(signatures, bands, rows):
    buckets = {}
    for doc_id, signature in enumerate(signatures):
        for band in range(bands):
            band_signature = tuple(signature[band * rows:(band + 1) * rows])
            if band_signature not in buckets:
                buckets[band_signature] = []
            buckets[band_signature].append(doc_id)
    return buckets

bands = 20
rows = n_hashes // bands
lsh_buckets = lsh(data['Minhash'], bands, rows)

# Display buckets
bucket_data = {
    bucket_id: [data.iloc[doc_id]['Uses'] for doc_id in doc_ids]
    for bucket_id, doc_ids in lsh_buckets.items()
}

bucket_df = pd.DataFrame(
    [{"Bucket": bucket_id, "Documents": doc_texts} for bucket_id, doc_texts in bucket_data.items()]
)

import ace_tools as tools; tools.display_dataframe_to_user(name="Buckets with Document Data", dataframe=bucket_df)
