import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

file1 = pd.read_excel('E:/simsim1/5_6final.xlsx')
file2 = pd.read_excel('E:/6yekta.xlsx')


# Fill NaN values with empty strings in relevant columns
file1['نشانی گیرنده'] = file1['نشانی گیرنده'].fillna("")
file2['RCIEVERNAME'] = file2['RCIEVERNAME'].fillna("")

# Initialize lists to store the cosine similarity and mapped data
cosine_similarities = []
mapped_data = []

# TF-IDF Vectorizer for similarity calculation
vectorizer = TfidfVectorizer()

# Vectorize the RCIEVERNAME column from file2
rcievername_vectors = vectorizer.fit_transform(file2['RCIEVERNAME'])

for index, row in file1.iterrows():
    # Check if SHENASEMELIGIRANDE is not in RCIEVERNAME (only process when it does not match)
    if row['SHENASEMELIGIRANDE'] not in file2['RCIEVERNAME'].values:
        # Vectorize the current نشانی گیرنده address
        address_vec = vectorizer.transform([row['نشانی گیرنده']])
        
        # Calculate cosine similarities with all RCIEVERNAME vectors
        similarities = cosine_similarity(address_vec, rcievername_vectors).flatten()
        
        # Find the best match
        best_match_idx = similarities.argmax()
        best_match = file2['RCIEVERNAME'].iloc[best_match_idx]
        best_similarity = similarities[best_match_idx]
        
        # Store the best similarity and corresponding matched name
        cosine_similarities.append(best_similarity)
        mapped_data.append(best_match)
    else:
        # If the SHENASEMELIGIRANDE is already in RCIEVERNAME, no mapping needed
        cosine_similarities.append(None)
        mapped_data.append(None)

# Add the new columns to file1
file1['Cosine Similarity'] = cosine_similarities
file1['Mapped Data'] = mapped_data


# Save the updated dataframe to a new Excel file
file1.to_excel('mapped_output_sim_5_6.xlsx', index=False)
