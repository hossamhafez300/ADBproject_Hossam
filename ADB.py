import numpy as np
import faiss
import time

class VectorIndexCosine:
    def __init__(self, dim):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # IndexFlatIP for cosine similarity
        self.ids = []

    def add_vector(self, vector, doc_id):
        # Normalize the vector to unit length for cosine similarity
        vector = vector / np.linalg.norm(vector)
        vector = np.array([vector]).astype('float32')
        self.index.add(vector)
        self.ids.append(doc_id)

    def search_vector(self, query_vector, k=5):
        # Normalize the query vector to unit length for cosine similarity
        query_vector = query_vector / np.linalg.norm(query_vector)
        query_vector = np.array([query_vector]).astype('float32')
        distances, indices = self.index.search(query_vector, k)
        return list(zip([self.ids[i] for i in indices[0]], distances[0]))

# Example usage
vector_index_cosine = VectorIndexCosine(dim=70)
vector_index_cosine.add_vector(np.random.rand(70), "doc_id_1")
vector_index_cosine.add_vector(np.random.rand(70), "doc_id_2")

# Search for similar vectors using cosine similarity
query_vector = np.random.rand(70)
results_cosine = vector_index_cosine.search_vector(query_vector, k=5)
print(results_cosine)



import time

# Function to measure time taken for a function
def measure_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    return result

# Example usage
measure_time(vector_index_cosine.search_vector, query_vector, k=5)


# Function to assess the quality of retrieved vectors
def assess_quality(retrieved_vectors, actual_vectors):
    # Implement your quality assessment logic
    pass

# Example usage
retrieved_vectors = vector_index_cosine.search_vector(query_vector, k=5)
#actual_vectors = [("doc_id_actual_1", 0.2), ("doc_id_actual_2", 0.25), ...]
#quality_score = assess_quality(retrieved_vectors, actual_vectors)
#print(f"Quality Score: {quality_score:.4f}")

#----------------------------------------------------------------------------------

class VectorIndexCosine:
    def __init__(self, dim):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.ids = []

    def add_vector(self, vector, doc_id):
        vector = vector / np.linalg.norm(vector)
        vector = np.array([vector]).astype('float32')
        self.index.add(vector)
        self.ids.append(doc_id)

    def search_vector(self, query_vector, k=5):
        query_vector = query_vector / np.linalg.norm(query_vector)
        query_vector = np.array([query_vector]).astype('float32')
        distances, indices = self.index.search(query_vector, k)
        return list(zip([self.ids[i] for i in indices[0]], distances[0]))

# Example usage
vector_index_cosine = VectorIndexCosine(dim=70)

# Add vectors to the index (replace these with the vectors provided by TAs and their corresponding IDs)
vector_index_cosine.add_vector(np.random.rand(70), "doc_id_1")
vector_index_cosine.add_vector(np.random.rand(70), "doc_id_2")
# Add more vectors...

# Query vector provided by TAs
query_vector_ta = np.random.rand(70)

# Measure time for retrieval
start_time = time.time()
results_cosine = vector_index_cosine.search_vector(query_vector_ta, k=5)
end_time = time.time()

# Assess quality (replace with actual most similar vectors provided by TAs)
actual_most_similar = [("doc_id_actual_1", 0.2), ("doc_id_actual_2", 0.25), ...]

# Quality assessment logic (this is a basic example, adjust as needed)
#def assess_quality(retrieved_vectors, actual_vectors):
    #retrieved_ids = [result[0] for result in retrieved_vectors]
    #actual_ids = [actual[0] for actual in actual_vectors]
    #intersection = set(retrieved_ids) & set(actual_ids)
    #recall = len(intersection) / len(actual_ids)
    #precision = len(intersection) / len(retrieved_ids)
    #return recall, precision

# Evaluate quality
''''recall, precision = assess_quality(results_cosine, actual_most_similar)

# Print results
print(f"Retrieved Vectors: {results_cosine}")
print(f"Actual Most Similar Vectors: {actual_most_similar}")
print(f"Recall: {recall:.4f}, Precision: {precision:.4f}")
print(f"Time taken for retrieval: {end_time - start_time:.4f} seconds")'''
