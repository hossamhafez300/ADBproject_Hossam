import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import pickle as pi
import time

# Helper functions and classes

def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    dot_product = np.dot(a, b.T)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

def sort_vectors_by_cosine_similarity(vectors, reference_vector):
    cos_similarities = cosine_similarity(reference_vector, vectors)
    sorted_indices = np.argsort(cos_similarities)
    sorted_indices = np.flip(sorted_indices, axis=1)
    return sorted_indices

class IVFile(object):
    def __init__(self, partitions: int, vectors: np.ndarray):
        self.partitions = partitions
        self.vectors = vectors

    def clustering(self):
        kmeans = MiniBatchKMeans(n_clusters=self.partitions)
        assignments = kmeans.fit_predict(self.vectors)
        centroids = kmeans.cluster_centers_
        self.data = (centroids, assignments)
        index = [[] for _ in range(self.partitions)]
        for n, k in enumerate(assignments):
            index[k].append(self.vectors[n])
        centroid_assignment = {}
        x = 0
        for k in index:
            byte_file = np.asarray(k)
            centroid_assignment[str(centroids[x])] = f"./data/file{x}.npy"
            file = open(f"./data/file{x}.npy", "a")
            np.save(f"./data/file{x}.npy", byte_file)
            file.close()
            x += 1
            k = None
        self.assigments = centroid_assignment
        self.vectors = None  # <===
        return self.assigments

    def get_closest_centroids(self, vector: np.ndarray, K: int):
        centroids = self.data[0]
        index = sort_vectors_by_cosine_similarity(centroids, vector)
        centroids = centroids[index]
        return centroids[0][len(centroids) - K - 1 :]

    def get_cluster_data(self, centroid: np.ndarray, vector: np.ndarray, K: int):
        data = np.load(self.assigments[str(centroid)])
        index = sort_vectors_by_cosine_similarity(data, vector)
        data = data[index]
        return data[0][len(data) - K - 1 :]

    def cluster_data(self, centroids: np.ndarray):
        return [np.load(self.assigments[str(centroid)]) for centroid in centroids]

    def get_closest_k_neighbors(self, vector: np.ndarray, K: int):
        centroids = self.get_closest_centroids(vector, K)
        closest = []
        for centroid in centroids:
            closest.append(self.get_cluster_data(centroid, vector, K))
        closest = np.asarray(closest)
        closest = closest.reshape((closest.shape[0] * closest.shape[1], closest.shape[2]))
        indices = sort_vectors_by_cosine_similarity(closest, vector)
        return [closest[i] for i in indices[0][len(closest) - K - 1 :]]

    def get_K_closest_neighbors_given_centroids(self, centroids: np.ndarray, vector: np.ndarray, K: int):
        closest = []
        for centroid in centroids:
            closest.append(self.get_cluster_data(centroid, vector, K))
        closest = np.asarray(closest)
        closest = closest.reshape((closest.shape[0] * closest.shape[1], closest.shape[2]))
        indices = sort_vectors_by_cosine_similarity(closest, vector)
        return [closest[i] for i in indices[0][len(closest) - K - 1 :]]

    def get_K_closest_neigbors_inside_centroid_space(self, centroids: np.ndarray, vector: np.ndarray, K: int):
        centroids = self.get_closest_centroids(vector, 1)
        closest = []
        for centroid in centroids:
            closest.append(self.get_cluster_data(centroid, vector, K))
        closest = np.asarray(closest)
        closest = closest.reshape((closest.shape[0] * closest.shape[1], closest.shape[2]))
        indices = sort_vectors_by_cosine_similarity(closest, vector)
        return [closest[i] for i in indices[0][len(closest) - K - 1:]]

class IVFile_optimized(object):
    def __init__(self, vectors: np.ndarray):
        self.partitions = np.ceil(len(vectors) / np.sqrt(len(vectors))) * 3
        self.vectors = vectors

    def clustering(self):
        (centroids, assignments) = kmeans2(self.vectors, int(self.partitions))
        self.data = (centroids, assignments)
        index = [[] for _ in range(int(self.partitions))]
        for n, k in enumerate(assignments):
            index[k].append(self.vectors[n])
        centroid_assignment = {}
        x = 0
        for k in index:
            byte_file = np.asarray(k)
            centroid_assignment[str(centroids[x])] = f"./data/file{x}.npy"
            np.save(f"./data/file{x}.npy", byte_file)
            x += 1
        self.assigments = centroid_assignment
        self.vectors = None  # <===
        return self.assigments

    def get_closest_centroids(self, vector: np.ndarray, K: int):
        centroids = self.data[0]
        index = sort_vectors_by_cosine_similarity(centroids, vector)
        centroids = centroids[index]
        return centroids[0][len(centroids) - K - 1:]

    def get_cluster_data(self, centroid: np.ndarray, vector: np.ndarray, K: int):
        data = np.load(self.assigments[str(centroid)])
        index = sort_vectors_by_cosine_similarity(data, vector)
        data = data[index]
        return data[0][len(data) - K - 1:]

    def cluster_data(self, centroids: np.ndarray):
        return [np.load(self.assigments[str(centroid)]) for centroid in centroids]

    def get_closest_k_neighbors(self, vector: np.ndarray, K: int):
        centroids = self.get_closest_centroids(vector, K)
        closest = []
        for centroid in centroids:
            closest.append(self.get_cluster_data(centroid, vector, K))
        closest = np.asarray(closest)
        closest = closest.reshape((closest.shape[0] * closest.shape[1], closest.shape[2]))
        indices = sort_vectors_by_cosine_similarity(closest, vector)
        return [closest[i] for i in indices[0][len(closest) - K - 1:]]

    def get_K_closest_neighbors_given_centroids(self, centroids: np.ndarray, vector: np.ndarray, K: int):
        closest = []
        for centroid in centroids:
            closest.append(self.get_cluster_data(centroid, vector, K))
        closest = np.asarray(closest)
        closest = closest.reshape((closest.shape[0] * closest.shape[1], closest.shape[2]))
        indices = sort_vectors_by_cosine_similarity(closest, vector)
        return [closest[i] for i in indices[0][len(closest) - K - 1:]]

    def get_K_closest_neigbors_inside_centroid_space(self, centroids: np.ndarray, vector: np.ndarray, K: int):
        centroids = self.get_closest_centroids(vector, 1)
        closest = []
        for centroid in centroids:
            closest.append(self.get_cluster_data(centroid, vector, K))
        closest = np.asarray(closest)
        closest = closest.reshape((closest.shape[0] * closest.shape[1], closest.shape[2]))
        indices = sort_vectors_by_cosine_similarity(closest, vector)
        return [closest[i] for i in indices[0][len(closest) - K - 1:]]

class VecDB(object):
    def __init__(self, vectors: np.ndarray, partitions: int, db_path: str):
        self.vecdb = IVFile_optimized(vectors)
        self.partitions = partitions
        self.db_path = db_path
        self.load_db()

    def load_db(self):
        if os.path.exists(self.db_path):
            loaded_data = np.load(self.db_path, allow_pickle=True)
            self.vecdb.assigments = loaded_data['assignments']
            self.vecdb.data = loaded_data['data']

    def save_db(self):
        np.savez(self.db_path, assignments=self.vecdb.assigments, data=self.vecdb.data)

    def insert_vectors(self):
        start_time = time.time()
        result = self.vecdb.clustering()
        end_time = time.time()
        insertion_time = end_time - start_time
        print(f"Insertion time: {insertion_time} seconds")
        return result

    def retrieve_vectors(self, query_vector, K):
        start_time = time.time()
        result = self.vecdb.get_K_closest_neigbors_inside_centroid_space([], query_vector, K)
        end_time = time.time()
        retrieval_time = end_time - start_time
        print(f"Retrieval time: {retrieval_time} seconds")
        return result


# Part 1: Database Size 10K
num_vectors_10k = 10000
vector_dim = 70
sample_data_10k = np.random.rand(num_vectors_10k, vector_dim)

# Specify the path to save the database
db_path_10k = "/content/vecdb_10k.npz"

# Create VecDB instance
vecdb_10k = VecDB(sample_data_10k, partitions=3, db_path=db_path_10k)

# Insert vectors into the database
assignments_10k = vecdb_10k.insert_vectors()

# Sample query vector
query_vector_10k = np.random.rand(vector_dim)

# Retrieve vectors from the database
result_10k = vecdb_10k.retrieve_vectors(query_vector_10k, K=3)
print("Retrieved Vectors:", result_10k)


# Part 2: Database Sizes 100K and More
# You should provide pre-generated database and index files
# Modify the paths accordingly
db_path_100k = "/content/vecdb_100k.npz"
db_path_1m = "/content/vecdb_1m.npz"
db_path_5m = "/content/vecdb_5m.npz"
db_path_10m = "/content/vecdb_10m.npz"
db_path_15m = "/content/vecdb_15m.npz"
db_path_20m = "/content/vecdb_20m.npz"

# Load pre-generated databases
vecdb_100k = VecDB(vectors=None, partitions=3, db_path=db_path_100k)
vecdb_1m = VecDB(vectors=None, partitions=3, db_path=db_path_1m)
vecdb_5m = VecDB(vectors=None, partitions=3, db_path=db_path_5m)
vecdb_10m = VecDB(vectors=None, partitions=3, db_path=db_path_10m)
vecdb_15m = VecDB(vectors=None, partitions=3, db_path=db_path_15m)
vecdb_20m = VecDB(vectors=None, partitions=3, db_path=db_path_20m)

# Retrieve vectors from the pre-generated databases
query_vector_large = np.random.rand(vector_dim)
result_100k = vecdb_100k.retrieve_vectors(query_vector_large, K=3)
result_1m = vecdb_1m.retrieve_vectors(query_vector_large, K=3)
result_5m = vecdb_5m.retrieve_vectors(query_vector_large, K=3)
result_10m = vecdb_10m.retrieve_vectors(query_vector_large, K=3)
result_15m = vecdb_15m.retrieve_vectors(query_vector_large, K=3)
result_20m = vecdb_20m.retrieve_vectors(query_vector_large, K=3)

print("Retrieved Vectors (100K):", result_100k)
print("Retrieved Vectors (1M):", result_1m)
print("Retrieved Vectors (5M):", result_5m)
print("Retrieved Vectors (10M):", result_10m)
print("Retrieved Vectors (15M):", result_15m)
print("Retrieved Vectors (20M):", result_20m)
