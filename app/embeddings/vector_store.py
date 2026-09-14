import numpy as np

class InMemoryVectorStore:
    def __init__(self):
        self.vectors = None
        self.metadata = []

    def add(self,vectors: list[list[float]], metadata: list[dict]) -> None:
        if len(vectors) != len(metadata):
            raise ValueError("Vectors and metadata must have the same length. Number of Vectors must match number of metadata records.")
        if not vectors:
            return
        self.vectors = np.array(vectors,dtype=np.float32)
        self.metadata = metadata

    def search(self,query_vector: list[float], top_k: int = 5) -> list[dict]:
        if self.vectors is None or len(self.metadata) == 0:
            return []

        query_vector = np.array(query_vector, dtype=np.float32)
        similarities = np.dot(self.vectors, query_vector)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_k = min(top_k, len(self.metadata))
        results = []

        for index in top_indices:
            results.append(
                {
                    "metadata": self.metadata[index],
                    "similarity": float(similarities[index]),
                }
            )
        return results