import numpy as np

from playgenie.data.dataset import DataSet
import pandas as pd
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer


class QdrantSearch:
    def __init__(self):
        self.files = DataSet(folder_path='./data/song_embeddings.parquet').dataset.files
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = QdrantClient(":memory:")
        self.idx = 0

        self.client.create_collection(
            collection_name="songs",
            vectors_config=models.VectorParams(
                size=self.encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
                distance=models.Distance.COSINE,
            ),
        )


        for file in self.files:
            df = pd.read_parquet(file)
            self.idx = self.idx +len(df)
            self.client.upload_collection(
            collection_name="songs",
            vectors=df['embedding'].values,
            payload=[{"song": s} for s in df['song'].tolist()],
            ids= [self.idx+i for i in df.index.values.tolist()],)
            break

    def get_item(self,song_vector:np.ndarray)->np.ndarray:

        search_res = self.client.query_points(collection_name="songs",query=song_vector
                                              ,with_vectors=True,limit=1).points
        return np.array(search_res)