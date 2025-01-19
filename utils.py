from pathlib import Path

import faiss
import numpy as np

from tqdm import tqdm

from constants import MODEL_EMBED_NAME


def get_text_embedding(client, input):
    embeddings_batch_response = client.embeddings.create(
        model=MODEL_EMBED_NAME, inputs=input
    )
    return embeddings_batch_response.data[0].embedding


def search_for_retrieval(question_embeddings, chunks):
    text_embeddings = []

    pathlist = Path("embeddings").rglob("*.npy")

    for path in tqdm(pathlist):
        filename = str(path)
        text_embeddings.append(np.load(filename, allow_pickle=True))
    text_embeddings = np.array(text_embeddings)

    d = text_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(text_embeddings)
    _, I = index.search(question_embeddings, k=2)
    retrieved_chunk = [chunks[i] for i in I.tolist()[0]]
    return retrieved_chunk


def save_embedding(embedding, chunk_id: int):
    np.save(f"embeddings/text_embedding_{chunk_id}", embedding, allow_pickle=True)
