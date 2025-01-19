import time

import numpy as np

from mistralai import Mistral
from tqdm import tqdm

from constants import API_KEY, CHUNK_SIZE
from mistral_service import run_mistral
from utils import get_text_embedding, save_embedding, search_for_retrieval


def main():
    client = Mistral(api_key=API_KEY)
    mode = input("Select mode (1: Embeddings Generation, 2: Answer Generation): ")
    text_file = input(
        "Enter the file path of your textbook to retrieve model output (default: 'textbook.txt'): "
    )
    text_file = text_file if text_file else "textbook.txt"

    try:
        with open(text_file, "r") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Text file '{text_file}' not found.")
        return
    chunk_size = CHUNK_SIZE
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    if mode == "1":
        embeddings_generation_mode(client, chunks)
    elif mode == "2":
        answer_generation_mode(client, chunks)
    else:
        print("Invalid mode selected.")
        return


def embeddings_generation_mode(client, chunks):

    for i, chunk in enumerate(tqdm(chunks)):
        embedding = get_text_embedding(client, chunk)
        save_embedding(embedding, i + 1)
        time.sleep(5)

    print("Embeddings generated and saved to disk.")


def answer_generation_mode(client, chunks):
    question_file = input(
        "Enter the file path of your question to retrieve model output (default: 'question.txt'): "
    )
    question_file = question_file if question_file else "question.txt"

    try:
        with open(question_file, "r") as f:
            question = f.read()
    except FileNotFoundError:
        print(f"Question file '{question_file}' not found.")
        return

    question_embeddings = np.array([get_text_embedding(client, question)])

    retrieved_chunks = search_for_retrieval(question_embeddings, chunks)
    prompt = f"""
    Context information is below.
    ---------------------
    {retrieved_chunks}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {question}
    Answer:
    """
    time.sleep(5)
    answer = run_mistral(client, prompt)

    print(answer)


if __name__ == "__main__":
    main()
