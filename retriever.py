from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os


def load_chunks(path, size=150):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        print("Run extract.py first to generate the text file.")
        return []

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    words = text.split()
    print(f"Words in file: {len(words)}")

    result = []
    for i in range(0, len(words), size):
        piece = " ".join(words[i: i + size])
        if len(piece.strip()) > 40:
            result.append(piece)

    print(f"Chunks made: {len(result)}")
    return result


def search(question, chunks, top=5):
    if not chunks:
        print("No chunks loaded.")
        return []

    if not question.strip():
        print("Question is empty.")
        return []

    all_text = chunks + [question]

    engine = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2)
    )

    matrix = engine.fit_transform(all_text)

    q_vec = matrix[-1]
    chunk_vecs = matrix[:-1]

    scores = cosine_similarity(q_vec, chunk_vecs)[0]
    top_idx = np.argsort(scores)[::-1][:top]

    found = []
    for idx in top_idx:
        s = float(scores[idx])
        if s > 0.01:
            found.append({
                "text": chunks[idx],
                "score": round(s, 4)
            })

    return found


if __name__ == "__main__":
    chunks = load_chunks("data/science_class10.txt")

    if not chunks:
        print("Could not load file. Stopping.")
    else:
        questions = [
            "What is photosynthesis?",
            "How does the human heart work?",
            "What is Ohm's law?"
        ]

        for q in questions:
            print(f"\nQuestion: {q}")
            results = search(q, chunks, top=3)
            if results:
                print(f"Best match (score {results[0]['score']}):")
                print(results[0]["text"][:200])
            else:
                print("Nothing found.")