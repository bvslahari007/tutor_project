# Context-Aware AI Tutor for Rural Education

This is our project built as part of the Intel Unnati Program. The idea is simple: instead of sending a student's full textbook to an AI model every time they ask a question, we pick only the relevant parts and send those. This reduces computation, saves bandwidth, and makes the system usable in low-connectivity areas like rural schools.

## What the project does

A student types a question. The system finds the most relevant chunks of text from a science textbook, compresses that context to remove redundant information, and then passes it to a language model (Grok Llama 3.3 70B Versatile) to generate an answer. The whole thing runs as a simple web app built with Streamlit.

## Why we built it this way

Running a large AI model with a full textbook as input every time is expensive and slow. In areas with poor internet or limited computing resources, that is not practical. Our approach is to prune the context first, so only what actually matters reaches the model. This is the core idea we are exploring.

## Project structure

```
tutor_project/
    data/
        science_class10.txt        # NCERT-style science content, manually prepared
    retriever.py                   # Chunks the text and finds relevant sections using TF-IDF
    pruner.py                      # Compresses the retrieved context using ScaleDown API
    app.py                         # Streamlit UI that connects everything together
    requirements.txt
    README.md
```

## How it works

1. The textbook content is split into small overlapping chunks
2. When a question comes in, TF-IDF is used to rank those chunks by relevance
3. The top chunks are passed to the pruner, which removes redundant content
4. The compressed context and the original question go to Grok Llama 3.3 70B Versatile
5. The answer is shown in the Streamlit app

## Tech stack

- Python
- scikit-learn (TF-IDF retrieval)
- ScaleDown API (context compression)
- Grok Llama 3.3 70B Versatile (answer generation)
- Streamlit (frontend)

## Setup

Clone the repo and install dependencies:

```bash
git clone https://github.com/your-username/tutor_project.git
cd tutor_project
pip install -r requirements.txt
```

Add your API keys. Create a `.env` file in the root folder:

```
SCALEDOWN_API_KEY=your_key_here
GROK_API_KEY=your_key_here
```

Run the app:

```bash
streamlit run app.py
```

## Team

This project is divided across three members:

- Member A - retriever.py, knowledge base preparation
- Member B - pruner.py, context compression
- Member C - app.py, Streamlit UI
