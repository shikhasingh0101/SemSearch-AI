# SemSearch AI — Group 3 Mini Project

## Project objective
Build a semantic search and recommendation system that retrieves text records by meaning using dense sentence embeddings and cosine similarity, and compare it with a traditional keyword baseline.

## Dataset
`data/records.csv` contains 60 curated sample records across internships, courses, job descriptions and research topics. Fields:
- `record_id`: unique record identifier
- `category`: record type
- `title`: record title
- `description`: descriptive text
- `text`: title + description used for embedding

## Technology
- Python
- Sentence Transformers: `sentence-transformers/all-MiniLM-L6-v2`
- FAISS `IndexFlatIP` for exact nearest-neighbour search
- L2-normalized embeddings; inner product therefore equals cosine similarity
- TF-IDF + cosine similarity as the keyword-search baseline
- Streamlit UI

## Setup (Mac / Windows / Linux)
Python 3.10–3.12 is recommended. Python 3.11 is a good choice.

```bash
cd SemSearch_AI_Group3
python3 -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate      # Windows PowerShell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first run downloads the pre-trained embedding model. Internet access is needed unless the model is already cached.

## Run command-line search
```bash
python src/semantic_search.py --query "machine learning internship" --top-k 5
python src/semantic_search.py --query "AI trainee opportunities" --top-k 5
python src/semantic_search.py --query "learn how computers understand human language" --top-k 5
```

## Run the web application
```bash
streamlit run src/app.py
```
Open the local URL printed by Streamlit (normally http://localhost:8501).

## Workflow
1. Read the 60 records from CSV.
2. Combine title and description into one text field.
3. Load the pre-trained `all-MiniLM-L6-v2` sentence encoder.
4. Generate one dense vector per record.
5. Normalize vectors to unit length.
6. Add vectors to FAISS `IndexFlatIP`.
7. Encode the user query and normalize it.
8. Search FAISS for top-k highest inner products (cosine similarity).
9. Run a TF-IDF keyword baseline on the same query.
10. Compare top-five results and inspect relevance.

## Important implementation detail
For unit-normalized vectors:
`cosine_similarity(a, b) = dot(a, b)`.
FAISS `IndexFlatIP` returns inner products, so normalized embeddings make its scores cosine similarities. This is exact flat search, appropriate for this small 60-record dataset. For much larger collections, an approximate index or managed vector database can reduce retrieval latency.

## Evaluation plan
Use at least these test queries:
- Q1: `machine learning internship`
- Q2: `AI trainee opportunities` (paraphrase of Q1)
- Q3: `learn how computers understand human language`
- Q4: `beginner role creating intelligent software`
- Q5: `vector search and meaning-based document retrieval`

For each query, inspect the top five records from semantic and TF-IDF search. Record whether relevant records appear, compare rank order, and calculate Precision@5 if relevance labels are assigned. Do not claim a numeric accuracy without manually labelled relevance judgements or a defined ground-truth set.

## Screenshots to capture after running
1. Streamlit home page showing the query field.
2. Semantic results for `AI trainee opportunities`.
3. Compare-both view for `machine learning internship`.
4. Terminal output from a CLI search.

The `screenshots/` folder is reserved for your actual screenshots. The report includes a workflow diagram and a screenshot checklist. Do not use mock results as measured experimental results.

## Limitations
- The sample dataset is small and curated, not a production-scale or externally validated corpus.
- Embedding similarity can return semantically related but not necessarily correct records.
- The model may underperform on specialised terms, abbreviations or languages outside its strengths.
- The flat FAISS index performs exact search but scans all vectors; very large datasets may need approximate indexing.
- Human relevance judgements are needed for rigorous evaluation.

## Project structure
```
SemSearch_AI_Group3/
├── data/records.csv
├── src/semantic_search.py
├── src/app.py
├── docs/
├── screenshots/
├── requirements.txt
└── README.md
```
