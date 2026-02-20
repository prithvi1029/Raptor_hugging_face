🧠 RAPTOR on PubMedQA (Healthcare Dataset)

This repository implements a RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) pipeline on the PubMedQA public healthcare dataset.

The system:

Builds a hierarchical semantic tree over a medical document

Uses UMAP + Gaussian Mixture Models (GMM) for clustering

Generates LLM-based summaries at each tree level

Performs Collapsed Tree Retrieval (retrieving from leaf + summary nodes)

Answers 1–2 queries using a HuggingFace-hosted LLM

📌 What is RAPTOR?

RAPTOR is a hierarchical retrieval method that:

Splits a document into chunks

Embeds them using sentence transformers

Clusters semantically similar chunks

Summarizes each cluster using an LLM

Recursively builds higher-level summaries

Retrieves context from both raw text and summaries

This allows:

Better long-document reasoning

Multi-level semantic abstraction

More efficient retrieval

🏥 Dataset

PubMedQA (pqa_labeled split)
A public biomedical question answering dataset containing:

Research context passages

Long answers

Final decision labels (yes / no / maybe)

Loaded via:

load_dataset("pubmed_qa", "pqa_labeled", split="train")
🏗️ System Architecture
Document
   ↓
Sentence-aware Chunking
   ↓
Embeddings (multi-qa-mpnet-base-cos-v1)
   ↓
UMAP Dimensionality Reduction
   ↓
GMM Clustering (BIC-selected)
   ↓
LLM Summaries (recursive)
   ↓
Hierarchical Tree
   ↓
Collapsed Tree Retrieval
   ↓
LLM Answer Generation
⚙️ Installation

Install required dependencies:

pip install datasets sentence-transformers umap-learn scikit-learn faiss-cpu huggingface_hub
🔐 HuggingFace Token (Colab Secret)

This project uses a hidden HuggingFace token via Google Colab secrets.

In Google Colab:

Click 🔑 Secrets (left sidebar)

Add key:

HF_TOKEN

Paste your HuggingFace token

Enable notebook access

The code automatically loads it via:

from google.colab import userdata
HF_TOKEN = userdata.get("HF_TOKEN")

⚠️ The token is never hardcoded in the notebook.

🚀 Running the Project

At the bottom of the script:

run_raptor_pubmedqa_single_example(
    example_idx=0,
    queries=None,
    chunk_tokens=100,
    max_levels=3,
    top_k=12,
    max_context_tokens=900,
)
What it does:

Loads one PubMedQA example

Builds a RAPTOR tree

Runs 1–2 queries:

Original dataset question

"What is the main conclusion and evidence from this study?"

Retrieves context from hierarchical tree

Generates an LLM answer

📊 Key Components
🔹 Sentence-aware Chunking

Preserves semantic boundaries while respecting token limits.

🔹 UMAP (Cosine Metric)

Reduces embedding dimensionality before clustering.

🔹 Gaussian Mixture Model (BIC-selected)

Automatically selects optimal cluster count.

🔹 Recursive Summarization

Uses:

mistralai/Mistral-7B-Instruct-v0.2

via HuggingFace Inference API.

🔹 Collapsed Tree Retrieval

Retrieves from:

Leaf nodes

Summary nodes

All hierarchy levels

🧪 Example Output
DATASET: PubMedQA (pqa_labeled)
Example index: 0
Question: ...
Gold label: yes

RAPTOR built.
Total nodes: 42
Root node ids: [35]

QUERY 1: ...
Retrieved nodes: 7

LLM Answer:
Decision: yes
Justification: ...
📈 Why RAPTOR?

Compared to flat RAG:

Feature	RAG	RAPTOR
Flat retrieval	✅	❌
Hierarchical reasoning	❌	✅
Cluster summarization	❌	✅
Multi-level abstraction	❌	✅
Better long-doc QA	⚠️	✅
🛠️ Customization

You can tune:

chunk_tokens
max_levels
top_k
max_context_tokens

You can also:

Swap the LLM

Swap embedding model

Run on multiple dataset examples

Extend to full benchmarking

📚 References

RAPTOR Paper: Recursive Abstractive Processing for Tree-Organized Retrieval

PubMedQA Dataset

Sentence Transformers

UMAP

FAISS

👨‍💻 Author

Abhishek Prithvi Teja Angadala
AI / ML / LLM Research & Systems
