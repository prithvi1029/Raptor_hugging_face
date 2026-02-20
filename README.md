# 🧠 RAPTOR on PubMedQA (Healthcare Dataset)

This repository implements a **RAPTOR (Recursive Abstractive Processing
for Tree-Organized Retrieval)** pipeline on the **PubMedQA** public
healthcare dataset.

The system:

-   Builds a hierarchical semantic tree over a medical document\
-   Uses UMAP + Gaussian Mixture Models (GMM) for clustering\
-   Generates LLM-based summaries at each tree level\
-   Performs Collapsed Tree Retrieval (retrieving from leaf + summary
    nodes)\
-   Answers 1--2 queries using a HuggingFace-hosted LLM

------------------------------------------------------------------------

## 📌 What is RAPTOR?

RAPTOR is a hierarchical retrieval method that:

1.  Splits a document into chunks\
2.  Embeds them using sentence transformers\
3.  Clusters semantically similar chunks\
4.  Summarizes each cluster using an LLM\
5.  Recursively builds higher-level summaries\
6.  Retrieves context from both raw text and summaries

This enables better long-document reasoning and multi-level semantic
abstraction.

------------------------------------------------------------------------

## 🏥 Dataset

**PubMedQA (pqa_labeled split)**\
A public biomedical question answering dataset containing:

-   Research context passages\
-   Long answers\
-   Final decision labels (yes / no / maybe)

Loaded via:

``` python
load_dataset("pubmed_qa", "pqa_labeled", split="train")
```

------------------------------------------------------------------------

## 🏗️ System Architecture

Document\
↓\
Sentence-aware Chunking\
↓\
Embeddings (multi-qa-mpnet-base-cos-v1)\
↓\
UMAP Dimensionality Reduction\
↓\
GMM Clustering (BIC-selected)\
↓\
LLM Summaries (recursive)\
↓\
Hierarchical Tree\
↓\
Collapsed Tree Retrieval\
↓\
LLM Answer Generation

------------------------------------------------------------------------

## ⚙️ Installation

Install required dependencies:

``` bash
pip install datasets sentence-transformers umap-learn scikit-learn faiss-cpu huggingface_hub
```

------------------------------------------------------------------------

## 🔐 HuggingFace Token (Colab Secret)

This project uses a hidden HuggingFace token via Google Colab secrets.

### In Google Colab:

1.  Click 🔑 Secrets (left sidebar)\
2.  Add key: `HF_TOKEN`\
3.  Paste your HuggingFace token\
4.  Enable notebook access

The code loads it securely using:

``` python
from google.colab import userdata
HF_TOKEN = userdata.get("HF_TOKEN")
```

⚠️ The token is never hardcoded.

------------------------------------------------------------------------

## 🚀 Running the Project

``` python
run_raptor_pubmedqa_single_example(
    example_idx=0,
    queries=None,
    chunk_tokens=100,
    max_levels=3,
    top_k=12,
    max_context_tokens=900,
)
```

### What it does:

-   Loads one PubMedQA example\
-   Builds a RAPTOR tree\
-   Runs 1--2 queries\
-   Retrieves hierarchical context\
-   Generates an LLM answer

------------------------------------------------------------------------

## 📈 Why RAPTOR?

  Feature                   RAG   RAPTOR
  ------------------------- ----- --------
  Flat retrieval            ✅    ❌
  Hierarchical reasoning    ❌    ✅
  Cluster summarization     ❌    ✅
  Multi-level abstraction   ❌    ✅
  Better long-doc QA        ⚠️    ✅

------------------------------------------------------------------------

## 📚 References

-   RAPTOR Paper: Recursive Abstractive Processing for Tree-Organized
    Retrieval\
-   PubMedQA Dataset\
-   Sentence Transformers\
-   UMAP\
-   FAISS

------------------------------------------------------------------------

## 👨‍💻 Author

Abhishek Prithvi Teja Angadala\
AI / ML / LLM Systems
