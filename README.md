# 🚀 Wikipedia English Corpus Information Retrieval Engine 🧐

This repository contains an **Information Retrieval (IR) engine** designed to search through the English Wikipedia corpus. It uses algorithms such as **BM25**, **PageRank**, and **Cosine Similarity** to optimize document retrieval based on relevance and popularity.

## 👥 Authors
- **Itay Carmel** (📧: carmelit@post.bgu.ac.il)
- **Shay Sitri** (📧: sitri@post.bgu.ac.il)

## 📝 Project Overview

The engine searches through a large corpus of Wikipedia documents, optimizing for both retrieval speed and accuracy. It is designed to handle millions of documents using advanced IR techniques.

### 🛠️ Key Components:
1. **📚 Corpus Construction**: Tested initially on a small corpus and later expanded to a large corpus of 6 million documents.
2. **🔍 Search Algorithms**:
   - **BM25**: Evaluates document relevance based on term frequency and inverse document frequency.
   - **Cosine Similarity**: Measures similarity between query and document vectors.
   - **PageRank**: Ranks documents based on interlinking and popularity.

### 🧪 Methodology
Optimizations were applied to filter documents using anchor texts, titles, and body content. Algorithms were combined to improve relevance with the following weight distribution:
- **BM25**: 40%
- **Cosine Similarity**: 10%
- **Page Views**: 20%

## 📊 Performance Evaluation
The engine was tested using MAP@40 for precision and retrieval time for speed. Notable queries like “Rick and Morty” returned high relevance with a MAP@40 score of 0.93.

## ⚙️ Setup and Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/shaySitri/IRproject.git
   ```

2. **🔧 Environment Setup**:
   Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **🏃 Running the Engine**:
   Provide the corpus and index files in the configuration.

4. **☁️ Google Cloud Buckets**:
   The dataset and indexes are stored in [Google Buckets](https://console.cloud.google.com/storage/browser/ir208909416).

## 🔑 Key Files and Functions

### `search_frontend.py`:
   - **`search_with_bm25()`**: Computes the BM25 relevance score.
   - **`search_with_pagerank()`**: Ranks documents using the PageRank algorithm.
   - **`combine_results()`**: Merges results from BM25 and PageRank.

### `inverted_index_gcp.py`:
   - **`build_inverted_index()`**: Creates the inverted index for fast lookups.
   - **`query_index()`**: Retrieves documents from the index based on a query.

## 🔮 Future Work

- **🧠 Query Expansion**: Integrate Word2Vec to enhance query matching.
- **📈 Scalability**: Refine the index handling for even larger corpora.

## 🏁 Conclusion

This project demonstrates the power of combining various IR algorithms to create a robust and scalable search engine. Despite scaling challenges, the final solution provides a balance between retrieval speed and relevance.
