# Information Retrieval System for Kannada Crime Records

An end-to-end **Information Retrieval (IR) and NLP system** built from first principles to process, index, and query **Kannada-language crime records** scraped from the Udupi Police Department website.  
The project demonstrates how high-quality retrieval forms a critical foundation for modern AI systems such as **LLMs and Retrieval-Augmented Generation (RAG)**, especially in **vernacular languages**.

---

## 📌 Project Overview

This project processes **13,000+ Kannada crime descriptions** and enables users to retrieve relevant records using:
- Classical keyword-based search
- Ranked retrieval using TF-IDF
- Semantic search using multilingual transformer embeddings

The system highlights challenges and solutions in handling **non-English, Unicode-rich text** for real-world AI applications.

---

## 🎯 Objectives

- Build an end-to-end IR pipeline **from scratch**
- Handle **Kannada Unicode text (U+0C80 – U+0CFF)** reliably
- Compare **symbolic (keyword-based)** and **neural (embedding-based)** retrieval approaches
- Demonstrate the role of retrieval as a backbone for **LLMs and RAG systems**

---

## 🗂 Dataset

- **Source:** Udupi Police Department public crime records
- **Language:** Kannada
- **Size:** 13,000+ crime descriptions
- **Nature:** Unstructured vernacular text

---

## 🛠️ Technologies Used

- **Programming:** Python  
- **Libraries & Frameworks:**  
  - PyTorch  
  - Scikit-learn  
  - Pandas  
  - NLTK  
  - Sentence-Transformers  
- **NLP & IR Techniques:**  
  - Unicode Normalization  
  - Tokenization  
  - Inverted Positional Index  
  - TF-IDF  
  - Cosine Similarity  
  - Multilingual Transformer Embeddings (MiniLM)  
- **Analysis & Visualization:**  
  - PCA  
  - K-Means Clustering  

---

## 🔍 System Architecture

### 1. Vernacular Text Preprocessing
- Custom Unicode-aware normalization for Kannada text
- Handling complex characters, diacritics, and multi-whitespace issues
- Tokenization designed specifically for Kannada morphology

### 2. Classical Information Retrieval
- Implemented an **Inverted Positional Index** from scratch
- Supports:
  - Boolean queries (AND, OR, NOT)
  - Phrase-based search
- Enables **transparent and explainable retrieval**

### 3. Ranked Retrieval (Vector Space Model)
- Implemented **TF-IDF** to capture local and global term importance
- Used **Cosine Similarity** for relevance-based document ranking

### 4. Semantic Search
- Integrated **multilingual transformer embeddings** (paraphrase-multilingual-MiniLM)
- Overcomes limitations of exact keyword matching
- Enables concept-level and context-aware retrieval

### 5. Custom Embedding Exploration
- Designed a lightweight PyTorch model using:
  - Token embeddings
  - Mean pooling
- Explored representation learning and contrastive loss for efficient retrieval

### 6. Analysis & Visualization
- Compared keyword-based vs embedding-based retrieval
- Used **PCA** for dimensionality reduction
- Applied **K-Means clustering** to identify crime pattern groupings

---

## 📊 Key Results

- Demonstrated strong performance of **semantic retrieval** over exact keyword matching
- Showed how classical IR remains essential for **precision and explainability**
- Highlighted retrieval as a foundational step for **LLM-powered systems in low-resource languages**

---

## 🚀 Future Work

- Integrate with a **Retrieval-Augmented Generation (RAG)** pipeline
- Add a web-based query interface
- Fine-tune transformer models on domain-specific Kannada data
- Extend to multilingual crime analytics

---

## 📁 Repository Structure

