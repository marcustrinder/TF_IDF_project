# TF-IDF Implementation Task Brief

## Overview
Implement the **Term Frequency–Inverse Document Frequency (TF‑IDF)** algorithm **from scratch in Python**, without using any pre‑built TF‑IDF libraries.  
You *may* use libraries such as **NLTK** for tokenization, stemming, and other preprocessing tasks.

---

## Background

TF‑IDF is a numerical statistic that indicates how important a term is within a document relative to a corpus.

### **1. Term Frequency (TF)**
Number of times a term appears in a document divided by the total number of terms in the document.

### **2. Inverse Document Frequency (IDF)**
A logarithmic measure of how common or rare a term is across the corpus:

```
IDF(term) = log( total_documents / documents_containing_term )
```

### **TF‑IDF Calculation**
```
TF‑IDF(term, document, corpus) = TF(term, document) * IDF(term, corpus)
```

---

## Requirements

Your implementation must include:

1. **A tokenizer**  
   Function that converts a document into a list of words.

2. **Term Frequency function**  
   Computes TF for each term in a given document.

3. **Inverse Document Frequency function**  
   Computes IDF for each unique term across the corpus.

4. **TF‑IDF function**  
   Calculates TF‑IDF scores for every term in every document.

Assume the corpus is a list of text documents.

---

## Corpus Provided

You should use the following corpus (5 documents):

- A legend of **Sir Lancelot**
- The story of **Jack and the Beanstalk**
- The fairy tale **Snow White**
- A summary of **Harry Potter**
- A story about **Shrek and Princess Fiona**

*(Full text provided in original document.)*

---

## Evaluation

Your solution will be assessed on:

- Correctness of TF‑IDF calculations  
- Code clarity and readability  
- Efficiency  
- Your ability to explain your implementation and reasoning

---

## Additional Notes

- Use **Python**.
- **Do not use pre‑built TF‑IDF** implementations.
- You *may* use libraries for:
  - Tokenization
  - Stemming
  - General data manipulation
