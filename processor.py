import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer
import csv
from collections import Counter
import math
import string

STOPWORDS = {
    "a","about","above","after","again","against","all","am","an","and","any","are",
    "aren't","as","at","be","because","been","before","being","below","between","both",
    "but","by","can't","cannot","could","couldn't","did","didn't","do","does","doesn't",
    "doing","don't","down","during","each","few","for","from","further","had","hadn't",
    "has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here",
    "here's","hers","herself","him","himself","his","how","how's","i","i'd","i'll",
    "i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's","me",
    "more","most","mustn't","my","myself","no","nor","not","of","off","on","once",
    "only","or","other","ought","our","ours","ourselves","out","over","own","same",
    "shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such",
    "than","that","that’s","the","their","theirs","them","themselves","then","there",
    "there's","these","they","they'd","they'll","they're","they've","this","those",
    "through","to","too","under","until","up","very","was","wasn't","we","we'd","we'll",
    "we're","we've","were","weren't","what","what's","when","when's","where","where's",
    "which","while","who","who's","whom","why","why's","with","won't","would","wouldn't",
    "you","you'd","you'll","you're","you've","your","yours","yourself","yourselves"
}



def preprocess_documents(corpus):
    """
    Lowercase and remove punctuation from each document in the corpus.
    Returns a new cleaned corpus.
    """
    cleaned_corpus = []
    
    # Create translation table once (much faster than regex)
    translator = str.maketrans('', '', string.punctuation)

    for doc in corpus:
        # Lowercase
        doc = doc.lower()
        # Remove punctuation
        doc = doc.translate(translator)
        cleaned_corpus.append(doc)

    return cleaned_corpus


def tokenize_corpus(corpus):
    """
    Takes a list of documents (strings) and returns
    a list of token lists (list of words per document).
    Uses NLTK's TreebankWordTokenizer.
    """
    tokenizer = TreebankWordTokenizer()

    tokenized_docs = []
    for doc in corpus:
        tokens = tokenizer.tokenize(doc)
        # to lower
        tokens = [t.lower() for t in tokens]
        tokenized_docs.append(tokens)
    return tokenized_docs


def remove_stopwords(tokenised_corpus):
    """
    Remove stopwords using a built-in stopword list (editable).
    """
    cleaned_corpus = []
    for tokens in tokenised_corpus:
        filtered = [t for t in tokens if t not in STOPWORDS]
        cleaned_corpus.append(filtered)
    return cleaned_corpus


def stem_corpus(tokenised_corpus):
    """
    Apply Porter stemming to each token in each document.
    
    Input:
        tokenised_corpus = [ [token1, token2, ...], [...], ... ]
    
    Output:
        stemmed_corpus   = same structure, with stemmed tokens.
    """
    ps = PorterStemmer()
    stemmed_corpus = []

    for tokens in tokenised_corpus:
        stemmed_tokens = [ps.stem(t) for t in tokens]
        stemmed_corpus.append(stemmed_tokens)

    return stemmed_corpus



def export_token_frequencies(tokenised_corpus, output_path):
    """
    Creates a CSV file containing:
    document_index, word, frequency
    sorted by frequency (descending) for each document.
    """
    tf = []

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["document", "word", "frequency"])

        

        for doc_index, tokens in enumerate(tokenised_corpus):
            # Count frequencies in the document
            freq = Counter(tokens)

            # Sort by descending frequency
            sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)

            # Write to CSV
            for word, count in sorted_freq:
                writer.writerow([doc_index, word, count])
                tf.append([doc_index, word, count])

    return tf


def compute_idf(tokenised_corpus, output_path):
    """
    Computes IDF for each unique word in the corpus.
    Returns a dict: { word: idf_value }
    """
    N = len(tokenised_corpus)  # total documents

    # Step 1: count documents containing each term
    doc_count = {}  # df(term)

    for tokens in tokenised_corpus:
        unique_tokens = set(tokens)   # only count once per document
        for token in unique_tokens:
            if token not in doc_count:
                doc_count[token] = 0
            doc_count[token] += 1

    # Step 2: compute IDF for each word
    idf = {}
    for word, df in doc_count.items():
        idf[word] = math.log(N / df)

    # --- Step 3: Sort IDF values (descending) ---
    sorted_idf = sorted(idf.items(), key=lambda x: x[1], reverse=True)

    # --- Step 4: Write to CSV ---
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "idf"])

        for word, value in sorted_idf:
            writer.writerow([word, value])

    return idf

def load_idf_from_csv(idf_path):
    """
    Load IDF values from a CSV file with columns: word, idf
    Returns a dict: { word: idf_value }
    """
    idf = {}
    with open(idf_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row["word"]
            value = float(row["idf"])
            idf[word] = value
    return idf

def compute_tfidf_from_csv(token_freq_path, idf_path, output_path):
    """
    Compute TF-IDF using:
      - token_frequencies.csv (document, word, frequency)
      - idf.csv (word, idf)

    Writes a CSV with:
      document, word, tfidf

    Returns:
      tfidf_corpus: dict[doc_id] -> dict[word] = tfidf
    """

    # --- Step 1: Load IDF values ---
    idf = load_idf_from_csv(idf_path)

    # --- Step 2: Load token frequencies and accumulate per-document totals ---
    doc_term_counts = {}   # {doc_id: {word: freq}}
    doc_total_terms = {}   # {doc_id: total_terms_in_doc}

    with open(token_freq_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = int(row["document"])
            word = row["word"]
            freq = int(row["frequency"])

            if doc_id not in doc_term_counts:
                doc_term_counts[doc_id] = {}
                doc_total_terms[doc_id] = 0

            doc_term_counts[doc_id][word] = freq
            doc_total_terms[doc_id] += freq

    # --- Step 3: Compute TF-IDF for each doc/word ---
    tfidf_corpus = {}  # {doc_id: {word: tfidf}}

    with open(output_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["document", "word", "tfidf"])

        for doc_id, term_counts in doc_term_counts.items():
            total_terms = doc_total_terms[doc_id]
            tfidf_corpus[doc_id] = {}

            # Compute TF-IDF per word
            # Optionally sort by descending TF-IDF
            tfidf_items = []

            for word, freq in term_counts.items():
                tf = freq / total_terms
                word_idf = idf.get(word, 0.0)  # 0 if not found
                tfidf = tf * word_idf
                tfidf_corpus[doc_id][word] = tfidf
                tfidf_items.append((word, tfidf))

            # Sort terms in this doc by descending tfidf
            tfidf_items.sort(key=lambda x: x[1], reverse=True)

            # Write rows
            for word, value in tfidf_items:
                writer.writerow([doc_id, word, value])

    return tfidf_items, tfidf_corpus
