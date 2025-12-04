import data_loader
import processor
import data_preview

# Load the corpus
corpus = data_loader.load_corpus_directly()

print(corpus)

# Apply preprocessing step 1 (lowercase and remove punctuation)
cleaned_corpus = processor.preprocess_documents(corpus)

# Tokenise corpus
corpus_tokenised = processor.tokenize_corpus(cleaned_corpus)

# Remove stopwords
corpus_tokenised_minus_stopwords = processor.remove_stopwords(corpus_tokenised)

# Apply stemming (Porter)
corpus_tokenised_minus_stopwords_stemmed = processor.stem_corpus(corpus_tokenised_minus_stopwords)

# Compute TF, IDF and TF_IDF
tf = processor.export_token_frequencies(corpus_tokenised_minus_stopwords_stemmed, 'token_frequencies.csv')

idf = processor.compute_idf(corpus_tokenised_minus_stopwords_stemmed, 'idf.csv')

tf_idf_items, tf_idf_corpus = processor.compute_tfidf_from_csv('token_frequencies.csv','idf.csv','tf_idf.csv')

# Preview outputs
data_preview.preview_tf_idf('tf_idf.csv')
