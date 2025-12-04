import data_loader
import processor
import data_preview

corpus = data_loader.load_corpus_directly()

print(corpus)

cleaned_corpus = processor.preprocess_documents(corpus)

corpus_tokenised = processor.tokenize_corpus(cleaned_corpus)

corpus_tokenised_minus_stopwords = processor.remove_stopwords(corpus_tokenised)

corpus_tokenised_minus_stopwords_stemmed = processor.stem_corpus(corpus_tokenised_minus_stopwords)

tf = processor.export_token_frequencies(corpus_tokenised_minus_stopwords_stemmed, 'token_frequencies.csv')

idf = processor.compute_idf(corpus_tokenised_minus_stopwords_stemmed, 'idf.csv')

tf_idf_items, tf_idf_corpus = processor.compute_tfidf_from_csv('token_frequencies.csv','idf.csv','tf_idf.csv')

data_preview.preview_tf_idf('tf_idf.csv')