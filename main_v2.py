import data_loader
import processor
import data_preview

# Decide which preprocessing steps to run
# 'c'  = clean (lowercase + remove punctuation)
# 'sw' = remove stopwords
# 'st' = stemming
preproc = ['c', 'sw', 'st'] 
#preproc = []

corpus = data_loader.load_corpus_directly()
print(corpus)

# This will hold the suffix for file names, e.g. "_c_sw_st"
suffix_parts = []

# ---- 1. Optional cleaning step ----
current_corpus = corpus
if 'c' in preproc:
    current_corpus = processor.preprocess_documents(current_corpus)
    suffix_parts.append('c')

# ---- 2. Tokenisation (always needed) ----
corpus_tokenised = processor.tokenize_corpus(current_corpus)

# ---- 3. Optional stopword removal ----
if 'sw' in preproc:
    corpus_tokenised = processor.remove_stopwords(corpus_tokenised)
    suffix_parts.append('sw')

# ---- 4. Optional stemming ----
if 'st' in preproc:
    corpus_tokenised = processor.stem_corpus(corpus_tokenised)
    suffix_parts.append('st')

# ---- 5. Build filename suffix ----
suffix = ''
if suffix_parts:
    suffix = '_' + '_'.join(suffix_parts)

tf_path   = f'outputs/token_frequencies{suffix}.csv'
idf_path  = f'outputs/idf{suffix}.csv'
tfidf_path = f'outputs/tf_idf{suffix}.csv'

# ---- 6. TF, IDF, TF-IDF using the processed tokens ----
tf = processor.export_token_frequencies(corpus_tokenised, tf_path)
idf = processor.compute_idf(corpus_tokenised, idf_path)

tf_idf_items, tf_idf_corpus = processor.compute_tfidf_from_csv(
    tf_path,
    idf_path,
    tfidf_path
)

data_preview.preview_tf_idf_top(tfidf_path)
data_preview.preview_tf_idf_bottom(tfidf_path)

