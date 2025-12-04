import pandas as pd

def preview_tf_idf(path):

    tf_idf = pd.read_csv(path)

    top5 = (
        tf_idf.sort_values(["document", "tfidf"], ascending=[True, False])
        .groupby("document")
        .head(5)
    )

    print(top5)