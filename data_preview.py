import pandas as pd

def preview_tf_idf_top(path):
    # Function to display top scores for each document in dataframe format.

    tf_idf = pd.read_csv(path)

    top5 = (
        tf_idf.sort_values(["document", "tfidf"], ascending=[True, False])
        .groupby("document")
        .head(5)
    )
    print("Top 5 scores per document.")
    print(top5)

def preview_tf_idf_bottom(path):
    # Function to display top scores for each document in dataframe format.

    tf_idf = pd.read_csv(path)

    bottom5 = (
        tf_idf.sort_values(["document", "tfidf"], ascending=[True, False])
        .groupby("document")
        .tail(5)
    )
    print("Bottom 5 scores per document.")

    print(bottom5)