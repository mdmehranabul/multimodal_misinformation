import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def clean_labels(series):
    """
    Cleans classification labels like '1: Real' -> 1
    Ignores rows with invalid/error values.
    """
    cleaned = []
    for val in series:
        try:
            num = str(val).split(":")[0].strip()
            if num.isdigit():
                cleaned.append(int(num))
            else:
                cleaned.append(None)
        except:
            cleaned.append(None)
    return cleaned

def evaluate(file_path, label_col, pred_col="gemini_classification"):
    df = pd.read_csv(file_path)

    # clean predictions
    df["clean_pred"] = clean_labels(df[pred_col])

    # drop rows with invalid predictions
    df = df.dropna(subset=["clean_pred"])

    df["clean_pred"] = df["clean_pred"].astype(int)
    df[label_col] = df[label_col].astype(int)

    y_true = df[label_col].tolist()
    y_pred = df["clean_pred"].tolist()

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    print(f"Results for {file_path} (label = {label_col}):")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"Rows used : {len(df)}")
    print("-"*40)

# Run for both datasets
evaluate("gemini_output_2_way.csv", label_col="2_way_label")
evaluate("gemini_output_6_way.csv", label_col="6_way_label")