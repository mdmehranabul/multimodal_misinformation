import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_train_label_distributions(train_df, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)

    # 2-Way Label Distribution
    plt.figure(figsize=(10, 4))
    sns.countplot(x='2_way_label_name', data=train_df)
    plt.title("2-Way Label Distribution (Train Set)")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig(os.path.join(save_dir, "2_way_label_distribution.png"))
    plt.show(block=False)

    # 6-Way Label Distribution
    plt.figure(figsize=(12, 5))
    sns.countplot(x='6_way_label_name', data=train_df, order=train_df['6_way_label_name'].value_counts().index)
    plt.title("6-Way Label Distribution (Train Set)")
    plt.xticks(rotation=45)
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig(os.path.join(save_dir, "6_way_label_distribution.png"))
    plt.show(block=False)

    # Post Year Distribution
    plt.figure(figsize=(10, 4))
    sns.histplot(train_df['created_datetime'].dt.year, kde=False, bins=20)
    plt.title("Post Distribution by Year (Train Set)")
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.savefig(os.path.join(save_dir, "post_year_distribution.png"))
    plt.show(block=False)