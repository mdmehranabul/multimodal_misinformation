import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you've already run:
# train_df = load_and_clean_data(TRAIN_PATH, sample_size=30000, seed=42)

# Label Distributions (2-way)
print("2-Way Label Distribution (Sampled Train):")
print(train_df['2_way_label_name'].value_counts())

# Label Distributions (6-way)
print("\n6-Way Label Distribution (Sampled Train):")
print(train_df['6_way_label_name'].value_counts())

# Temporal Distribution
train_df['year'] = train_df['created_datetime'].dt.year
print("\nPosts by Year:")
print(train_df['year'].value_counts().sort_index())

# Optional plots
plt.figure(figsize=(10, 4))
sns.countplot(x='2_way_label_name', data=train_df)
plt.title("2-Way Label Distribution (Train Sample)")
plt.savefig("2_way_label_distribution.png")
plt.show()

plt.figure(figsize=(12, 5))
sns.countplot(x='6_way_label_name', data=train_df, order=train_df['6_way_label_name'].value_counts().index)
plt.xticks(rotation=45)
plt.title("6-Way Label Distribution (Train Sample)")
plt.savefig("6_way_label_distribution.png")
plt.show()

plt.figure(figsize=(10, 4))
sns.histplot(train_df['year'], bins=range(2008, 2021), discrete=True)
plt.title("Post Year Distribution (Train Sample)")
plt.xlabel("Year")
plt.ylabel("Count")
plt.savefig("post_year_distribution.png")
plt.show()