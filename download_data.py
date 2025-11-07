import pandas as pd
from datasets import load_dataset

print("--- Starting Data Download from Hugging Face ---")

# 1. Load the 'ag_news' dataset
# This dataset is all REAL news, categorized.
# It has two splits, 'train' and 'test'. We'll combine them.
print("Downloading 'ag_news' dataset...")
dataset_train = load_dataset("ag_news", split='train')
dataset_test = load_dataset("ag_news", split='test')

# 2. Convert to Pandas DataFrames
df_train = dataset_train.to_pandas()
df_test = dataset_test.to_pandas()

# 3. Combine them into one big DataFrame
df_ag_news = pd.concat([df_train, df_test])

print(f"Total articles downloaded: {len(df_ag_news)}")
print("Categories found (0=World, 1=Sports, 2=Business, 3=Sci/Tech):")
print(df_ag_news['label'].value_counts())

# 4. We only care about the 'text' column for our 'True' file.
# Let's save just the text.
new_true_df = df_ag_news[['text']]

# 5. Save this as our new, diverse 'True' file
new_true_df.to_csv("True_diverse.csv", index=False)

print("\n--- SUCCESS! ---")
print("Saved 127,600 diverse real news articles to 'True_diverse.csv'.")
print("You can now modify your train.py to use this file.")