import pandas as pd

def basic_eda(df):
    print("\n--- SHAPE OF DATA ---")
    print(df.shape)

    print("\n--- NULL VALUES ---")
    print(df.isnull().sum())

    print("\n--- SAMPLE ROWS ---")
    print(df.head())

    print("\n--- CLASS BALANCE ---")
    print(df['label'].value_counts())

    print("\n--- AVERAGE TEXT LENGTH ---")
    df['text_length'] = df['text'].apply(lambda x: len(x.split()))
    print(df.groupby('label')['text_length'].mean())

if __name__ == "__main__":
    df = pd.read_csv("data/combined.csv")
    basic_eda(df)
