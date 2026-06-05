import pandas as pd

def load_data(fake_path, true_path):
              fake_df = pd.read_csv(fake_path)
              true_df = pd.read_csv(true_path)

              fake_df["label"] = 1
              true_df["label"] = 0

              df = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)
              return df


if __name__ == "__main__":
            df = load_data("data/Fake.csv", "data/True.csv")
            print(df.head())
            print(df['label'].value_counts())

            df.to_csv("data/combined.csv", index=False)

    