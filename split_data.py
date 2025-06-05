import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    df = pd.read_csv('data/labels.csv')
    
    # First split train+val and test (80/20)
    train_val_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df['flower_type'], random_state=1)
    
    # Then split train and val (80/20)
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.2, stratify=train_val_df['flower_type'], random_state=1)
    
    # Save
    train_df.to_csv('data/train.csv', index=False)
    val_df.to_csv('data/val.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    print("Data split completed and saved to data/train.csv, val.csv, test.csv")

if __name__ == '__main__':
    main()