import sys
import pandas as pd


def main():
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    df = pd.read_csv(in_path)
    cmap = {c.lower(): c for c in df.columns}
    text_col = cmap.get('text', cmap.get('review', cmap.get('content', list(df.columns)[0])))
    label_col = cmap.get('label', cmap.get('sentiment', cmap.get('target', list(df.columns)[1])))
    df = df.rename(columns={text_col: 'text', label_col: 'label'})
    df['text'] = df['text'].astype(str).str.strip()
    df['label'] = df['label'].astype(str).str.strip()
    df[['text','label']].to_csv(out_path, index=False)
    print(f"mapped -> {out_path}")


if __name__ == '__main__':
    main()
