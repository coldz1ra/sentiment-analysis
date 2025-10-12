import pandas as pd
from collections import Counter
df=pd.read_csv('data/reviews_mapped.csv')
df['text']=df['text'].astype(str).str.strip()
df['label']=df['label'].astype(str).str.strip()
n=len(df)
lbl=Counter(df['label'])
print(f'total={n}')
for k,v in lbl.items():
    print(f'{k}={v}')
