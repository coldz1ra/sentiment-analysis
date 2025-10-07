import re, pandas as pd, sys, os
inp = 'data/reviews.csv'
outp = 'data/reviews_mapped.csv'
df = pd.read_csv(inp)
cols = [c.strip() for c in df.columns]

text_candidates = ['text','review','content','reviewText','body','comment','message','summary','title','reviews']
label_candidates = ['label','sentiment','polarity','target','stars','rating','class','y','score']

# выбрать колонку текста
text_col = next((c for c in text_candidates if c in cols), None)
if text_col is None:
    obj_cols = [c for c in cols if df[c].dtype == 'O']
    if obj_cols:
        text_col = max(obj_cols, key=lambda c: df[c].astype(str).str.len().mean())
    else:
        sys.exit("Не нашёл текстовую колонку — пришли имена столбцов.")

# выбрать колонку метки
label_col = next((c for c in label_candidates if c in cols), None)
if label_col is None:
    few = sorted([(c, df[c].nunique()) for c in cols if c != text_col], key=lambda x: x[1])
    label_col = next((c for c,u in few if u<=5), None)
    if label_col is None:
        sys.exit("Не нашёл колонку меток — пришли имена столбцов.")

out = pd.DataFrame({'text': df[text_col].astype(str), 'label': df[label_col]})

def norm_label(v):
    s = str(v).strip().lower()
    if re.fullmatch(r'-?\d+(\.\d+)?', s):
        x = float(s)
        if 1 <= x <= 5:
            if x == 3: return None
            return 'positive' if x >= 4 else 'negative'
        if x in (0.0, 1.0):  return 'positive' if x==1.0 else 'negative'
        if x in (-1.0, 1.0): return 'positive' if x==1.0 else 'negative'
    if s in ('pos','positive','positif','positivo','позитив','положительный','good','great','true','yes'):
        return 'positive'
    if s in ('neg','negative','negatif','negativo','негатив','отрицательный','bad','poor','false','no'):
        return 'negative'
    return s

out['label'] = out['label'].map(norm_label)
out = out[out['label'].isin(['positive','negative'])].dropna(subset=['text','label'])
out = out[out['text'].str.strip().astype(bool)]

print("Колонки исходного файла:", cols)
print("Определил text_col =", text_col, "; label_col =", label_col)
print("Распределение классов:", out['label'].value_counts().to_dict())

os.makedirs('data', exist_ok=True)
out.to_csv(outp, index=False)
print("Сохранено в", outp)

