import os
import streamlit as st
from src.utils import load_artifact
from src.config import cfg

st.set_page_config(page_title='Sentiment Analysis Demo', page_icon='🙂', layout='centered')

st.title('Sentiment Analysis — Demo')
st.write('TF-IDF + Logistic Regression on your reviews.')

@st.cache_resource
def load_model():
    model = load_artifact(os.path.join(cfg.model_dir, 'model_logreg.joblib'))
    le = load_artifact(os.path.join(cfg.model_dir, 'label_encoder.joblib'))
    return model, le

model_loaded = False
try:
    model, le = load_model()
    model_loaded = True
except Exception as e:
    st.warning('Model not found. Train it first with `make train`.')
    st.stop()

txt = st.text_area('Enter a review:', height=150, placeholder='Type or paste a review...')
btn = st.button('Predict sentiment')

if btn and txt.strip():
    pred_idx = model.predict([txt])[0]
    label = le.inverse_transform([pred_idx])[0]
    st.subheader(f'Prediction: {label}')
    if hasattr(model.named_steps['clf'], 'predict_proba'):
        probs = model.predict_proba([txt])[0]
        for i, cls in enumerate(le.classes_):
            st.write(f'{cls}: {probs[i]:.3f}')
