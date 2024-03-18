# Analisis Sentimen pada Tweet Terkait Pilpres 2019 di Indonesia

Proyek ini bertujuan untuk melakukan analisis sentimen menggunakan data Twitter yang berkaitan dengan pemilihan presiden 2019 di Indonesia. Dataset yang digunakan terdiri dari 1815 baris data tweet.

## Model yang Digunakan
- Random Forest
- LSTM (Long Short-Term Memory)

## Eksperimen Random Forest
1. Random Forest dengan Word2Vec Skip Gram (Baseline dan Tuned) dengan stopwords
2. Random Forest dengan Word2Vec CBOW (Baseline dan Tuned) dengan stopwords
3. Random Forest dengan Word2Vec Skip Gram (Baseline dan Tuned) tanpa stopwords
4. Random Forest dengan Word2Vec CBOW (Baseline dan Tuned) tanpa stopwords

## Eksperimen LSTM
1. LSTM dengan pemrosesan teks
2. LSTM tanpa pemrosesan teks

## Struktur Proyek

```
├── dataset
│   └── tweet.csv
├── models
│   ├── model_rf_sg_tuned_sw.joblib
│   ├── word2vec_model_sg_min_8_window_6_sw.bin
│   └── [dan file model lainnya]
├── plugins
│   └── text_prep_function.py
├── resources
│   ├── emoji_dictionary.json
│   └── slang_word_dictionary.json
├── .gitignore
├── app_flask.py
├── app_streamlit.py
├── notebook_eda.ipynb
├── notebook_modelling.ipynb
└── README.md
```

- **dataset**: Direktori untuk menyimpan file dataset.
  - **tweet.csv**: File CSV yang berisi data tweet.
- **models**: Direktori untuk menyimpan model-model yang telah dilatih.
  - **model_rf_sg_tuned_sw.joblib**: Model Random Forest dengan Word2Vec Skip Gram yang telah dituning dengan stopwords.
  - **word2vec_model_sg_min_8_window_6_sw.bin**: Model Word2Vec dengan Skip Gram yang ditrain pada data dengan stopwords.
- **plugins**: Direktori untuk menyimpan plugin atau modul tambahan.
  - **text_prep_function.py**: Modul untuk fungsi pemrosesan teks.
- **resources**: Direktori untuk menyimpan berkas-berkas sumber daya tambahan.
  - **emoji_dictionary.json**: Kamus emoji.
  - **slang_word_dictionary.json**: Kamus kata slang.
- **.gitignore**: file yang berisi daftar file dan direktori yang diabaikan oleh Git.
- **app_flask.py (backend)**: file Python yang berisi backend aplikasi menggunakan Flask.
- **app_streamlit.py (frontend)**: file Python yang berisi frontend aplikasi menggunakan Streamlit.
- **notebook_eda.ipynb**: Notebook Jupyter yang berisi analisis eksplorasi data.
- **notebook_modelling.ipynb**: Notebook Jupyter yang berisi proses pemodelan.
- **README.md**: file yang berisi deskripsi proyek dan dokumentasi singkat.

## Model Terbaik
Model dengan performa terbaik adalah Random Forest dengan Word2Vec Skip Gram yang telah dituning dengan stopwords, mencapai metrik kinerja berikut:
- Train Accuracy: 71%
- Validation Accuracy: 61%
- Test Accuracy: 55%

## Deployment
Deployment menggunakan library flask dan streamlit
