import pandas as pd
import numpy as np
import os
import re
import joblib

from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from plugins.text_prep_function import EMOJI_DICT
from plugins import text_prep_function as tp

current_dir = os.getcwd()
model_folder = os.path.join(current_dir, "models")
template_folder = os.path.join(current_dir, "templates")

from flask import Flask, request, app, jsonify, url_for, render_template
app=Flask(__name__, template_folder=template_folder)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# rf_model_sg_tuned = joblib.load(os.path.join(model_folder, 'model_rf_sg_tuned.joblib'))
# model_w2v_sg = Word2Vec.load(model_folder, 'word2vec_model_sg_min_8_window_6.bin')

rf_model_sg_tuned = joblib.load(r"C:\Users\alfan\Bootcamp NLP Indonesia AI\pilpres-sentiment-analysis\models\model_rf_sg_tuned_sw.joblib")
model_w2v_sg = Word2Vec.load(r"C:\Users\alfan\Bootcamp NLP Indonesia AI\pilpres-sentiment-analysis\models\word2vec_model_sg_min_8_window_6_sw.bin")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        json_data = request.json['text']
        df = pd.DataFrame([json_data], columns=['tweet'])

        special_character_list = list()
        for text in df.tweet.values:
            non_ascii_characters = re.findall(r'[^\x00-\x7F]+', text)

            for c in non_ascii_characters:
                if c not in special_character_list:
                    special_character_list.append(c)
        
        index = list()
        special_character = list()
        label = list()
        for s in special_character_list:
            for idx,text in enumerate(df.tweet.values):
                if s in text:
                    index.append(idx)
                    special_character.append(s)
                    label.append(df.iloc[idx, 0])

        emoji_list = pd.DataFrame({
        'row': index,
        'special_character': special_character,
        'label': label
        })

        for idx in emoji_list.index:
            try:
                row = emoji_list.iloc[idx, 0]
                text = df.iloc[row, 1]
                word = emoji_list.iloc[idx, 1]
                replacement = EMOJI_DICT[word]
                df.iloc[row, 1] = text.replace(word, replacement)
            except:
                print(word, 'Cannot be found the associated emoji or text')

        emoji_list['emoji'] = emoji_list['special_character']

        for idx in emoji_list.index:
            text = emoji_list.iloc[idx, 3]
            replacement = EMOJI_DICT[text]
            emoji_list.iloc[idx, 3] = text.replace(text, replacement)

        #Text cleaning
        df['remove_unrelevant_emoji'] = df['tweet'].apply(tp.remove_emojis)
        df['emojis'] = df['remove_unrelevant_emoji'].apply(tp.extract_emojis)
        df['tweet_clean'] = df['remove_unrelevant_emoji'].apply(tp.clean_tweet)
        df['tweet_proc_punct'] = df['tweet_clean'].apply(tp.process_punctuation)
        df['tweet_no_punct'] = df['tweet_proc_punct'].apply(tp.remove_punctuation)
        df['tweet_no_sc'] = df['tweet_no_punct'].apply(tp.remove_special_characters)
        df['tweet_lowercase'] = df['tweet_no_sc'].apply(tp.lowercase)
        df['tweet_no_slang'] = df['tweet_lowercase'].apply(tp.process_slang)
        df['tweet_no_number'] = df['tweet_no_slang'].apply(tp.remove_specific_numbers)
        df['tweet_no_extra_spaces'] = df['tweet_no_number'].apply(tp.remove_extra_spaces)

        # Text normalization
        exclude_words = ['asian', 'setuju']
        df['tweet_lem'] = df['tweet_no_extra_spaces'].apply(lambda x: tp.custom_lemmatization(x, exclude_words))
        df['final_tweet'] = df['tweet_lem'] + ' ' + df['emojis']

        # Text tokenization
        df['tweet_tokenization'] = df['final_tweet'].apply(word_tokenize)

        # Word embedding
        sentences = df['tweet_tokenization'].tolist()
        sentences_embedding = tp.mean_vector_df_val(model_w2v_sg, sentences)
        
        # Prediction
        mapping_dict = {0: 'positif', 1: 'netral', 2: 'negatif'}
        prediction_probabilities = rf_model_sg_tuned.predict_proba(sentences_embedding)
        result_probabilities = {mapping_dict[i]: round(prob, 4) for i, prob in enumerate(prediction_probabilities[0])}

        return jsonify({"sentiment probabilities": result_probabilities})

        # return jsonify({"sentimen" :result_category})
    
    except Exception as e:
        # Print the full exception details for debugging
        import traceback
        traceback.print_exc()

        # Return an error response
        return jsonify({"error": str(e)})

if __name__=="__main__":
    app.run(debug=True)