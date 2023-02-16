import pandas as pd
import numpy as np
from transformers import pipeline
from tqdm.auto import tqdm

dialog_texts = pd.read_pickle('./data/dialog_texts')

extractor = pipeline("feature-extraction",
                      model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

def get_text_features(text):
    features = extractor(text)
    return np.mean(features[0], axis=0)

def get_emot_feats(df, from_i):
    for i in tqdm(range(from_i, len(df))):
        dt = dialog_texts.iloc[i]
        feats = get_text_features(dt.text)
        fdict = {'f%d' % i:feats[i] for i in range(len(feats))}
        fdict['movie_id'] = dt.movie_id
        yield fdict

movie_emot_feats = pd.read_pickle('./data/movie_emot_feats')

from_i = len(movie_emot_feats)
save_interval = 100
i = 0
for emot in get_emot_feats(dialog_texts, from_i):
    movie_emot_feats = pd.concat([movie_emot_feats, pd.DataFrame([emot])])
    i += 1
    if i == save_interval:
        i = 0
        movie_emot_feats.to_pickle('./data/movie_emot_feats')

movie_emot_feats.to_pickle('./data/movie_emot_feats')