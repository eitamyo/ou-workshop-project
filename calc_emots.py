import pandas as pd
from transformers import pipeline
from tqdm.auto import tqdm

dialog_texts = pd.read_pickle('./data/dialog_texts')

classifier = pipeline("text-classification",
                      model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

def get_emotion_feats(classifier_res):
    return {x['label']: x['score'] for x in classifier_res[0]}

def get_emotions(texts):
    for i in tqdm(range(texts.shape[0])):
        row = texts.iloc[i]
        feats = get_emotion_feats(classifier(row.text))
        feats['movie_id'] = row.movie_id
        feats['text'] = row.text
        yield feats

movie_emots = pd.DataFrame()

done_movies = 0
curr_mid = None
for emot in get_emotions(dialog_texts):
    if curr_mid != emot['movie_id']:
        done_movies += 1
        curr_mid = emot['movie_id']
    edf = pd.DataFrame([emot])
    movie_emots = pd.concat([movie_emots, edf])
    movie_emots.to_pickle('./data/movie_emots')
