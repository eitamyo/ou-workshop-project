import pandas as pd
from transformers import pipeline
from tqdm.auto import tqdm

dialog_texts = pd.read_pickle('./data/dialog_texts')

classifier = pipeline("text-classification",
                      model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

meta_cols = ['movie_id', 'title', 'year', 'rating', 'no. votes', 'genres']
meta = pd.read_table(
    './datasets/movie-dialog-corpus/movie_titles_metadata.tsv', sep='\t', header=None, names=meta_cols, index_col='movie_id')

def get_emotion_feats(classifier_res: list[list[dict]]):
    return {x['label']: x['score'] for x in classifier_res[0]}


def get_dialog_emotions(df):
    for i in range(df.shape[0]):
        dialog = df.iloc[i]
        feats = get_emotion_feats(classifier(dialog.text))
        feats['movie_id'] = dialog.movie_id
        feats['text'] = dialog.text
        yield feats

def get_movie_emots(movie_id):
    return pd.DataFrame(get_dialog_emotions(dialog_texts[dialog_texts['movie_id'] == movie_id]))

movie_emots = pd.read_pickle('data/movie_emots_1')

for i in tqdm(range(100, meta.shape[0])):
    movie = meta.iloc[i]
    movie_id = meta.index[i]
    emots = get_movie_emots(movie_id)
    movie_emots = pd.concat([movie_emots, emots])
    movie_emots.to_pickle('data/movie_emots_1')