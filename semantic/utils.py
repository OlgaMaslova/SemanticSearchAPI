import nltk
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util

nltk.download('punkt')
model = SentenceTransformer("allenai-specter")


def create_vocabulary(text: str):
    sentence_list = nltk.tokenize.sent_tokenize(text)
    return sentence_list


def create_embeddings_for_sentences(sentences: list):
    return np.stack([model.encode(sentence) for sentence in sentences])


def create_embedding_for_sentence(sentence: str):
    return model.encode(sentence)


def semantic_search(query_embedding, sentence_embeddings):
    return util.semantic_search(query_embedding, sentence_embeddings, top_k=1)


def is_valid_query(query: str):
    if len(query) == 0:
        return False
    if not re.match("[A-Za-z0-9_-]+", query):
        return False
    return True
