import logging
import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import faiss
import numpy as np
import os
if not os.path.exists("./word2vec_model"):
    os.mkdir("./word2vec_model")
import time
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# read dataset
with open(r"./data/all_data.txt") as f:
    sentences = f.readlines()
# training
processed_sentences = [simple_preprocess(sentence) for sentence in sentences]
model = Word2Vec(min_count=1, window=5, vector_size=100, workers=12)
model.build_vocab(processed_sentences)
model.train(processed_sentences, total_examples=model.corpus_count, epochs=20)
model.save(r"./word2vec_model/word2vec.model")
# # load model
model = Word2Vec.load(r"./word2vec_model/word2vec.model")
words = list(model.wv.index_to_key)
vectors = np.array([model.wv[word] for word in words]).astype('float32')
d = vectors.shape[1]
faiss.normalize_L2(vectors)
try:
    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = 100
    index.add(vectors)
except Exception as e:
    logging.error(f"Error creating or adding vectors to HNSW index: {e}")
    exit(1)

# 保存索引
try:
    faiss.write_index(index, r"./word2vec_model/hnsw_index.faiss")
except Exception as e:
    logging.error(f"Error saving HNSW index: {e}")
    exit(1)

