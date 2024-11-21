import os
from torch.utils.data import Dataset
from arguments import DataArguments
from transformers import PreTrainedTokenizer, BatchEncoding
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
import datasets
import random
from gensim.models import Word2Vec
import logging
from transformers import DataCollator
from dataclasses import dataclass
import faiss
import numpy as np
logger = logging.getLogger(__name__)
logging.getLogger('gensim').setLevel(logging.ERROR)

class TrainDatasetForCL(Dataset):
    def __init__(
            self,
            args: DataArguments,
            tokenizer: PreTrainedTokenizer
    ):
        if not args.hard_negative:
            data_files = {}
            if args.train_file is not None:
                data_files["train"] = args.train_file
            extension = args.train_file.split(".")[-1]
            if extension == "txt":
                extension = "text"
            if extension == "csv":
                self.datasets = datasets.load_dataset(extension, data_files=data_files, cache_dir="./data/", delimiter=args.delimiter)
            else:
                self.datasets = datasets.load_dataset(extension, data_files=data_files, cache_dir="./data/")
            self.datasets.shuffle(seed=42)
            self.tokenizer = tokenizer
            self.args = args

            self.column_names = self.datasets["train"].column_names
            if len(self.column_names) == 1:
                self.sent1 = self.column_names[0]
                self.sent2 = self.column_names[0]
            elif len(self.column_names) == 2:
                self.sent1 = self.column_names[0]
                self.sent2 = self.column_names[1]
            else:
                raise NotImplementedError("The dataset should have only one or two columns")

            self.total_len = len(self.datasets["train"])
            # print(self.total_len)
            if self.args.delete_word:
                logger.info("Delete words with probability %f", self.args.delete_word_probability)
            if self.args.swap_word:
                logger.info("Swap words with probability %f", self.args.swap_word_probability)
            if self.args.replace_word:
                logger.info("Replace words with probability %f", self.args.replace_word_probability)
                self.FAISS_AVAILABLE = True if self.args.hnsw_index else False
                if self.FAISS_AVAILABLE:
                    if not os.path.exists(self.args.hnsw_index):
                        raise ValueError("hnws index is an invalid path!")
                    self.INDEX = faiss.read_index(self.args.hnsw_index)
                if not self.args.word2vec_model or not os.path.exists(self.args.word2vec_model):
                    raise ValueError("word2vec model is an invalid path!")
                self.WORD2VEC_MODEL = Word2Vec.load(self.args.word2vec_model)
        else:
            self.dataset = datasets.load_dataset("json", data_files=args.train_file, split="train", cache_dir="./data/")
            self.tokenizer = tokenizer
            self.args = args
            self.total_len = len(self.dataset)
        
    def __getitem__(self, item):
        query = self.dataset[item]["query"]
        pos = random.choice(self.dataset[item]["pos"])
        neg = self.dataset[item]["neg"]
        data = [query] + [pos] + neg[:(self.args.grouped_size-1)]
        # data = [query] + [pos] + neg[:10]
        tokenizer_dataset = self.tokenizer(
            data,
            padding="max_length" if self.args.pad_to_max_length else False,
            truncation=True,
            max_length=self.args.max_seq_length,
        )
        return tokenizer_dataset


    def __len__(self):
        return self.total_len
    
    
    def prepare_features(self, examples):
        total = len(examples[self.sent1])
        # Avoid "None" fields 
        for idx in range(total):
            if examples[self.sent1][idx] is None:
                examples[self.sent1][idx] = " "
            if examples[self.sent2][idx] is None:
                examples[self.sent2][idx] = " "
        new_sentences2 = []
        for idx in range(total):
            temp = examples[self.sent2][idx]
            if self.args.delete_word:
                if self.args.delete_word_probability is None:
                    raise "The probability must be provided and the range is [0, 1]."
                elif not 0 <= self.args.delete_word_probability <= 1.0:
                    raise "The probability has to be greater than or equal to 0 and less than or equal to 1."
                temp = self.delete_words(temp, p=self.args.delete_word_probability)
            if self.args.swap_word:
                if self.args.swap_word_probability is None:
                    raise "The probability must be provided and the range is [0, 1]."
                elif not 0 <= self.args.swap_word_probability <= 1.0:
                    raise "The probability has to be greater than or equal to 0 and less than or equal to 1."
                temp = self.swap_words(temp, p=self.args.swap_word_probability)
            if self.args.replace_word:
                if self.args.replace_word_probability is None:
                    raise "The probability must be provided and the range is [0, 1]."
                elif not 0 <= self.args.replace_word_probability <= 1.0:
                    raise "The probability has to be greater than or equal to 0 and less than or equal to 1."
                temp = self.replace_words(temp, p=self.args.replace_word_probability, is_random=False)
            new_sentences2.append(temp)
        sentences = examples[self.sent1] + new_sentences2

        sent_features = self.tokenizer(
            sentences,
            max_length=self.args.max_seq_length,
            truncation=True,
            padding="max_length" if self.args.pad_to_max_length else False,
        )
        features = {}
        for key in sent_features:
            features[key] = [[sent_features[key][i], sent_features[key][i+total]] for i in range(total)]
            
        return features

    # DA
    def delete_words(self, sentence, p=0.1):
        words = self.tokenizer.tokenize(sentence)
        words = [word for word in words if random.random() > p]
        return " ".join(words)

    def swap_words(self, sentence, p=0.1):
        words = self.tokenizer.tokenize(sentence)
        for i in range(1, len(words)):
            if random.random() < p:
                words[i], words[i-1] = words[i-1], words[i]
        return " ".join(words)

    def replace_words(self, sentence, p=0.1, is_random=False):
        words = self.tokenizer.tokenize(sentence)

        for i in range(len(words)):
            if random.random() < p:
                if not is_random:
                    try:
                        if not self.FAISS_AVAILABLE:
                            sim_words = self.WORD2VEC_MODEL.wv.most_similar(words[i].lower(), topn=5)
                            words[i] = sim_words[random.randint(0, 4)][0]
                        else:
                                query_word = words[i].lower()
                                if query_word in self.WORD2VEC_MODEL.wv:
                                    query_vector = np.array([self.WORD2VEC_MODEL.wv[query_word]]).astype('float32')
                                    faiss.normalize_L2(query_vector)
                                    distances, indices = self.INDEX.search(query_vector, 6)
                                    words[i] = self.WORD2VEC_MODEL.wv.index_to_key[indices[0][random.randint(1, 5)]]
                                else:
                                    words[i] = self.WORD2VEC_MODEL.wv.index_to_key[
                                        random.randint(0, len(self.WORD2VEC_MODEL.wv.index_to_key) - 1)]
                    except Exception as e:
                        print(e)
                        words[i] = self.WORD2VEC_MODEL.wv.index_to_key[random.randint(0, len(self.WORD2VEC_MODEL.wv.index_to_key) - 1)]
                else:
                    words[i] = self.WORD2VEC_MODEL.wv.index_to_key[random.randint(0, len(self.WORD2VEC_MODEL.wv.index_to_key) - 1)]
        return " ".join(words)


@dataclass
class CLCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding = True
    max_length = None
    pad_to_multiple_of = None

    def __call__(self, features):
        special_keys = ['input_ids', 'attention_mask', 'token_type_ids']
        bs = len(features)
        if bs > 0:
            num_sent = len(features[0]['input_ids'])
        else:
            return
        flat_features = []
        for feature in features:
            for i in range(num_sent):
                flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in feature})

        batch = self.tokenizer.pad(
            flat_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )


        batch = {k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0] for
                 k in batch}

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        return batch