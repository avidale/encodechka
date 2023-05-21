import json
import os
import pandas as pd
import time
from sklearn.model_selection import train_test_split

from .task_specific_evaluators import eval_pairs_multiotput, eval_pairs_clf_multiotput, eval_accuracy_multiotput, \
    eval_auc_multiotput, evaluate_ner_embedder

DATA_PATH_NAME = 'ENCODECHKA_DATA_PATH'
SENTENCE_TASK = 'sentence'
WORD_TASK = 'word'
# os.environ[DATA_PATH_NAME] = '.'


class EncoderTask:
    TASK_TYPE = SENTENCE_TASK

    def eval(self, embedder, name):
        raise NotImplementedError()


def find_file(filename):
    if not os.path.exists(filename):
        if os.environ.get(DATA_PATH_NAME):
            p2 = os.path.join(os.environ[DATA_PATH_NAME], filename)
            if os.path.exists(p2):
                return p2
        p2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        if os.path.exists(p2):
            return p2
    return filename


class STSBTask(EncoderTask):
    def __init__(self, filename='data/stsb_dev.csv'):
        self.data = pd.read_csv(find_file(filename))
        self.full_cache = {}
        self.score_cache = {}

    def eval(self, embedder, name):
        results = eval_pairs_multiotput(
            embedder, self.data, text1='sentence1', text2='sentence2', y_col='similarity_score'
        )
        max_score = max(results.values())
        self.full_cache[name] = results
        self.score_cache[name] = max_score
        return max_score, results


class ParaphraserTask(EncoderTask):
    def __init__(self, filename='data/para_gold.csv'):
        self.data = pd.read_csv(find_file(filename))
        self.full_cache = {}
        self.score_cache = {}

    def eval(self, embedder, name):
        results = eval_pairs_multiotput(embedder, self.data)
        max_score = max(results.values())
        self.full_cache[name] = results
        self.score_cache[name] = max_score
        return max_score, results


class XnliTask(EncoderTask):
    def __init__(self, filename='data/xnli_ru.csv'):
        self.data = pd.read_csv(find_file(filename))
        self.full_cache = {}
        self.score_cache = {}

    def eval(self, embedder, name):
        results = eval_pairs_clf_multiotput(
            embedder, self.data, text1='sentence1', text2='sentence2', y_col='gold_label'
        )
        max_score = max(results.values())
        self.full_cache[name] = results
        self.score_cache[name] = max_score
        return max_score, results


class SentimentTask(EncoderTask):
    def __init__(
            self,
            train_filename='data/SentiRuEval2016_small_train.csv',
            test_filename='data/SentiRuEval2016_small_test.csv'
    ):
        self.train_data = pd.read_csv(find_file(train_filename))
        self.test_data = pd.read_csv(find_file(test_filename))
        self.full_cache = {}
        self.score_cache = {}

    def eval(self, embedder, name):
        results = eval_accuracy_multiotput(embedder, self.train_data, self.test_data)
        max_score = max(results.values())
        self.full_cache[name] = results
        self.score_cache[name] = max_score
        return max_score, results


class ToxicityTask(EncoderTask):
    def __init__(self, train_filename='data/toxic_train.csv', test_filename='data/toxic_test.csv'):
        self.train_data = pd.read_csv(find_file(train_filename))
        self.test_data = pd.read_csv(find_file(test_filename))
        self.full_cache = {}
        self.score_cache = {}

    def eval(self, embedder, name):
        results = eval_auc_multiotput(embedder, self.train_data, self.test_data, y_col='toxic')
        max_score = max(results.values())
        self.full_cache[name] = results
        self.score_cache[name] = max_score
        return max_score, results


class InappropriatenessTask(EncoderTask):
    def __init__(self, train_filename='data/inappropriate_train.csv', test_filename='data/inappropriate_test.csv'):
        self.train_data = pd.read_csv(find_file(train_filename))
        self.test_data = pd.read_csv(find_file(test_filename))
        self.full_cache = {}
        self.score_cache = {}

    def eval(self, embedder, name):
        results = eval_auc_multiotput(embedder, self.train_data, self.test_data, y_col='y')
        max_score = max(results.values())
        self.full_cache[name] = results
        self.score_cache[name] = max_score
        return max_score, results


class IntentsTask(EncoderTask):
    def __init__(self, train_filename='data/intents_train.csv', test_filename='data/intents_test.csv'):
        self.train_data = pd.read_csv(find_file(train_filename))
        self.test_data = pd.read_csv(find_file(test_filename))
        self.full_cache = {}
        self.score_cache = {}

    def eval(self, embedder, name):
        results = eval_accuracy_multiotput(embedder, self.train_data, self.test_data, y_col='label')
        max_score = max(results.values())
        self.full_cache[name] = results
        self.score_cache[name] = max_score
        return max_score, results


class IntentsXTask(EncoderTask):
    def __init__(self, train_filename='data/intents_train.csv', test_filename='data/intents_test.csv'):
        self.train_data = pd.read_csv(find_file(train_filename))
        self.test_data = pd.read_csv(find_file(test_filename))
        self.full_cache = {}
        self.score_cache = {}

    def eval(self, embedder, name):
        results = eval_accuracy_multiotput(
            embedder, self.train_data, self.test_data, y_col='label', x_col='answer', x_col_test='text'
        )
        max_score = max(results.values())
        self.full_cache[name] = results
        self.score_cache[name] = max_score
        return max_score, results


class NERTask(EncoderTask):
    TASK_TYPE = WORD_TASK

    def __init__(self, filename):
        with open(find_file(filename), 'r') as f:
            data = json.load(f)
        self.texts_train, self.texts_test, self.labels_train, self.labels_test = train_test_split(
            data['texts'], data['labels'], test_size=0.5, random_state=1
        )
        self.full_cache = {}
        self.score_cache = {}

    def eval(self, embedder, name):
        results = evaluate_ner_embedder(
            embedder, self.texts_train, self.labels_train, self.texts_test, self.labels_test
        )
        max_score = max(results.values())
        self.full_cache[name] = results
        self.score_cache[name] = max_score
        return max_score, results


class FactRuTask(NERTask):
    def __init__(self, filename='data/factru.json'):
        super(FactRuTask, self).__init__(filename=filename)


class RudrTask(NERTask):
    def __init__(self, filename='data/rudrec.json'):
        super(RudrTask, self).__init__(filename=filename)


class SpeedTask(EncoderTask):
    def __init__(self, filename='data/sentences_sample.txt'):
        with open(find_file(filename), 'r') as f:
            self.sentences = [line.strip() for line in f.readlines()]
        self.full_cache = {}
        self.score_cache = {}

    def eval(self, embedder, name):
        t = time.time()
        for text in self.sentences:
            _ = embedder(text)
        ms_per_text = (time.time() - t) / len(self.sentences) * 1000
        results = {'ms_per_text': ms_per_text}
        max_score = ms_per_text
        self.full_cache[name] = results
        self.score_cache[name] = max_score
        return max_score, results


SENTENCE_TASKS = [
    STSBTask,
    ParaphraserTask,
    XnliTask,
    SentimentTask,
    ToxicityTask,
    InappropriatenessTask,
    IntentsTask,
    IntentsXTask,
]

WORD_TASKS = [
    FactRuTask,
    RudrTask,
]
