import razdel
import scipy.stats
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from tqdm.auto import tqdm


def eval_pairs_multiotput(embedder, train_data, text1='text_1', text2='text_2', y_col='class'):
    train_embs1 = [embedder(x) for x in tqdm(train_data[text1])]
    train_embs2 = [embedder(x) for x in tqdm(train_data[text2])]
    results = {}
    for k in train_embs1[0].keys():
        train_emb1 = np.stack([row[k] for row in train_embs1])
        train_emb2 = np.stack([row[k] for row in train_embs2])
        scores = np.sum(train_emb1 * train_emb2, axis=1)  # in the previous version, there was a plus sign
        results[k] = scipy.stats.spearmanr(train_data[y_col], scores).correlation
    return results


def eval_pairs_clf_multiotput(embedder, train_data, text1='text_1', text2='text_2', y_col='class'):
    train_embs1 = [embedder(x) for x in tqdm(train_data[text1])]
    train_embs2 = [embedder(x) for x in tqdm(train_data[text2])]
    results = {}
    cv = KFold(n_splits=3, shuffle=True, random_state=1)
    for k in train_embs1[0].keys():
        train_emb1 = np.stack([row[k] for row in train_embs1])
        train_emb2 = np.stack([row[k] for row in train_embs2])
        scores = np.sum(train_emb1 * train_emb2, axis=1)
        preds = cross_val_predict(LogisticRegression(max_iter=10_000), scores[:, np.newaxis], train_data[y_col], cv=cv)
        results[k] = np.mean(train_data[y_col] == preds)
    return results


def eval_accuracy_multiotput(embedder, train_data, test_data, x_col='text', y_col='answer', x_col_test=None):
    train_embs = [embedder(x) for x in tqdm(train_data[x_col])]
    test_embs = [embedder(x) for x in tqdm(test_data[x_col_test or x_col])]
    results = {}
    for k in train_embs[0].keys():
        train_emb = np.stack([row[k] for row in train_embs])
        test_emb = np.stack([row[k] for row in test_embs])
        models = {
            'lr': LogisticRegression(max_iter=10_000),
            'knn': KNeighborsClassifier(n_neighbors=3, weights='distance'),
        }
        for clf_name, clf in models.items():
            clf.fit(train_emb, train_data[y_col])
            results[f'{k}__{clf_name}'] = np.mean(clf.predict(test_emb) == test_data[y_col])
    return results


def eval_auc_multiotput(embedder, train_data, test_data, x_col='text', y_col='answer'):
    train_embs = [embedder(x) for x in tqdm(train_data[x_col])]
    test_embs = [embedder(x) for x in tqdm(test_data[x_col])]
    results = {}
    for k in train_embs[0].keys():
        train_emb = np.stack([row[k] for row in train_embs])
        test_emb = np.stack([row[k] for row in test_embs])
        lr = LogisticRegression(max_iter=10_000).fit(train_emb, train_data[y_col])
        results[k] = roc_auc_score(test_data[y_col], lr.predict_proba(test_emb)[:, 1])
    return results


def make_word_labels(text, entities, bi=True):
    words = []
    char2word = np.zeros(len(text), dtype=int) - 1
    for i, w in enumerate(razdel.tokenize(text)):
        words.append(w.text)
        char2word[w.start:w.stop] = i

    labels = ['O'] * len(words)
    for (entity_type, e_start, e_end) in entities:
        b = False
        for char in range(e_start, e_end):
            token_id = char2word[char]
            if not token_id or token_id == -1:
                continue
            if labels[token_id] != 'O':
                continue
            if not b:
                labels[token_id] = 'B-' + entity_type
                b = True
            else:
                labels[token_id] = ('I-' if bi else 'B-') + entity_type
    return words, labels


def get_ner_X_y(texts, raw_labels, words_embedder):
    vecs = []
    vec_labels = []

    for text, text_labels in tqdm(zip(texts, raw_labels), total=len(texts)):
        words, word_labels = make_word_labels(text, text_labels)
        # this part should be abstracted away
        id2vecs = words_embedder(words) # get_word_vectors_with_bert(words, model, tokenizer)
        for i, label in enumerate(word_labels):
            vec_labels.append(label)
            vecs.append(id2vecs[i])
    return np.stack(vecs), np.array(vec_labels)


def evaluate_ner_embedder(embedder, texts_train, labels_train, texts_test, labels_test):
    X_train, y_train = get_ner_X_y(texts_train, labels_train, words_embedder=embedder)
    X_test, y_test = get_ner_X_y(texts_test, labels_test, words_embedder=embedder)

    clf = SGDClassifier(loss='log', shuffle=True, verbose=0, random_state=1, early_stopping=False, tol=1e-4)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    results = {
        'macro_f1': f1_score(y_test, preds, average='macro', labels=sorted(set(y_test).difference({'O'}))),
    }
    return results
