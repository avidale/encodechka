# encodechka
## encodechka-eval
Yet another framework for evaluation of Russian sentence encoders

Этот репозиторий - развитие подхода к оценке моделей из статьи
[Маленький и быстрый BERT для русского языка](https://habr.com/ru/post/562064).
Идея в том, чтобы понять, как хорошо разные модели превращают короткие тексты
в осмысленные векторы.

Похожие проекты:
* [RussianSuperGLUE](https://russiansuperglue.com/): фокус на дообучаемых моделях
* [MOROCCO](https://github.com/RussianNLP/MOROCCO/): RussianSuperGLUE + оценка производительности, трудновоспроизводим
* [RuSentEval](https://github.com/RussianNLP/RuSentEval): более академические/лингвистические задачи
* Статья от Вышки [Popov et al, 2019](https://arxiv.org/abs/1910.13291): первая научная статья на эту тему, но маловато моделей и задач
* [SentEvalRu](https://github.com/comptechml/SentEvalRu) и [deepPavlovEval](https://github.com/deepmipt/deepPavlovEval): два хороших, но давно не обновлявшихся бенчмарка. 

Пример запуска метрик - в блокноте [evaluation example](https://github.com/avidale/encodechka/blob/master/evaluation%20example.ipynb). 

Блокнот для воспроизведения лидерборда: https://colab.research.google.com/drive/1fu2i7A-Yr-85Ex_NvIyeCIO7lN2R7P-k?usp=sharing.

### Лидерборд
| model                                             |   STSBTask |   ParaphraserTask |   XnliTask |   SentimentTask |   ToxicityTask |   InappropriatenessTask |   IntentsTask |   IntentsXTask | FactRuTask   | RudrTask   |   cpu_speed | gpu_speed   |   disk_size |   mean_s | mean_sw   |
|:--------------------------------------------------|-----------:|------------------:|-----------:|----------------:|---------------:|------------------------:|--------------:|---------------:|:-------------|:-----------|------------:|:------------|------------:|---------:|:----------|
| MUSE-3                                            |      0.813 |             0.612 |      0.424 |           0.769 |          0.965 |                   0.786 |         0.768 |          0.752 |              |            |      66.17  | 18.544      |         303 |    0.736 |           |
| sentence-transformers/LaBSE                       |      0.773 |             0.644 |      0.431 |           0.76  |          0.944 |                   0.766 |         0.748 |          0.745 | 0.352        | 0.405      |     105.905 | 7.473       |        1750 |    0.726 | 0.657     |
| cointegrated/LaBSE-en-ru                          |      0.773 |             0.645 |      0.431 |           0.759 |          0.943 |                   0.766 |         0.746 |          0.741 | 0.343        | 0.412      |     105.961 | 7.521       |         492 |    0.725 | 0.656     |
| laser                                             |      0.753 |             0.601 |      0.407 |           0.731 |          0.959 |                   0.722 |         0.725 |          0.697 |              |            |     118.786 | 8.823       |         200 |    0.699 |           |
| cointegrated/rubert-tiny2                         |      0.75  |             0.651 |      0.417 |           0.732 |          0.929 |                   0.746 |         0.692 |          0.595 | 0.398        | 0.399      |       4.895 | 2.731       |         112 |    0.689 | 0.631     |
| sberbank-ai/sbert_large_mt_nlu_ru                 |      0.772 |             0.641 |      0.395 |           0.793 |          0.978 |                   0.796 |         0.696 |          0.423 | 0.299        | 0.336      |     356.033 | 14.429      |        1590 |    0.687 | 0.613     |
| DeepPavlov/rubert-base-cased-sentence             |      0.734 |             0.663 |      0.489 |           0.749 |          0.894 |                   0.746 |         0.605 |          0.364 | 0.359        | 0.337      |     100.637 | 7.647       |         678 |    0.656 | 0.594     |
| sberbank-ai/sbert_large_nlu_ru                    |      0.654 |             0.612 |      0.382 |           0.779 |          0.969 |                   0.791 |         0.679 |          0.366 | 0.363        | 0.399      |     348.613 | 14.185      |        1590 |    0.654 | 0.599     |
| DeepPavlov/distilrubert-base-cased-conversational |      0.57  |             0.521 |      0.36  |           0.731 |          0.978 |                   0.778 |         0.671 |          0.424 | 0.402        | 0.435      |      49.686 | 4.505       |         517 |    0.629 | 0.587     |
| DeepPavlov/distilrubert-tiny-cased-conversational |      0.59  |             0.515 |      0.366 |           0.71  |          0.976 |                   0.778 |         0.665 |          0.359 | 0.351        | 0.443      |      16.805 | 2.124       |         409 |    0.62  | 0.575     |
| ft_geowac_full                                    |      0.686 |             0.532 |      0.365 |           0.719 |          0.966 |                   0.756 |         0.659 |          0.255 | 0.223        | 0.339      |       0.571 |             |        1910 |    0.617 | 0.55      |
| cointegrated/rut5-base-paraphraser                |      0.643 |             0.53  |      0.358 |           0.686 |          0.914 |                   0.695 |         0.609 |          0.499 | 0.445        | 0.409      |     118.788 | 9.066       |         932 |    0.617 | 0.579     |
| cointegrated/rubert-tiny                          |      0.652 |             0.509 |      0.398 |           0.685 |          0.861 |                   0.683 |         0.585 |          0.539 | 0.23         | 0.345      |       6.083 | 2.753       |          45 |    0.614 | 0.549     |
| ft_geowac_21mb                                    |      0.68  |             0.524 |      0.359 |           0.722 |          0.956 |                   0.736 |         0.65  |          0.152 | 0.208        | 0.323      |       1.367 |             |          21 |    0.597 | 0.531     |
| DeepPavlov/rubert-base-cased-conversational       |      0.539 |             0.526 |      0.344 |           0.721 |          0.968 |                   0.76  |         0.615 |          0.259 | 0.403        | 0.434      |      98.832 | 7.574       |         681 |    0.591 | 0.557     |
| cointegrated/rut5-base-multitask                  |      0.621 |             0.5   |      0.356 |           0.665 |          0.878 |                   0.694 |         0.566 |          0.322 | 0.47         | 0.411      |     117.85  | 9.367       |         932 |    0.575 | 0.548     |
| sberbank-ai/ruRoberta-large                       |      0.384 |             0.583 |      0.326 |           0.702 |          0.977 |                   0.773 |         0.56  |          0.237 | 0.293        | 0.447      |     354.926 | 14.069      |        1320 |    0.568 | 0.528     |
| bert-base-multilingual-cased                      |      0.622 |             0.514 |      0.359 |           0.662 |          0.85  |                   0.686 |         0.564 |          0.233 | 0.354        | 0.37       |     124.635 | 7.842       |         681 |    0.561 | 0.521     |
| hashing_1000_char                                 |      0.701 |             0.531 |      0.398 |           0.701 |          0.844 |                   0.59  |         0.635 |          0.053 | 0.048        | 0.137      |       0.502 |             |           1 |    0.557 | 0.464     |
| cointegrated/rut5-small                           |      0.595 |             0.519 |      0.342 |           0.654 |          0.862 |                   0.67  |         0.531 |          0.15  | 0.44         | 0.383      |      31.843 | 7.115       |         247 |    0.54  | 0.514     |
| hashing_300_char                                  |      0.686 |             0.505 |      0.392 |           0.671 |          0.751 |                   0.57  |         0.61  |          0.043 | 0.026        | 0.08       |       0.478 |             |           1 |    0.528 | 0.433     |
| hashing_1000                                      |      0.627 |             0.488 |      0.389 |           0.665 |          0.772 |                   0.547 |         0.569 |          0.045 | 0.018        | 0.036      |       0.217 |             |           1 |    0.513 | 0.416     |
| sberbank-ai/ruT5-large                            |      0.399 |             0.341 |      0.35  |           0.674 |          0.938 |                   0.733 |         0.471 |          0.159 | 0.463        | 0.442      |     339.895 | 14.372      |        2750 |    0.508 | 0.497     |
| hashing_300                                       |      0.61  |             0.48  |      0.397 |           0.639 |          0.712 |                   0.536 |         0.505 |          0.05  | 0.017        | 0.024      |       0.231 |             |           1 |    0.491 | 0.397     |
| sberbank-ai/ruT5-base                             |      0.285 |             0.23  |      0.352 |           0.62  |          0.884 |                   0.661 |         0.37  |          0.137 | 0.448        | 0.411      |      94.328 | 8.12        |         850 |    0.442 | 0.44      |
| cointegrated/rut5-base                            |      0.366 |             0.213 |      0.339 |           0.605 |          0.826 |                   0.682 |         0.351 |          0.133 | 0.483        | 0.39       |     119.579 | 9.614       |         932 |    0.439 | 0.439     |
