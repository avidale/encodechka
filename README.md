# encodechka
## encodechka-eval

Этот репозиторий - развитие подхода к оценке моделей из поста
[Маленький и быстрый BERT для русского языка](https://habr.com/ru/post/562064), 
эволюционировавшего в [Рейтинг русскоязычных энкодеров предложений](https://habr.com/ru/post/669674/).
Идея в том, чтобы понять, как хорошо разные модели превращают короткие тексты
в осмысленные векторы.

Похожие проекты:
* [RussianSuperGLUE](https://russiansuperglue.com/): фокус на дообучаемых моделях
* [MOROCCO](https://github.com/RussianNLP/MOROCCO/): RussianSuperGLUE + оценка производительности, трудновоспроизводим
* [RuSentEval](https://github.com/RussianNLP/RuSentEval): более академические/лингвистические задачи
* Статья от Вышки [Popov et al, 2019](https://arxiv.org/abs/1910.13291): первая научная статья на эту тему, но маловато моделей и задач
* [SentEvalRu](https://github.com/comptechml/SentEvalRu) и [deepPavlovEval](https://github.com/deepmipt/deepPavlovEval): два хороших, но давно не обновлявшихся бенчмарка. 

Пример запуска метрик – в блокноте [evaluation example](https://github.com/avidale/encodechka/blob/master/evaluation%20example.ipynb). 

Блокнот для воспроизведения лидерборда: https://colab.research.google.com/drive/1fu2i7A-Yr-85Ex_NvIyeCIO7lN2R7P-k?usp=sharing.

### Лидерборд

Ранжирование моделей в по среднему качеству и производительности. 
Подсвечены Парето-оптимальные модели по каждому из критериев. 

| model                                             | CPU      | GPU     | size       |   Mean S | Mean S+W   |
|:--------------------------------------------------|:---------|:--------|:-----------|---------:|:-----------|
| paraphrase-multilingual-mpnet-base-v2             | **10.9** | **9.8** | **1064.0** |    0.756 |            |
| MUSE-3                                            | 66.2     | 18.5    | **303.0**  |    0.736 |            |
| paraphrase-multilingual-MiniLM-L12-v2             | **9.9**  | **9.5** | 449.0      |    0.728 |            |
| sentence-transformers/LaBSE                       | 105.9    | **7.5** | 1750.0     |    0.726 | 0.657      |
| cointegrated/LaBSE-en-ru                          | 106.0    | 7.5     | 492.0      |    0.725 | 0.656      |
| distiluse-base-multilingual-cased-v1              | **6.4**  | **5.9** | 514.0      |    0.721 |            |
| laser                                             | 118.8    | 8.8     | **200.0**  |    0.699 |            |
| cointegrated/rubert-tiny2                         | **4.9**  | **2.7** | **112.0**  |    0.689 | 0.631      |
| sberbank-ai/sbert_large_mt_nlu_ru                 | 356.0    | 14.4    | 1590.0     |    0.687 | 0.613      |
| DeepPavlov/rubert-base-cased-sentence             | 100.6    | 7.6     | 678.0      |    0.656 | 0.594      |
| sberbank-ai/sbert_large_nlu_ru                    | 348.6    | 14.2    | 1590.0     |    0.654 | 0.599      |
| DeepPavlov/distilrubert-base-cased-conversational | 49.7     | 4.5     | 517.0      |    0.629 | 0.587      |
| DeepPavlov/distilrubert-tiny-cased-conversational | 16.8     | **2.1** | 409.0      |    0.62  | 0.575      |
| ft_geowac_full                                    | **0.6**  |         | 1910.0     |    0.617 | 0.55       |
| cointegrated/rut5-base-paraphraser                | 118.8    | 9.1     | 932.0      |    0.617 | 0.579      |
| cointegrated/rubert-tiny                          | 6.1      | 2.8     | **45.0**   |    0.614 | 0.549      |
| ft_geowac_21mb                                    | 1.4      |         | **21.0**   |    0.597 | 0.531      |
| DeepPavlov/rubert-base-cased-conversational       | 98.8     | 7.6     | 681.0      |    0.591 | 0.557      |
| cointegrated/rut5-base-multitask                  | 117.8    | 9.4     | 932.0      |    0.575 | 0.548      |
| sberbank-ai/ruRoberta-large                       | 354.9    | 14.1    | 1320.0     |    0.568 | 0.528      |
| bert-base-multilingual-cased                      | 124.6    | 7.8     | 681.0      |    0.561 | 0.521      |
| hashing_1000_char                                 | **0.5**  |         | **1.0**    |    0.557 | 0.464      |
| cointegrated/rut5-small                           | 31.8     | 7.1     | 247.0      |    0.54  | 0.514      |
| hashing_300_char                                  | 0.5      |         | 1.0        |    0.528 | 0.433      |
| hashing_1000                                      | **0.2**  |         | 1.0        |    0.513 | 0.416      |
| sberbank-ai/ruT5-large                            | 339.9    | 14.4    | 2750.0     |    0.508 | 0.497      |
| hashing_300                                       | 0.2      |         | 1.0        |    0.491 | 0.397      |
| sberbank-ai/ruT5-base                             | 94.3     | 8.1     | 850.0      |    0.442 | 0.44       |
| cointegrated/rut5-base                            | 119.6    | 9.6     | 932.0      |    0.439 | 0.439      |

Ранжирование моделей по задачам.
Подсвечены наилучшие модели по каждой из задач. 

| model                                             | STS      | PI       | NLI      | SA       | TI       | IA      | IC       | ICX      | NE1      | NE2      |
|:--------------------------------------------------|:---------|:---------|:---------|:---------|:---------|:--------|:---------|:---------|:---------|:---------|
| paraphrase-multilingual-mpnet-base-v2             | **0.85** | 0.66     | **0.54** | 0.79     | 0.95     | 0.78    | 0.75     | 0.73     |          |          |
| MUSE-3                                            | 0.81     | 0.61     | 0.42     | 0.77     | 0.96     | 0.79    | **0.77** | **0.75** |          |          |
| paraphrase-multilingual-MiniLM-L12-v2             | 0.84     | 0.62     | 0.5      | 0.76     | 0.92     | 0.74    | 0.73     | 0.71     |          |          |
| sentence-transformers/LaBSE                       | 0.77     | 0.64     | 0.43     | 0.76     | 0.94     | 0.77    | 0.75     | 0.74     | 0.35     | 0.41     |
| cointegrated/LaBSE-en-ru                          | 0.77     | 0.64     | 0.43     | 0.76     | 0.94     | 0.77    | 0.75     | 0.74     | 0.34     | 0.41     |
| distiluse-base-multilingual-cased-v1              | 0.8      | 0.6      | 0.43     | 0.75     | 0.94     | 0.76    | 0.76     | 0.74     |          |          |
| laser                                             | 0.75     | 0.6      | 0.41     | 0.73     | 0.96     | 0.72    | 0.72     | 0.7      |          |          |
| cointegrated/rubert-tiny2                         | 0.75     | 0.65     | 0.42     | 0.73     | 0.93     | 0.75    | 0.69     | 0.59     | 0.4      | 0.4      |
| sberbank-ai/sbert_large_mt_nlu_ru                 | 0.77     | 0.64     | 0.4      | **0.79** | 0.98     | **0.8** | 0.7      | 0.42     | 0.3      | 0.34     |
| DeepPavlov/rubert-base-cased-sentence             | 0.73     | **0.66** | 0.49     | 0.75     | 0.89     | 0.75    | 0.61     | 0.36     | 0.36     | 0.34     |
| sberbank-ai/sbert_large_nlu_ru                    | 0.65     | 0.61     | 0.38     | 0.78     | 0.97     | 0.79    | 0.68     | 0.37     | 0.36     | 0.4      |
| DeepPavlov/distilrubert-base-cased-conversational | 0.57     | 0.52     | 0.36     | 0.73     | **0.98** | 0.78    | 0.67     | 0.42     | 0.4      | 0.43     |
| DeepPavlov/distilrubert-tiny-cased-conversational | 0.59     | 0.52     | 0.37     | 0.71     | 0.98     | 0.78    | 0.66     | 0.36     | 0.35     | 0.44     |
| ft_geowac_full                                    | 0.69     | 0.53     | 0.37     | 0.72     | 0.97     | 0.76    | 0.66     | 0.26     | 0.22     | 0.34     |
| cointegrated/rut5-base-paraphraser                | 0.64     | 0.53     | 0.36     | 0.69     | 0.91     | 0.69    | 0.61     | 0.5      | 0.45     | 0.41     |
| cointegrated/rubert-tiny                          | 0.65     | 0.51     | 0.4      | 0.68     | 0.86     | 0.68    | 0.58     | 0.54     | 0.23     | 0.34     |
| ft_geowac_21mb                                    | 0.68     | 0.52     | 0.36     | 0.72     | 0.96     | 0.74    | 0.65     | 0.15     | 0.21     | 0.32     |
| DeepPavlov/rubert-base-cased-conversational       | 0.54     | 0.53     | 0.34     | 0.72     | 0.97     | 0.76    | 0.62     | 0.26     | 0.4      | 0.43     |
| cointegrated/rut5-base-multitask                  | 0.62     | 0.5      | 0.36     | 0.66     | 0.88     | 0.69    | 0.57     | 0.32     | 0.47     | 0.41     |
| sberbank-ai/ruRoberta-large                       | 0.38     | 0.58     | 0.33     | 0.7      | 0.98     | 0.77    | 0.56     | 0.24     | 0.29     | **0.45** |
| bert-base-multilingual-cased                      | 0.62     | 0.51     | 0.36     | 0.66     | 0.85     | 0.69    | 0.56     | 0.23     | 0.35     | 0.37     |
| hashing_1000_char                                 | 0.7      | 0.53     | 0.4      | 0.7      | 0.84     | 0.59    | 0.63     | 0.05     | 0.05     | 0.14     |
| cointegrated/rut5-small                           | 0.59     | 0.52     | 0.34     | 0.65     | 0.86     | 0.67    | 0.53     | 0.15     | 0.44     | 0.38     |
| hashing_300_char                                  | 0.69     | 0.51     | 0.39     | 0.67     | 0.75     | 0.57    | 0.61     | 0.04     | 0.03     | 0.08     |
| hashing_1000                                      | 0.63     | 0.49     | 0.39     | 0.66     | 0.77     | 0.55    | 0.57     | 0.05     | 0.02     | 0.04     |
| sberbank-ai/ruT5-large                            | 0.4      | 0.34     | 0.35     | 0.67     | 0.94     | 0.73    | 0.47     | 0.16     | 0.46     | 0.44     |
| hashing_300                                       | 0.61     | 0.48     | 0.4      | 0.64     | 0.71     | 0.54    | 0.5      | 0.05     | 0.02     | 0.02     |
| sberbank-ai/ruT5-base                             | 0.28     | 0.23     | 0.35     | 0.62     | 0.88     | 0.66    | 0.37     | 0.14     | 0.45     | 0.41     |
| cointegrated/rut5-base                            | 0.37     | 0.21     | 0.34     | 0.61     | 0.83     | 0.68    | 0.35     | 0.13     | **0.48** | 0.39     |
