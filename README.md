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

Блокнот для воспроизведения лидерборда: [v2021](https://colab.research.google.com/drive/1fu2i7A-Yr-85Ex_NvIyeCIO7lN2R7P-k?usp=sharing), 
[v2023](https://colab.research.google.com/drive/1t956aJsp5qPnst3379vI8NNRqiqJUFMn?usp=sharing).

Лидерборд на [HuggingFace Space](https://huggingface.co/spaces/Samoed/Encodechka).

### Лидерборд

Ранжирование моделей в по среднему качеству и производительности. 
Подсвечены Парето-оптимальные модели по каждому из критериев. 

| model                                                       | CPU       | GPU      | size          |   Mean S | Mean S+W   |   dim |
|:------------------------------------------------------------|:----------|:---------|:--------------|---------:|:-----------|------:|
| BAAI/bge-m3                                                 | 523.4     | 22.5     | **2166.0**    |    0.787 | 0.696      |  1024 |
| intfloat/multilingual-e5-large-instruct                     | 501.5     | 25.71    | **2136.0**    |    0.784 | 0.684      |  1024 |
| intfloat/multilingual-e5-large                              | **506.8** | **30.8** | **2135.9389** |    0.78  | 0.686      |  1024 |
| deepvk/USER-base                                            | 33.1      | **12.2** | 473.2402      |    0.772 | 0.688      |   768 |    
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | **20.5**  | **19.9** | **1081.8485** |    0.762 |            |   768 |
| intfloat/multilingual-e5-base                               | 130.61    | 14.39    | **1061.0**    |    0.761 | 0.669      |   768 |
| intfloat/multilingual-e5-small                              | 40.86     | 12.09    | **449.0**     |    0.742 | 0.645      |   384 |
| symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli             | **20.2**  | **16.5** | **1081.8474** |    0.739 |            |   768 |
| cointegrated/LaBSE-en-ru                                    | 133.4     | **15.3** | **489.6621**  |    0.739 | 0.668      |   768 |
| sentence-transformers/LaBSE                                 | 135.1     | **13.3** | 1796.5078     |    0.739 | 0.667      |   768 |
| MUSE-3                                                      | 200.1     | 30.7     | **303.0**     |    0.736 |            |   512 |
| text-embedding-ada-002                                      | ?         |          |              |    0.734 |            |  1536 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | **18.2**  | 14.9     | 479.2547      |    0.734 |            |   384 |
| sentence-transformers/distiluse-base-multilingual-cased-v1  | **11.8**  | **8.0**  | 517.7452      |    0.722 |            |   512 |
| SONAR                                                       | ?         | ?        | 3060.0        |    0.721 |            |  1024 |
| facebook/nllb-200-distilled-600M                            | 252.3     | 15.9     | 1577.4828     |    0.709 | 0.64       |  1024 |
| sentence-transformers/distiluse-base-multilingual-cased-v2  | **11.2**  | 9.2      | 517.7453      |    0.708 |            |   512 |
| cointegrated/rubert-tiny2                                   | **6.2**   | **4.6**  | **111.3823**  |    0.704 | 0.638      |   312 |
| ai-forever/sbert_large_mt_nlu_ru                            | 504.5     | 29.7     | 1628.6539     |    0.703 | 0.626      |  1024 |
| laser                                                       | 192.5     | 13.5     | 200.0         |    0.699 |            |  1024 |
| laser2                                                      | 163.4     | 8.6      | 175.0         |    0.694 |            |  1024 |
| ai-forever/sbert_large_nlu_ru                               | 497.7     | 29.9     | 1628.6539     |    0.688 | 0.626      |  1024 |
| clips/mfaq                                                  | 18.1      | 18.2     | 1081.8576     |    0.687 |            |   768 |
| cointegrated/rut5-base-paraphraser                          | 137.0     | 15.6     | 412.0015      |    0.685 | 0.634      |   768 |
| DeepPavlov/rubert-base-cased-sentence                       | 128.4     | 13.2     | 678.5215      |    0.678 | 0.612      |   768 |
| DeepPavlov/distilrubert-base-cased-conversational           | 64.2      | 10.4     | 514.002       |    0.676 | 0.624      |   768 |
| DeepPavlov/distilrubert-tiny-cased-conversational           | 21.2      | **3.3**  | 405.8292      |    0.67  | 0.616      |   768 |
| cointegrated/rut5-base-multitask                            | 136.9     | 12.7     | 412.0015      |    0.668 | 0.623      |   768 |
| ai-forever/ruRoberta-large                                  | 512.3     | 25.5     | 1355.7162     |    0.666 | 0.609      |  1024 |
| DeepPavlov/rubert-base-cased-conversational                 | 127.5     | 16.3     | 678.5215      |    0.653 | 0.606      |   768 |
| deepvk/deberta-v1-base                                      | 128.6     | 19.0     | 473.2402      |    0.653 | 0.591      |   768 |
| cointegrated/rubert-tiny                                    | 7.5       | 5.9      | **44.97**     |    0.645 | 0.575      |   312 |
| ai-forever/FRED-T5-large                                    | 479.4     | 23.3     | 1372.9988     |    0.639 | 0.551      |  1024 |
| inkoziev/sbert_synonymy                                     | 6.9       | 4.2      | 111.3823      |    0.637 | 0.566      |   312 |
| numind/NuNER-multilingual-v0.1                              | 186.9     | 10       | 678.0         |    0.633 | 0.572      |   768 |
| cointegrated/rubert-tiny-toxicity                           | 10        | 5.5      | 47.2          |    0.621 | 0.553      |   312 |
| ft_geowac_full                                              | **0.3**   |          | 1910.0        |    0.617 | 0.55       |   300 |
| bert-base-multilingual-cased                                | 141.4     | 13.7     | 678.5215      |    0.614 | 0.565      |   768 |
| ai-forever/ruT5-large                                       | 489.6     | 20.2     | 1277.7571     |    0.61  | 0.578      |  1024 |
| cointegrated/rut5-small                                     | 37.6      | 8.6      | 111.3162      |    0.602 | 0.564      |   512 |
| ft_geowac_21mb                                              | 1.2       |          | **21.0**      |    0.597 | 0.531      |   300 |
| inkoziev/sbert_pq                                           | 7.4       | 4.2      | 111.3823      |    0.596 | 0.526      |   312 |
| ai-forever/ruT5-base                                        | 126.3     | 12.8     | 418.2325      |    0.571 | 0.544      |   768 |
| hashing_1000_char                                           | 0.5       |          | **1.0**       |    0.557 | 0.464      |  1000 |
| cointegrated/rut5-base                                      | 127.8     | 15.5     | 412.0014      |    0.554 | 0.53       |   768 |
| hashing_300_char                                            | 0.8       |          | 1.0           |    0.529 | 0.433      |   300 |
| hashing_1000                                                | **0.2**   |          | 1.0           |    0.513 | 0.416      |  1000 |
| hashing_300                                                 | 0.3       |          | 1.0           |    0.491 | 0.397      |   300 |

Ранжирование моделей по задачам.
Подсвечены наилучшие модели по каждой из задач. 

| model                                                       | STS      | PI       | NLI      | SA       | TI       | IA       | IC       | ICX      | NE1      | NE2      |
|:------------------------------------------------------------|:---------|:---------|:---------|:---------|:---------|:---------|:---------|:---------|:---------|:---------|
| BAAI/bge-m3                                                 | **0.86** | **0.75** | 0.51     | **0.82** | 0.97    | 0.79     | 0.81     | **0.78** | 0.24     | 0.42     |
| intfloat/multilingual-e5-large-instruct                     | 0.86     | 0.74     | 0.47     | 0.81     | 0.98    | 0.8      | **0.82** | 0.77     | 0.21     | 0.35     |
| intfloat/multilingual-e5-large                              | 0.86     | 0.73     | 0.47     | 0.81     | 0.98    | 0.8      | 0.82     | 0.77     | 0.24     | 0.37     |
| deepvk/USER-base                                            | 0.85     | 0.74     | 0.48     | 0.81     | 0.99     | **0.81**  | 0.8    | 0.7      | 0.29     | 0.41     |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | 0.85     | 0.66     | 0.54     | 0.79     | 0.95     | 0.78     | 0.79     | 0.74     |          |          |
| intfloat/multilingual-e5-base                               | 0.83     | 0.7      | 0.46     | 0.8      | 0.96    | 0.78     | 0.8      | 0.74     | 0.23     | 0.38     |
| intfloat/multilingual-e5-small                              | 0.82     | 0.71     | 0.46     | 0.76     | 0.96    | 0.76     | 0.78     | 0.69     | 0.23     | 0.27     |
| symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli             | 0.76     | 0.6      | **0.86** | 0.76     | 0.91     | 0.72     | 0.71     | 0.6      |          |          |
| cointegrated/LaBSE-en-ru                                    | 0.79     | 0.66     | 0.43     | 0.76     | 0.95     | 0.77     | 0.79     | 0.77     | 0.35     | 0.42     |
| sentence-transformers/LaBSE                                 | 0.79     | 0.66     | 0.43     | 0.76     | 0.95     | 0.77     | 0.79     | 0.76     | 0.35     | 0.41     |
| MUSE-3                                                      | 0.81     | 0.61     | 0.42     | 0.77     | 0.96     | 0.79     | 0.77     | 0.75     |          |          |
| text-embedding-ada-002                                      | 0.78     | 0.66     | 0.44     | 0.77     | 0.96     | 0.77     | 0.75     | 0.73     |          |          |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 0.84     | 0.62     | 0.5      | 0.76     | 0.92     | 0.74     | 0.77     | 0.72     |          |          |
| sentence-transformers/distiluse-base-multilingual-cased-v1  | 0.8      | 0.6      | 0.43     | 0.75     | 0.94     | 0.76     | 0.76     | 0.74     |          |          |
| SONAR                                                       | 0.71     | 0.58     | 0.41     | 0.77     | 0.98     | 0.79     | 0.78     | 0.74     |          |          |
| facebook/nllb-200-distilled-600M                            | 0.71     | 0.54     | 0.41     | 0.76     | 0.95     | 0.76     | 0.8      | 0.75     | 0.31     | 0.42     |
| sentence-transformers/distiluse-base-multilingual-cased-v2  | 0.79     | 0.55     | 0.42     | 0.75     | 0.91     | 0.75     | 0.76     | 0.73     |          |          |
| cointegrated/rubert-tiny2                                   | 0.75     | 0.65     | 0.42     | 0.74     | 0.94     | 0.75     | 0.76     | 0.64     | 0.36     | 0.39     |
| ai-forever/sbert_large_mt_nlu_ru                           | 0.78     | 0.65     | 0.4      | 0.8      | 0.98     | 0.8      | 0.76     | 0.45     | 0.3      | 0.34     |
| laser                                                       | 0.75     | 0.6      | 0.41     | 0.73     | 0.96     | 0.72     | 0.72     | 0.7      |          |          |
| laser2                                                      | 0.74     | 0.6      | 0.41     | 0.73     | 0.95     | 0.72     | 0.72     | 0.69     |          |          |
| ai-forever/sbert_large_nlu_ru                              | 0.68     | 0.62     | 0.39     | 0.78     | 0.98     | 0.8      | 0.78     | 0.48     | 0.36     | 0.4      |
| clips/mfaq                                                  | 0.63     | 0.59     | 0.35     | 0.79     | 0.95     | 0.74     | 0.76     | 0.69     |          |          |
| cointegrated/rut5-base-paraphraser                          | 0.65     | 0.53     | 0.4      | 0.78     | 0.95     | 0.75     | 0.75     | 0.67     | 0.45     | 0.41     |
| DeepPavlov/rubert-base-cased-sentence                       | 0.74     | 0.66     | 0.49     | 0.75     | 0.92     | 0.75     | 0.72     | 0.39     | 0.36     | 0.34     |
| DeepPavlov/distilrubert-base-cased-conversational           | 0.7      | 0.56     | 0.39     | 0.76     | 0.98     | 0.78     | 0.76     | 0.48     | 0.4      | 0.43     |
| DeepPavlov/distilrubert-tiny-cased-conversational           | 0.7      | 0.55     | 0.4      | 0.74     | 0.98     | 0.78     | 0.76     | 0.45     | 0.35     | 0.44     |
| cointegrated/rut5-base-multitask                            | 0.65     | 0.54     | 0.38     | 0.76     | 0.95     | 0.75     | 0.72     | 0.59     | 0.47     | 0.41     |
| ai-forever/ruRoberta-large                                 | 0.7      | 0.6      | 0.35     | 0.78     | 0.98     | 0.8      | 0.78     | 0.32     | 0.3      | **0.46** |
| DeepPavlov/rubert-base-cased-conversational                 | 0.68     | 0.52     | 0.38     | 0.73     | 0.98     | 0.78     | 0.75     | 0.42     | 0.41     | 0.43     |
| deepvk/deberta-v1-base                                      | 0.68     | 0.54     | 0.38     | 0.76     | 0.98     | 0.8      | 0.78     | 0.29     | 0.29     | 0.4      |
| cointegrated/rubert-tiny                                    | 0.66     | 0.53     | 0.4      | 0.71     | 0.89     | 0.68     | 0.7      | 0.58     | 0.24     | 0.34     |
| ai-forever/FRED-T5-large                                    | 0.62     | 0.44     | 0.37     | 0.78     | 0.98     | **0.81** | 0.67     | 0.45     | 0.25     | 0.15     |
| inkoziev/sbert_synonymy                                     | 0.69     | 0.49     | 0.41     | 0.71     | 0.91     | 0.72     | 0.69     | 0.47     | 0.32     | 0.24     |
| numind/NuNER-multilingual-v0.1                              | 0.67     | 0.53     | 0.4      | 0.71     | 0.89    | 0.72     | 0.7      | 0.46     | 0.32     | 0.34     |
| cointegrated/rubert-tiny-toxicity                           | 0.57     | 0.44     | 0.37     | 0.68     | **1.0** | 0.78     | 0.7      | 0.43     | 0.24     | 0.32     |
| ft_geowac_full                                              | 0.69     | 0.53     | 0.37     | 0.72     | 0.97     | 0.76     | 0.66     | 0.26     | 0.22     | 0.34     |
| bert-base-multilingual-cased                                | 0.66     | 0.53     | 0.37     | 0.7      | 0.89     | 0.7      | 0.69     | 0.38     | 0.36     | 0.38     |
| ai-forever/ruT5-large                                      | 0.51     | 0.39     | 0.35     | 0.77     | 0.97     | 0.79     | 0.72     | 0.38     | 0.46     | 0.44     |
| cointegrated/rut5-small                                     | 0.61     | 0.53     | 0.34     | 0.73     | 0.92     | 0.71     | 0.7      | 0.27     | 0.44     | 0.38     |
| ft_geowac_21mb                                              | 0.68     | 0.52     | 0.36     | 0.72     | 0.96     | 0.74     | 0.65     | 0.15     | 0.21     | 0.32     |
| inkoziev/sbert_pq                                           | 0.57     | 0.41     | 0.38     | 0.7      | 0.92     | 0.69     | 0.68     | 0.43     | 0.26     | 0.24     |
| ai-forever/ruT5-base                                       | 0.5      | 0.28     | 0.34     | 0.73     | 0.97     | 0.76     | 0.7      | 0.29     | 0.45     | 0.41     |
| hashing_1000_char                                           | 0.7      | 0.53     | 0.4      | 0.7      | 0.84     | 0.59     | 0.63     | 0.05     | 0.05     | 0.14     |
| cointegrated/rut5-base                                      | 0.44     | 0.28     | 0.33     | 0.74     | 0.92     | 0.75     | 0.58     | 0.39     | **0.48** | 0.39     |
| hashing_300_char                                            | 0.69     | 0.51     | 0.39     | 0.67     | 0.75     | 0.57     | 0.61     | 0.04     | 0.03     | 0.08     |
| hashing_1000                                                | 0.63     | 0.49     | 0.39     | 0.66     | 0.77     | 0.55     | 0.57     | 0.05     | 0.02     | 0.04     |
| hashing_300                                                 | 0.61     | 0.48     | 0.4      | 0.64     | 0.71     | 0.54     | 0.5      | 0.05     | 0.02     | 0.02     |

#### Задачи
- Semantic text similarity (**STS**) на основе переведённого датасета [STS-B](https://huggingface.co/datasets/stsb_multi_mt);
- Paraphrase identification (**PI**) на основе датасета paraphraser.ru;
- Natural language inference (**NLI**) на датасете [XNLI](https://github.com/facebookresearch/XNLI);
- Sentiment analysis (**SA**) на данных [SentiRuEval2016](http://www.dialog-21.ru/evaluation/2016/sentiment/).
- Toxicity identification (**TI**) на датасете токсичных комментариев из [OKMLCup](https://cups.mail.ru/ru/contests/okmlcup2020);
- Inappropriateness identification (**II**) на [датасете Сколтеха](https://github.com/skoltech-nlp/inappropriate-sensitive-topics);
- Intent classification (**IC**) и её кросс-язычная версия **ICX** на датасете [NLU-evaluation-data](https://github.com/xliuhw/NLU-Evaluation-Data), который я автоматически перевёл на русский. В IC классификатор обучается на русских данных, а в ICX – на английских, а тестируется в обоих случаях на русских.
- Распознавание именованных сущностей на датасетах [factRuEval-2016](https://github.com/dialogue-evaluation/factRuEval-2016) (**NE1**) и [RuDReC](https://github.com/cimm-kzn/RuDReC) (**NE2**). Эти две задачи требуют получать эмбеддинги отдельных токенов, а не целых предложений; поэтому там участвуют не все модели.

### Changelog
* Август 2023 - обновил рейтинг:
   * поправив ошибку в вычислении mean token embeddings
   * добавил несколько моделей, включая нового лидера - `intfloat/multilingual-e5-large`
   * по просьбам трудящихся, добавил `text-embedding-ada-002` (размер и производительность указаны от балды)
* Лето 2022 - опубликовал первый рейтинг
