# Language Technologies in Practice Final Project - Sarcasm Detection
  
As the purpose of this course is to realize how different things can be in a real life setting, this project attempts to tackle a problem that will be useful in a real life context.
Namely, sarcasm detection. There are various use cases for this, among them more accurate evaluation of feedback/reviews.

Additionally, not only modern approaches are tried to tackle this problem.
The approaches used for tackling this task are:
  
* Naive TF/IDF vectorization techniques
* Word embedding vectorization techniques
* Deep transfer learning techniques
 
All of these provide different advantages and an *overall* winner will be picked as a go to approach depending on the available resources.

## Usage

```
usage: main.py [-h] [--vectorizer {tfidf,emb,deep}] [--classifier {MLPClassifier,SVC,AdaBoostClassifier}] [--epochs EPOCHS] [--predict]
               [--aggressive_clean]
               input_path

positional arguments:
  input_path            Path to directory containing data. If `--predict` is set, this file contains phrases to be classified.

optional arguments:
  -h, --help            show this help message and exit
  --vectorizer {tfidf,emb,deep}
                        Pick vectorize method.
  --classifier {MLPClassifier,SVC,AdaBoostClassifier}
                        Pick a classifying method.
  --epochs EPOCHS       Specify number of training epochs.
  --predict             Predict using a cached model.
  --aggressive_clean    Aggressive clean up of data.
```

## Results

| Approach | Accuracy | F1 Score|
|----------|----------|---------|
| Naive baseline | 0.60 | 0.02|
| TF-IDF | 0.66 | | 0.58|
| FastText Embeddings | 0.68 | 0.61 |
| DistilRoberta FT | 0.61 | 0.04 |

Where:

* The naive baseline is simply predicting no sarcasm for all tweets.
* FastText is using 30 epochs to train and 2-grams.
* DistilRoberta is fine tuned over 3 epochs on a GPU.

## Acknowledgements

Data used was provided by the SemEval2018 Shared Task (#3).

Specifically from [this](https://github.com/cbaziotis/ntua-slp-semeval2018) repository.
