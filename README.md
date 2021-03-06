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
usage: main.py [-h] [--vectorizer {tfidf,emb,deep}] [--classifier {MLPClassifier,SVC,AdaBoostClassifier}] [--epochs EPOCHS] [--no_cache] [--predict]
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
  --no_cache            Ignore cached files.
  --predict             Predict using a cached model.
  --aggressive_clean    Aggressive clean up of data.
```

### Usage Examples

To train a model using FastText embeddings and assuming our data lies in a directory called `my_data`:

`python main.py --vectorizer=emb --aggressive_clean my_data`

To predict using a TF-IDF model over a file called `sentences.txt` you would run:

`python main.py --vectorizer=tfidf --aggressive_clean --predict sentences.txt`

After predicting, predictions will be written to a new file called `sentences.txt_preds.txt` in our example.

## Results

| Approach | Accuracy | F1 Score|
|----------|----------|---------|
| Naive baseline | 0.60 | 0.02|
| TF-IDF | 0.66 | 0.58|
| FastText Embeddings | **0.68** | **0.61** |
| DistilRoberta FT | 0.61 | 0.04 |

Where:

* The naive baseline is simply predicting no sarcasm for all tweets.
* FastText is using 30 epochs to train and 2-grams.
* DistilRoberta is fine tuned over 3 epochs on a GPU.


## Conclusion

It seems that when running on limited resources, the best approach is to use precomputed embeddings for the classification task.

The obvious caveat here is that the performance for the deep learning approach probably requires a lot of time and resources to fine tune properly.
The winner of this task as SemEval2018 got 0.73 accuracy for some context.

## Acknowledgements

Data used was provided by the SemEval2018 Shared Task (#3).

Specifically from [this](https://github.com/cbaziotis/ntua-slp-semeval2018) repository.
