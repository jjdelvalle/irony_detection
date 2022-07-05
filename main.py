#!/usr/bin/env python3
# Import utils
# ML libraries will be imported dyncamically
import os.path
from os import mkdir
import argparse
import numpy as np
import pandas as pd

from typing import Union, Tuple
from pathlib import Path

# Classifier libraries and submodules
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.svm
import sklearn.ensemble
import sklearn.neural_network

# Import our metric-related stuff
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Set some nice CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--vectorizer", choices=['tfidf', 'emb', 'deep'],
                                    default='tfidf',
                                    type=str,
                                    help="Pick vectorize method.")
parser.add_argument("--classifier", choices=['MLPClassifier', 'SVC', 'AdaBoostClassifier'],
                                    default='SVC',
                                    type=str,
                                    help="Pick a classifying method.")
parser.add_argument('--epochs', type=int, default=30, help='Specify number of training epochs.')
parser.add_argument("--no_cache", action='store_true', default=False, help="Ignore cached files.")
parser.add_argument("--predict", action='store_true', default=False, help="Predict using a cached model.")
parser.add_argument("--aggressive_clean", action='store_true', default=False, help="Aggressive clean up of data.")
parser.add_argument("input_path", type=Path, help="Path to directory containing data. If `--predict` is set, this file contains phrases to be classified.")

def read_data(read_path: Path) -> pd.DataFrame:
    df = pd.read_csv(read_path/"SemEval_T3_taskA.txt", sep="\t")
    return df

def clean_data(df: pd.DataFrame, args) -> pd.DataFrame:
    df.drop(columns=["Tweet index"], inplace=True)
    df.columns = ["label", "text"]

    df["text"] = df["text"].str.lower()

    if args.aggressive_clean:
        # Twitter specific clean up
        df["text"] = df["text"].str.replace(r"#([a-z0-9])", r"_hashtag \1", regex=True)
        df["text"] = df["text"].str.replace(r"@[a-z0-9_]+", r"_mention", regex=True)
        df["text"] = df["text"].str.replace(r"https?://[^\s]+", "_url ", regex=True)

        # Separate out symbols since tokenizers might not handle them properly.
        # as per fastText documentation:
        df["text"] = df["text"].str.replace(r"([\.\\!\?,'/\(\)])", r" \1 ", regex=True)
    return df

def prepare_data(input_path: Path) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    train = read_data(input_path/"train")
    test = read_data(input_path/"gold")

    train = clean_data(train, args)
    test = clean_data(test, args)

    # Return our clean and split up dataset
    return train['text'],\
           train['label'],\
           test['text'],\
           test['label']

def prepare_fasttext(texts: pd.Series, labels: pd.Series, out_name: str, ignore_cache: bool) -> None:
    # Prepare data if not already cached
    # Maybe move this to a method instead of having it here
    if not os.path.isfile(out_name) or ignore_cache:
        fast_df = pd.DataFrame({'label': labels, 'text': texts})

        # We need to prepare labels for fasttext
        fast_df['label'] = '__label__' + fast_df['label'].astype('str')

        # Write to file. Tab separator might not be ideal but works fine
        # for the fasttext tokenizer
        fast_df.to_csv(out_name, sep='\t', header=False, index=False)

def tfidf_vectorize(X_train: pd.Series, X_test: pd.Series) -> Tuple[TfidfVectorizer,
                                                                         np.array,
                                                                         np.array]:
    """
    Instantiate and fit a tf-idf vectorizer
    Also cache in case we want to just predict in the future,
    alternatively load a cached vectorizer.

    Finally return the vectorized inputs.
    """
    vectorizer = TfidfVectorizer(max_features = 5000)
    train_vec = vectorizer.fit_transform(X_train.values)
    test_vec = vectorizer.transform(X_test.values)

    return train_vec, test_vec

def get_tfidf_classifier(clas_string: str) -> Union[sklearn.svm.SVC,
                                                    sklearn.ensemble.AdaBoostClassifier,
                                                    sklearn.neural_network.MLPClassifier]:
    """
    Instantiate classifier depending on the passed string
    """
    
    # Can't find a way to instantiate from string without specifying
    # the module. This is ugly for now.
    # Cleaner code would probably involve importing everything
    # and avoiding this `if`.
    if clas_string == 'SVC':
        classifier = getattr(sklearn.svm, clas_string)()
    elif clas_string == 'AdaBoostClassifier':
        classifier = getattr(sklearn.ensemble, clas_string)()
    else:
        classifier = getattr(sklearn.neural_network, clas_string)()

    return classifier

def get_fasttext_model(X_train, y_train, X_test, y_test, args):
    # Prepare data if necessary
    FAST_TRAIN_FILE = 'data/fast_train.txt'
    FAST_TEST_FILE = 'data/fast_test.txt'
    prepare_fasttext(X_train, y_train, FAST_TRAIN_FILE, args.no_cache)
    prepare_fasttext(X_test, y_test, FAST_TEST_FILE, args.no_cache)

    import fasttext
    # Train our classifier using bigrams and finetuned epochs
    # and learning rate values
    model = fasttext.train_supervised(input=FAST_TRAIN_FILE,
                                        lr=1,
                                        epoch=args.epochs,
                                        wordNgrams=2)

    return model

def get_dl_classifier(X_train, y_train, X_test, y_test, args):
    from transformers import RobertaTokenizer, RobertaForSequenceClassification
    from transformers import Trainer, TrainingArguments

    from datasets import load_metric
    from datasets import Dataset

    model = RobertaForSequenceClassification.from_pretrained("distilroberta-base", num_labels=2)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    training_args = TrainingArguments(output_dir="deep_trainer",
                                        num_train_epochs=args.epochs,
                                        warmup_ratio=0.05
                                        )

    train = Dataset.from_dict({'text': X_train, 'label': y_train})

    # Recommended approach from HF docs
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_train = train.map(tokenize_function, batched=True)

    trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train
            )
    trainer.train()

    from transformers import pipeline
    classifier = pipeline(task="text-classification", model=model, tokenizer=tokenizer, device=0)

    return classifier

def main(args):
    # Spliting into X & y
    X_train, y_train, X_test, y_test = prepare_data(args.input_path)

    # Building a TF IDF matrix out of the corpus of reviews
    if args.vectorizer == 'tfidf':
        train_vec, test_vec = tfidf_vectorize(X_train, X_test)
        classifier = get_tfidf_classifier(args.classifier)
        classifier.fit(train_vec, y_train)
    
        y_pred = classifier.predict(test_vec)

    elif args.vectorizer == 'emb':
        model = get_fasttext_model(X_train, y_train, X_test, y_test, args)

        # Couldn't find a better way to predict over an array
        # This is fast though
        y_pred = X_test.map(model.predict)

        # Post process model output since it looks like e.g.
        # `((__label__1,), 0.004324)` without any processing
        y_pred = y_pred.map(lambda x: x[0][0])
        y_pred = y_pred.str.replace('__label__', '')
        y_pred = y_pred.astype(int)
    
    elif args.vectorizer == 'deep':
        classifier = get_dl_classifier(X_train, y_train, X_test, y_test, args)

        y_pred = classifier(X_test.tolist(), batch_size=8)

        # Again, postprocess the output from the model as the output looks something like this
        # `{'label': 'LABEL_0', 'score': 0.9808016419410706}`
        y_pred = [int('1' in pred['label']) for pred in y_pred]

    # Classification metrics
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"F1 Score: {f1_score(y_test, y_pred)}")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
