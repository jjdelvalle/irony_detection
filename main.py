#!/usr/bin/env python3
# Import utils
# ML libraries will be imported dyncamically
import os.path
from os import mkdir

import pickle
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

def clean_data(df: pd.DataFrame, args, training: bool = True) -> pd.DataFrame:
    if training:
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

def prepare_data(input_path: Path, predict: bool) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    if predict:
        data = pd.read_csv(input_path, sep="\t", header=None)
        if len(data.columns) != 1:
            print("CSV file with just one column expected")
            return None
        data.columns = ["text"]
        data = clean_data(data, args, training=False)
        return None, None, data['text'], None
    else:    
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
    """
    Prepare input files for the fasttext model to train on.

    This is needed because fasttext expects its input in a very
    specific format.
    """
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
    vectorizer_file = f"models/tfidf_vectorizer.pkl"

    # Load if model exists
    if os.path.isfile(vectorizer_file) and not args.no_cache:
        with open(vectorizer_file, 'rb') as f:
            vectorizer = pickle.load(f)
        if X_train is not None:
            train_vec = vectorizer.transform(X_train.values)
        else:
            train_vec = None
    else:
        vectorizer = TfidfVectorizer(max_features = 5000)
        train_vec = vectorizer.fit_transform(X_train.values)

        # Cache the vectorizer
        with open(vectorizer_file, 'wb') as f:
            pickle.dump(vectorizer, f)

    test_vec = vectorizer.transform(X_test.values)

    return train_vec, test_vec

def get_tfidf_classifier(args: argparse.Namespace,
                         clas_string: str,
                         train_vec: np.array,
                         y_train: pd.Series) -> Union[sklearn.svm.SVC,
                                                      sklearn.ensemble.AdaBoostClassifier,
                                                      sklearn.neural_network.MLPClassifier]:
    """
    Instantiate classifier depending on the passed string
    """
    model_file = f"models/tfidf_{args.classifier}.pkl"

    # Load if model exists
    if os.path.isfile(model_file) and not args.no_cache:
        with open(model_file, 'rb') as f:
            classifier = pickle.load(f)
    else:
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

        classifier.fit(train_vec, y_train)
        
        # Cache the classifier
        with open(model_file, 'wb') as f:
            pickle.dump(classifier, f)

    return classifier

def get_fasttext_model(X_train: pd.Series,
                       y_train: pd.Series,
                       X_test: pd.Series,
                       y_test: pd.Series,
                       args: argparse.Namespace):
    """
    Prepare data, instatiate model and train it.

    If the prepared input files already exist, use them
    instead of creating them again.
    """
    # Prepare data if necessary
    FAST_TRAIN_FILE = 'data/fast_train.txt'
    FAST_TEST_FILE = 'data/fast_test.txt'
    prepare_fasttext(X_train, y_train, FAST_TRAIN_FILE, args.no_cache)
    prepare_fasttext(X_test, y_test, FAST_TEST_FILE, args.no_cache)

    import fasttext
    model_file = f"models/fasttext_model.bin"

    # Load if model exists
    if os.path.isfile(model_file) and not args.no_cache:
        model = fasttext.load_model(model_file)
    else:
        # Train our classifier using bigrams and finetuned epochs
        # and learning rate values
        model = fasttext.train_supervised(input=FAST_TRAIN_FILE,
                                          lr=1,
                                          epoch=args.epochs,
                                          wordNgrams=2)
                
        # Cache model
        model.save_model(model_file)

    return model

def get_dl_classifier(X_train: pd.Series,
                      y_train: pd.Series,
                      X_test: pd.Series,
                      y_test: pd.Series,
                      args: argparse.Namespace):
    """
    Download pretrained model from HF, instantiate it, and train it.

    Training will be done using some settings found by empirically testing
    different hyper-parameters, the rest are set by the CLI options.
    """
    from transformers import RobertaTokenizer, RobertaForSequenceClassification
    from transformers import Trainer, TrainingArguments

    from datasets import load_metric
    from datasets import Dataset

    model_output = "models/deep"

    # Load if model exists
    if os.path.isdir(model_output) and not args.no_cache:
        model = RobertaForSequenceClassification.from_pretrained(model_output)
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    else:
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

        mkdir(model_output)
        model = RobertaForSequenceClassification.save_pretrained(model_output)

    from transformers import pipeline
    classifier = pipeline(task="text-classification", model=model, tokenizer=tokenizer, device=0)

    return classifier

def main(args):
    # Create directories if they do not exist
    if not os.path.isdir("models"):
        mkdir("models")

    # Spliting into train/test and X and y
    X_train, y_train, X_test, y_test = prepare_data(args.input_path, args.predict)

    # Building a TF IDF matrix out of the corpus of reviews
    if args.vectorizer == 'tfidf':
        train_vec, test_vec = tfidf_vectorize(X_train, X_test)
        classifier = get_tfidf_classifier(args, args.classifier, train_vec, y_train)
    
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

    if args.predict:
        # Save predictions to file
        np.savetxt(str(args.input_path) + "_preds.txt", y_pred.astype(int), fmt='%1.0f')
    else:
        # Classification metrics
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"F1 Score: {f1_score(y_test, y_pred)}")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
