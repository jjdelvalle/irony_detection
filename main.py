#!/usr/bin/env python3
# Import utils
# ML libraries will be imported dyncamically
import os.path
import argparse
import pandas as pd
import numpy as np

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
parser.add_argument("--predict", action='store_true', default=False, help="Predict instead of train.")
parser.add_argument("--aggressive_clean", action='store_true', default=False, help="Aggressive clean up of data.")
parser.add_argument("input_path", type=str, help="Path to directory containing data. If `--predict` is set, this file contains phrases to be classified.")

def read_data(read_path: str):
    df = pd.read_csv(f"{read_path}/SemEval_T3_taskA.txt", sep="\t")
    return df

def clean_data(df: pd.DataFrame, args):
    df.drop(columns=["Tweet index"], inplace=True)
    df.columns = ["label", "text"]

    df["text"] = df["text"].str.lower()

    if args.aggressive_clean:
        df["text"] = df["text"].str.replace(r"#[a-z0-9]", "_hashtag", regex=True)
        df["text"] = df["text"].str.replace(r"@[a-z0-9_]", "_mention", regex=True)
    return df

def prepare_data(input_path: str):
    train = read_data(f"{input_path}/train")
    test = read_data(f"{input_path}/gold")

    train = clean_data(train, args)
    test = clean_data(test, args)

    # Return our clean and split up dataset
    return train['text'],\
           train['label'],\
           test['text'],\
           test['label']

def prepare_fasttext(texts: pd.Series, labels: pd.Series, out_name: str):
    # Prepare data if not already cached
    # Maybe move this to a method instead of having it here
    if not os.path.isfile(out_name):
        fast_df = pd.DataFrame({'label': labels, 'text': texts})

        # We need to prepare labels for fasttext
        fast_df['label'] = '__label__' + fast_df['label'].astype('str')

        # Write to file. Tab separator might not be ideal but works fine
        # for the fasttext tokenizer
        fast_df.to_csv(out_name, sep='\t', header=False, index=False)

def main(args):
    # Spliting into X & y
    X_train, y_train, X_test, y_test = prepare_data(args.input_path)

    # Building a TF IDF matrix out of the corpus of reviews
    if args.vectorizer == 'tfidf':
        from sklearn.feature_extraction.text import TfidfVectorizer
        import sklearn.svm
        import sklearn.ensemble
        import sklearn.neural_network

        vectorizer = TfidfVectorizer(max_features = 5000)
        train_vec = vectorizer.fit_transform(X_train.values)
        test_vec = vectorizer.transform(X_test.values)

        # Can't find a way to instantiate from string without specifying
        # the module. This is ugly for now.
        if args.classifier == 'SVC':
            classifier = getattr(sklearn.svm, args.classifier)()
        elif args.classifier == 'AdaBoostClassifier':
            classifier = getattr(sklearn.ensemble, args.classifier)()
        elif args.classifier == 'MLPClassifier':
            classifier = getattr(sklearn.neural_network, args.classifier)()

        classifier.fit(train_vec, y_train)
    
        y_pred = classifier.predict(test_vec)

    elif args.vectorizer == 'emb':
        # Prepare data if necessary
        FAST_TRAIN_FILE = 'data/fast_train.txt'
        FAST_TEST_FILE = 'data/fast_test.txt'
        prepare_fasttext(X_train, y_train, FAST_TRAIN_FILE)
        prepare_fasttext(X_test, y_test, FAST_TEST_FILE)

        import fasttext
        # Train our classifier using bigrams and finetuned epochs
        # and learning rate values
        model = fasttext.train_supervised(input=FAST_TRAIN_FILE,
                                          lr=1,
                                          epoch=args.epochs,
                                          wordNgrams=2)

        y_pred = X_test.map(model.predict)

        # Post process model output since it looks like e.g.
        # `((__label__1,), 0.004324)` without any processing
        y_pred = y_pred.map(lambda x: x[0][0])
        y_pred = y_pred.str.replace('__label__', '')
        y_pred = y_pred.astype(int)
    
    elif args.vectorizer == 'deep':
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

        y_pred = classifier(X_test.tolist(), batch_size=8)

        # Again, postprocess the output from the model as the output looks something like this
        # `{'label': 'LABEL_0', 'score': 0.9808016419410706}`
        y_pred = [int('1' in pred['label']) for pred in y_pred]

    # Classification metrics
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"F1 Score: {f1_score(y_test, y_pred)}")

    # report = classification_report(y_test, y_pred)
    # print(report)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
