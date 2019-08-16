import json
from tqdm import tqdm
from numpy import array, argmax
from allennlp.predictors.predictor import Predictor
from allennlp.data import Instance
from allennlp.data.fields import TextField
from allennlp.data.dataset_readers import DatasetReader
from typing import Iterator, List, Dict


# 1 - Load MultiNLI Data
DATA_PT = 'data/multinli_1.0/multinli_1.0_dev_matched.jsonl'

premises, hypotheses, labels = [], [], []
with open(DATA_PT, 'r') as f_in:
    for line in tqdm(f_in.readlines()):
        line_obj = json.loads(line)
        premise = line_obj['sentence1']
        hypothesis = line_obj['sentence2']
        label = line_obj['gold_label']
        premises.append(premise)
        hypotheses.append(hypothesis)
        labels.append(label)

X = list(zip(premises, hypotheses))
y_true = array(labels)


# 2 - Test Allen TE Model
MODEL_PT = "models/decomposable-attention-elmo-2018.02.19.tar.gz"
predictor = Predictor.from_path(MODEL_PT)
labels = ['entailment', 'contradiction', 'neutral']

predictions = []
for x in tqdm(X):
    probs = predictor.predict(premise=x[0], hypothesis=x[1])['label_probs']
    predictions.append(labels[argmax(probs)])

y_true = array(labels)

"""
After launching this code I get this ETA: 0 % | | 9 / 10000[00:11 < 4:22: 40, 1.58s / it]
Clearly running predictions in this fashion in unreasonable
Better getting familiar with AllenNLP, starting from here ... https://allennlp.org/tutorials
"""
