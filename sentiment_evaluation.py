import numpy as np
import torch
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.elmo import ELMoTokenCharactersIndexer
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.predictors import TextClassifierPredictor

from sentiment_simple_lstm import LstmClassifier

if __name__ == "__main__":
    simple_lstm = False
    elmo_lstm = True

    # Simple LSTM
    if simple_lstm:
        EMBEDDING_DIM = 128
        HIDDEN_DIM = 128
        reader = StanfordSentimentTreeBankDatasetReader()
        train_dataset = reader.read('data/stanfordSentimentTreebank/trees/train.txt')
        dev_dataset = reader.read('data/stanfordSentimentTreebank/trees/dev.txt')
        test_dataset = reader.read('data/stanfordSentimentTreebank/trees/test.txt')
        vocab = Vocabulary.from_instances(train_dataset + dev_dataset, min_count={'tokens': 3})
        token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=EMBEDDING_DIM)
        word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
        lstm = PytorchSeq2VecWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
        model = LstmClassifier(word_embeddings, lstm, vocab)
        with open("models/simple_LSTM_sentiment_classifier.th", 'rb') as f:
            model.load_state_dict(torch.load(f))
        predictor = TextClassifierPredictor(model, dataset_reader=reader)
        test_results = predictor.predict_batch_instance(test_dataset)

    # ELMo LSTM
    if elmo_lstm:
        elmo_embedding_dim = 256
        HIDDEN_DIM = 128
        elmo_token_indexer = ELMoTokenCharactersIndexer()
        reader = StanfordSentimentTreeBankDatasetReader(token_indexers={'tokens': elmo_token_indexer})
        train_dataset = reader.read('data/stanfordSentimentTreebank/trees/train.txt')
        dev_dataset = reader.read('data/stanfordSentimentTreebank/trees/dev.txt')
        test_dataset = reader.read('data/stanfordSentimentTreebank/trees/test.txt')
        vocab = Vocabulary.from_instances(train_dataset + dev_dataset, min_count={'tokens': 3})
        options_file = 'data/elmo/elmo_2x1024_128_2048cnn_1xhighway_options.json'
        weight_file = 'data/elmo/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'
        elmo_embedder = ElmoTokenEmbedder(options_file, weight_file)
        word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})
        lstm = PytorchSeq2VecWrapper(torch.nn.LSTM(elmo_embedding_dim, HIDDEN_DIM, batch_first=True))
        model = LstmClassifier(word_embeddings, lstm, vocab)
        with open("models/elmo_LSTM_sentiment_classifier.th", 'rb') as f:
            model.load_state_dict(torch.load(f))
        predictor = TextClassifierPredictor(model, dataset_reader=reader)
        test_results = predictor.predict_batch_instance(test_dataset)

    # Computing Accuracy
    true_labels = [int(test_dataset[x]['label'].label) for x in range(len(test_dataset))]
    pred_labels = [np.argmax(test_results[x]['logits']) for x in range(len(test_dataset))]
    accuracy = sum([true_labels[y] == pred_labels[y] for y in range(len(test_dataset))]) / len(test_dataset)

    print('Accuracy', round(100 * accuracy, 2), '%')