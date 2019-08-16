"""
Following the tutorial at http://www.realworldnlpbook.com/blog/training-sentiment-analyzer-using-allennlp.html
For the purpose of developing interpretable ELMo embeddings
"""

from typing import Dict

import torch
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader
from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.elmo import ELMoTokenCharactersIndexer
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.trainer import Trainer
from torch.optim import Adam


# Model in AllenNLP represents a model that is trained.
class LstmClassifier(Model):
    def __init__(self, word_embeddings: TextFieldEmbedder, encoder: Seq2VecEncoder, vocab: Vocabulary) -> None:
        super().__init__(vocab)
        # We need the embeddings to convert word IDs to their vector representations
        self.word_embeddings = word_embeddings
        self.encoder = encoder

        # After converting a sequence of vectors to a single vector, we feed it into
        # a fully-connected linear layer to reduce the dimension to the total number of labels.
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()

        # We use the cross-entropy loss because this is a classification task.
        # Note that PyTorch's CrossEntropyLoss combines softmax and log likelihood loss,
        # which makes it unnecessary to add a separate softmax layer.
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor = None) -> torch.Tensor:
        # In deep NLP, when sequences of tensors in different lengths are batched together,
        # shorter sequences get padded with zeros to make them of equal length.
        # Masking is the process to ignore extra zeros added by padding
        mask = get_text_field_mask(tokens)

        # Forward pass
        embeddings = self.word_embeddings(tokens)
        encoder_out = self.encoder(embeddings, mask)
        logits = self.hidden2tag(encoder_out)

        # In AllenNLP, the output of forward() is a dictionary.
        # Your output dictionary must contain a "loss" key for your model to be trained.
        output = {"logits": logits}
        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = self.loss_function(logits, label)

        return output


if __name__ == "__main__":
    elmo_token_indexer = ELMoTokenCharactersIndexer()
    reader = StanfordSentimentTreeBankDatasetReader(token_indexers={'tokens': elmo_token_indexer})
    train_dataset = reader.read('data/stanfordSentimentTreebank/trees/train.txt')
    dev_dataset = reader.read('data/stanfordSentimentTreebank/trees/dev.txt')

    # You can optionally specify the minimum count of tokens/labels.
    vocab = Vocabulary.from_instances(train_dataset + dev_dataset, min_count={'tokens': 3})

    HIDDEN_DIM = 128
    EPOCHS = 100
    BATCH_SIZE = 32

    options_file = 'data/elmo/elmo_2x1024_128_2048cnn_1xhighway_options.json'
    weight_file = 'data/elmo/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'
    elmo_embedder = ElmoTokenEmbedder(options_file, weight_file)
    word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})
    elmo_embedding_dim = 256

    lstm = PytorchSeq2VecWrapper(torch.nn.LSTM(elmo_embedding_dim, HIDDEN_DIM, batch_first=True))
    model = LstmClassifier(word_embeddings, lstm, vocab)
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    iterator = BucketIterator(batch_size=BATCH_SIZE, sorting_keys=[("tokens", "num_tokens")])
    iterator.index_with(vocab)

    trainer = Trainer(model=model, optimizer=optimizer, iterator=iterator, train_dataset=train_dataset,
                      validation_dataset=dev_dataset, patience=10, num_epochs=EPOCHS)
    trainer.train()

    with open("models/elmo_LSTM_sentiment_classifier.th", 'wb') as f:
        torch.save(model.state_dict(), f)
    vocab.save_to_files("models/elmo_LSTM_sentiment_classifier_vocabulary")
