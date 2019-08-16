import torch
from allennlp.data.dataset_readers import SnliReader
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers import CharacterTokenizer
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer
torch.manual_seed(1)

DATA_PT = 'data/multinli_1.0/multinli_1.0_dev_matched.jsonl'
reader = SnliReader(CharacterTokenizer(), {"elmo": ELMoTokenCharactersIndexer()})
data = reader.read(DATA_PT)

MODEL_PT = "models/decomposable-attention-elmo-2018.02.19.tar.gz"
predictor = Predictor.from_path(MODEL_PT)
result = predictor.predict_instance(data[0])
print(result)