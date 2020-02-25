"""
The system trains BERT on the SNLI + MultiNLI (AllNLI) dataset
with softmax loss function. At every 1000 training steps, the model is evaluated on the
STS benchmark dataset
"""
import os
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import *
import logging
from datetime import datetime
from collections import defaultdict


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--working_dir", default='output/')
parser.add_argument("--freq_weighting", action='store_true')
parser.add_argument("--alpha", default=1e-3, help='smoothing term')
parser.add_argument("--finetune_freq_weights", action='store_true')
parser.add_argument("--freq_weighting_eval_only", action='store_true')
parser.add_argument("--freq_weighting_train_only", action='store_true')
ARGS = parser.parse_args()



#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# Read the dataset
model_name = 'bert-base-uncased'
batch_size = 2
nli_reader = NLIDataReader('examples/datasets/AllNLI')
sts_reader = STSDataReader('examples/datasets/stsbenchmark')
train_num_labels = nli_reader.get_num_labels()
model_save_path = 'output/training_nli_'+model_name+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
working_dir = 'output/'

if not os.path.exists(working_dir):
  os.makedirs(working_dir)


# Use BERT for mapping tokens to embeddings
word_embedding_model = models.BERT(model_name)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False,
                               pooling_mode_ParsePOOL=False)


def get_tok_weights(model, weightfile, a=1e-3):
  vocab = list(model.tokenizer.vocab)

  d = defaultdict(float)
  N = 0.0
  for l in open(weightfile):
      word, freq = l.strip().split()
      freq = float(freq)
      for tok_idx in model.tokenize(word):
          d[vocab[tok_idx]] += freq
          N += freq

  for key, value in d.items():
      d[key] = a / (a + value/N)

  return d

wid2weight = get_tok_weights(
  word_embedding_model,
  weightfile='aux/enwiki_vocab_min200.txt',
  a=ARGS.alpha)
word_weights = models.WordWeights(
  vocab=list(word_embedding_model.tokenizer.vocab),
  word_weights=wid2weight,
  unknown_word_weight=1.0,
  finetune=ARGS.finetune_freq_weights,
  eval_only=ARGS.freq_weighting_eval_only,
  train_only=ARGS.freq_weighting_train_only)

if ARGS.freq_weighting:
  model = SentenceTransformer(modules=[word_embedding_model, word_weights, pooling_model])
else:
  model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Convert the dataset to a DataLoader ready for training
logging.info("Read AllNLI train dataset")
if os.path.exists(working_dir + 'train_data.toks'):
  train_data = SentencesDataset.load(working_dir + 'train_data')
else:
  train_data = SentencesDataset(nli_reader.get_examples('train.gz'), model=model)
  train_data.save(working_dir + 'train_data')
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=train_num_labels)



logging.info("Read STSbenchmark dev dataset")

if os.path.exists(working_dir + 'dev_data.toks'):
  dev_data = SentencesDataset.load(working_dir + 'dev_data')
else:
  dev_data = SentencesDataset(examples=sts_reader.get_examples('sts-dev.csv'), model=model)
  dev_data.save(working_dir + 'dev_data')
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)
evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)

# Configure the training
num_epochs = 1

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))



# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path
          )



##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

model = SentenceTransformer(model_save_path)
test_data = SentencesDataset(examples=sts_reader.get_examples("sts-test.csv"), model=model)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
evaluator = EmbeddingSimilarityEvaluator(test_dataloader)

model.evaluate(evaluator, output_path=working_dir + 'final_output')
