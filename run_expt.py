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
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from sentence_transformers.readers import *
import logging
from datetime import datetime
from collections import defaultdict

import aux.util as util

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--working_dir", default='output/')
parser.add_argument("--fw", action='store_true', help='freq weighting')
parser.add_argument("--alpha", default=2e-5, help='smoothing term')
parser.add_argument("--fw_finetune", action='store_true')
parser.add_argument("--fw_eval", action='store_true')
parser.add_argument("--fw_train", action='store_true')
parser.add_argument("--remove_pc_train", action='store_true')
parser.add_argument("--remove_pc_test", action='store_true')
parser.add_argument("--pc_sample_size", default=100000, help='sample size for pca')
parser.add_argument("--replicates", default=1, type=int, help='number of experimental replicates')
ARGS = parser.parse_args()



#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# Read the dataset
model_name = 'bert-base-uncased'
batch_size = 16
nli_reader = NLIDataReader('examples/datasets/AllNLI')
sts_reader = STSDataReader('examples/datasets/stsbenchmark')
train_num_labels = nli_reader.get_num_labels()
working_dir = ARGS.working_dir + '/' #'output/'
model_save_path = working_dir + '/training_nli_'+model_name+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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
      # print(a, value, N)
      d[key] = a / (a + value / N)

  return d

wid2weight = get_tok_weights(
  word_embedding_model,
  weightfile='aux/enwiki_vocab_min200.txt',
  a=float(ARGS.alpha))


# for replicate in range(ARGS.replicates):

word_weights = models.WordWeights(
  vocab=list(word_embedding_model.tokenizer.vocab),
  word_weights=wid2weight,
  unknown_word_weight=1.0,
  finetune=ARGS.fw_finetune,
  eval_only=ARGS.fw_eval,
  train_only=ARGS.fw_train)

if ARGS.fw:
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
# Load the stored model and evaluate on each test set
#
##############################################################################

model = SentenceTransformer(model_save_path)

test_sets = {
  'SICK-r': SentencesDataset(examples=STSDataReader(
      'examples/datasets/SICK',
      s1_col_idx=1,
      s2_col_idx=2,
      score_col_idx=4).get_examples("SICK.txt"), model=model),

  'STSb': SentencesDataset(
    examples=sts_reader.get_examples("sts-test.csv"), model=model),

  'STS12': SentencesDataset(
    examples=STSDataReader(
      'examples/datasets/sts12',
      s1_col_idx=1,
      s2_col_idx=2,
      score_col_idx=0).get_examples("test.csv"), model=model),
  'STS13': SentencesDataset(    
    examples=STSDataReader(
      'examples/datasets/sts13',
      s1_col_idx=1,
      s2_col_idx=2,
      score_col_idx=0).get_examples("test.csv"), model=model),
  'STS14': SentencesDataset(    
    examples=STSDataReader(
      'examples/datasets/sts14',
      s1_col_idx=1,
      s2_col_idx=2,
      score_col_idx=0).get_examples("test.csv"), model=model),

  'STS15': SentencesDataset(examples=STSDataReader(
      'examples/datasets/sts15',
      s1_col_idx=1,
      s2_col_idx=2,
      score_col_idx=0).get_examples("test.csv"), model=model),
  'STS16': SentencesDataset(examples=STSDataReader(
      'examples/datasets/sts16',
      s1_col_idx=1,
      s2_col_idx=2,
      score_col_idx=0).get_examples("test.csv"), model=model),

}

out = open(os.path.join(working_dir, 'output.csv'), 'a')

for name, test_data in test_sets.items():
  test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

  evaluator = EmbeddingSimilarityEvaluator(test_dataloader,
    main_similarity=SimilarityFunction.COSINE)

  plain_score = model.evaluate(evaluator)
  out.write('%d\t%s\tplain\t%.2f\n' % (replicate, name, plain_score))

  if ARGS.remove_pc_train:
    embs = util.embed_dataloader(train_dataloader, model,
      sample_size=int(ARGS.pc_sample_size), data_size=len(train_data))
    pc = util.compute_pc(embs)
    evaluator = EmbeddingSimilarityEvaluator(test_dataloader,
      removal_direction=pc,
      main_similarity=SimilarityFunction.COSINE)
    train_pc_score = model.evaluate(evaluator)
    out.write('%d\t%s\ttrainPC\t%.2f\n' % (replicate, name, train_pc_score))

  if ARGS.remove_pc_test:
    embs = util.embed_dataloader(test_dataloader, model,
      sample_size=int(ARGS.pc_sample_size), data_size=len(train_data))
    pc = util.compute_pc(embs)
    evaluator = EmbeddingSimilarityEvaluator(test_dataloader,
      removal_direction=pc,
      main_similarity=SimilarityFunction.COSINE)
    test_pc_score = model.evaluate(evaluator)
    out.write('%d\t%s\ttestPC\t%.2f\n' % (replicate, name, test_pc_score))

  # if os.path.exists(model_save_path) and replicate < ARGS.replicates - 1:
  #   os.rmdir(model_save_path)
