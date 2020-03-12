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
import numpy as np


import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--working_dir", default='output/')
parser.add_argument("--glove_vecs", action='store_true', help='use a glove vector model')
parser.add_argument("--alpha", default=2e-5, help='smoothing term')
parser.add_argument("--weighting_strategy", default=None, help='strategy for weighting vectors')
parser.add_argument("--normalize_weights", action='store_true', help='normalize weights')
parser.add_argument("--finetune_weights", action='store_true')
parser.add_argument("--eval_weights", action='store_true', help='only use weights during eval')
parser.add_argument("--train_weights", action='store_true', help='only use weights during training')
parser.add_argument("--remove_pc_train", action='store_true')
parser.add_argument("--remove_pc_test", action='store_true')
parser.add_argument("--pc_sample_size", default=100000, help='sample size for pca')
parser.add_argument("--npc", default=1, type=int, help='number of principal components to remove')
parser.add_argument("--replicates", default=1, type=int, help='number of experimental replicates')
parser.add_argument("--save_vecs", action='store_true', help='save vectors')

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
if not os.path.exists(working_dir):
  os.makedirs(working_dir)

model_save_path = [x for x in os.listdir(working_dir) if 'training_nli' in x]
if len(model_save_path) == 0:
  model_save_path = working_dir + '/training_nli_'+model_name+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
else:
  model_save_path = os.path.join(working_dir, model_save_path[0])



# Use BERT for mapping tokens to embeddings
if ARGS.glove_vecs:
  word_embedding_model = models.WordEmbeddings.from_text_file('glove.6B.300d.txt.gz')
else:
  word_embedding_model = models.BERT(model_name)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False,
                               pooling_mode_ParsePOOL=False)


def get_word_weights(model, word_freqs, doc_freqs, a=1e-3,
                     mode='tfidf', normalize=False):

  word_freq_lines = open(word_freqs).readlines()
  doc_freq_lines = open(word_freqs).readlines()

  num_words = int(word_freq_lines[0])
  num_docs = int(doc_freq_lines[0])

  word_freqs = dict([tuple(line.strip().split("\t")) for line in word_freq_lines[1:]])
  doc_freqs = dict([tuple(line.strip().split("\t")) for line in doc_freq_lines[1:]])
  s = 0
  out = {}
  for word, freq in word_freqs.items():
    freq = float(freq)
    s += freq

    tf = math.log(freq * 1.0)
    inv_freq = a / (a + (freq / num_words) )
    idf = math.log(num_docs / float(freq))

    if mode == 'tfidf':
      out[word] = tf * idf
    elif mode == 'idf':
      out[word] = idf
    elif mode == 'tf':
      out[word] = tf
    elif mode == 'itf':
      out[word] = inv_freq

    # Words in the vocab that are not in the doc_frequencies file get a frequency of 1
    if mode == 'tfidf':
      unknown_word_weight = math.log(1.0) * math.log(num_docs / 1)
    elif mode == 'idf':
      unknown_word_weight = math.log(num_docs / 1)
    elif mode == 'tf':
      unknown_word_weight = math.log(1.0)
    elif mode == 'itf':
      unknown_word_weight = 1.0

  if normalize:
    for k, v in out.items():
      out[k] = v * 1.0 / s
    unknown_word_weight = 1.0 / s

  return out, unknown_word_weight


# for replicate in range(ARGS.replicates):
replicate = 1


if ARGS.weighting_strategy is not None:
  wid2weight, unk_weight = get_word_weights(
    word_embedding_model,
    word_freqs='aux/wikipedia_word_frequencies.txt',
    doc_freqs='aux/wikipedia_doc_frequencies.txt',
    a=ARGS.alpha,
    mode=ARGS.weighting_strategy)
  word_weights = models.WordWeights(
    vocab=list(word_embedding_model.tokenizer.vocab),
    word_weights=wid2weight,
    unknown_word_weight=unk_weight,
    finetune=ARGS.finetune_weights,
    eval_only=ARGS.eval_weights,
    train_only=ARGS.train_weights)
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
# model.fit(train_objectives=[(train_dataloader, train_loss)],
#          evaluator=evaluator,
#          epochs=num_epochs,
#          evaluation_steps=1000,
#          warmup_steps=warmup_steps,
#          output_path=model_save_path
#          )


##############################################################################
#
# Load the stored model and evaluate on each test set
#
##############################################################################
print('loading from', model_save_path)
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

# hard code for now
replicate = 1


# SAVE VECS
# e.g. python run_expt.py --glove_vecs --working_dir --save_vecs
if ARGS.save_vecs:
  tokemb1, tokemb2, embs1, embs2, labels, in1, in2 = util.embed_dataloader(train_dataloader, model,
    sample_size=int(ARGS.pc_sample_size), data_size=len(train_data))
  with open(os.path.join(working_dir, 'train.embs'), 'wb') as f:
      np.savez(f, labels=labels, emb1=embs1, emb2=embs2,
        tokemb1=tokemb1, tokemb2=tokemb2)
  with open(os.path.join(working_dir, 'train.ids'), 'wb') as f:
      np.savez(f, in1=in1, in2=in2)

  for name, test_data in test_sets.items():
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    tokemb1, tokemb2, embs1, embs2, labels, in1, in2 = util.embed_dataloader(test_dataloader, model,
    sample_size=int(ARGS.pc_sample_size), data_size=len(test_dataloader))
    with open(os.path.join(working_dir, name + '.embs'), 'wb') as f:
        np.savez(f, labels=np.array(labels), emb1=embs1, emb2=embs2,
          tokemb1=tokemb1, tokemb2=tokemb2)
    with open(os.path.join(working_dir, name + '.ids'), 'wb') as f:
        np.savez(f, in1=in1, in2=in2)
  quit()

# precompute train pc
if ARGS.remove_pc_train:
  embs1, embs2, labels, _, _ = util.embed_dataloader(train_dataloader, model,
    sample_size=int(ARGS.pc_sample_size), data_size=len(train_data))
  embs = np.concatenate((embs1, embs2), axis=0)
  train_pc = util.compute_pc(embs, npc=int(ARGS.npc))


# do each benchmark
for name, test_data in test_sets.items():
  test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

  evaluator = EmbeddingSimilarityEvaluator(test_dataloader,
    main_similarity=SimilarityFunction.COSINE, name=name)

  plain_score = model.evaluate(evaluator)
  out.write('%s\t%d\t%s\tplain\t%.2f\n' % (working_dir, replicate, name, plain_score))

  if ARGS.remove_pc_train:
    evaluator = EmbeddingSimilarityEvaluator(test_dataloader,
      removal_direction=train_pc,
      main_similarity=SimilarityFunction.COSINE)
    train_pc_score = model.evaluate(evaluator)
    out.write('%s\t%d\t%s\ttrainPC\t%.2f\n' % (working_dir, replicate, name, train_pc_score))

  if ARGS.remove_pc_test:
    embs1, embs2, labels, _, _ = util.embed_dataloader(test_dataloader, model,
      sample_size=int(ARGS.pc_sample_size), data_size=len(train_data))
    embs = np.concatenate((embs1, embs2), axis=0)
    pc = util.compute_pc(embs, npc=int(ARGS.npc))
    evaluator = EmbeddingSimilarityEvaluator(test_dataloader,
      removal_direction=pc,
      main_similarity=SimilarityFunction.COSINE)
    test_pc_score = model.evaluate(evaluator)
    out.write('%s\t%d\t%s\ttestPC\t%.2f\n' % (working_dir, replicate, name, test_pc_score))

  # if os.path.exists(model_save_path) and replicate < ARGS.replicates - 1:
  #   os.rmdir(model_save_path)
