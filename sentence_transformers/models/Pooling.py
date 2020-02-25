import torch
from torch import Tensor
from torch import nn
from typing import Union, Tuple, List, Iterable, Dict
import os
import json


class Pooling(nn.Module):
    """Performs pooling (max or mean) on the token embeddings.

    Using pooling, it generates from a variable sized sentence a fixed sized sentence embedding. This layer also allows to use the CLS token if it is returned by the underlying word embedding model.
    You can concatenate multiple poolings together.
    """
    def __init__(self,
                 word_embedding_dimension: int,
                 pooling_mode_cls_token: bool = False,
                 pooling_mode_max_tokens: bool = False,
                 pooling_mode_mean_tokens: bool = True,
                 pooling_mode_mean_sqrt_len_tokens: bool = False,
                 pooling_mode_ParsePOOL: bool = False,
                 # pooling_mode_mean_freq: bool = False
                 ):
        super(Pooling, self).__init__()

        self.config_keys = ['word_embedding_dimension',  'pooling_mode_cls_token', 'pooling_mode_mean_tokens', 'pooling_mode_max_tokens', 'pooling_mode_mean_sqrt_len_tokens']
        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_mode_cls_token = pooling_mode_cls_token
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self.pooling_mode_max_tokens = pooling_mode_max_tokens
        self.pooling_mode_mean_sqrt_len_tokens = pooling_mode_mean_sqrt_len_tokens
        self.pooling_mode_ParsePOOL = pooling_mode_ParsePOOL
        # self.pooling_mode_mean_freq = pooling_mode_mean_freq

        pooling_mode_multiplier = sum([pooling_mode_cls_token, pooling_mode_max_tokens, pooling_mode_mean_tokens, pooling_mode_mean_sqrt_len_tokens, pooling_mode_ParsePOOL])
        self.pooling_output_dimension = (pooling_mode_multiplier * word_embedding_dimension)

    def forward(self, features_aux):
        # features_aux is (Dict[str, Tensor], dict)
        if features_aux[0] is not None:
            features, aux = features_aux
        else:
            features, aux = features_aux, None

        # trees = aux['trees']
        # print(aux['word2weight'])

        token_embeddings = features['token_embeddings']
        cls_token = features['cls_token_embeddings']
        input_mask = features['input_mask']

        ## Pooling strategy
        output_vectors = []
        # if self.pooling_mode_mean_freq:
        #     word2weight = aux['word2weight']
        #     word2weight = torch.tensor(list(word2weight))

        #     input_ids = features['input_ids']

        #     print(input_ids)
        #     token_weights = torch.gather(word2weight, 1, input_ids)
        #     print(token_weights)
        #     print(token_embeddings.shape)
        #     print(input_mask_expanded.shape)
        #     print(features['input_ids'])
        #     quit()
        #     input_mask_expanded = input_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        #     token_embeddings = token_embeddings * input_mask_expanded
        #     sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)


        if self.pooling_mode_ParsePOOL:
            input_mask_expanded = input_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            output_vectors.append(self.ParsePOOL(token_embeddings))

        if self.pooling_mode_cls_token:
            output_vectors.append(cls_token)
        if self.pooling_mode_max_tokens:
            input_mask_expanded = input_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            input_mask_expanded = input_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            #If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if 'token_weights_sum' in features:
                sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / sum_mask)
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

        output_vector = torch.cat(output_vectors, 1)
        features.update({'sentence_embedding': output_vector})
        return features

    def get_sentence_embedding_dimension(self):
        return self.pooling_output_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        return Pooling(**config)

    def ParsePOOL(self, token_embeddings):
        parse = {'S': ['Feds', {'VP': ['raised', {'NP': ['the', 'interest', 'rates']}]}]}
        token_embeddings = token_embeddings[:, :5, :]

        print('$' * 100)
        # [B T H]
        print(token_embeddings.shape)
        print('$' * 100)
        quit()






