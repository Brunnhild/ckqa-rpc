# -*- coding: utf-8 -*-
from sentence_transformers import util, models
from sklearn.decomposition import PCA
import numpy as np
import torch


class NodeSimi:

    def __init__(self, model):
        self._model_sent = model
        self.keys = None
        self.keys_embedding = None

    def _encode(self, sentences, batch_size=128):
        sentences = [s.replace('_', ' ') for s in sentences] if isinstance(sentences, list) else sentences.replace('_',
                                                                                                                   ' ')
        return self._model_sent.encode(sentences, batch_size=batch_size)

    @torch.no_grad()
    def add_compressed_layer(self, refs, dim, batch_size=128):
        emb = self._encode(refs, batch_size=batch_size)
        pca = PCA(n_components=dim)
        pca.fit(emb)
        pca_comp = np.asarray(pca.components_)
        dense = models.Dense(in_features=self._model_sent.get_sentence_embedding_dimension(),
                             out_features=dim, bias=False, activation_function=torch.nn.Identity())
        dense.linear.weight = torch.nn.Parameter(torch.tensor(pca_comp))

        self._model_sent.add_module('dense', dense)

    def set_keys(self, keys, batch_size=128):
        self.keys = keys
        self.keys_embedding = self._encode(keys, batch_size=batch_size)

    @torch.no_grad()
    def query(self, q, k=5, threshold=0.5, batch_size=128):
        if self.keys is None:
            raise ValueError('Keys should be set !')

        query = self._encode(q, batch_size=batch_size)

        confi = util.pytorch_cos_sim(query, self.keys_embedding)
        n = confi.size(-1)
        score, idx = torch.topk(confi, min(k, n))
        score, idx = score.tolist()[0], idx.tolist()[0]

        return [self.keys[i] for i, s in zip(idx, score) if s > threshold]
