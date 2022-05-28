# -*- coding: utf-8 -*-
from packages.ckqa.machine import Machine
from sentence_transformers import SentenceTransformer, util, models
from sklearn.decomposition import PCA
from nltk.stem import WordNetLemmatizer
import numpy as np
import spacy
import torch


class SentParser:
    """parse KEY WORD from the sentence"""

    def __init__(self, user_dict=None, name='en_core_web_sm'):
        self.user_dict = user_dict
        self._model_spacy = spacy.load(name)

        if user_dict:
            if 'zh' in name:
                self._model_spacy.tokenizer.pkuseg_update_user_dict(user_dict)

            self.machine = Machine()
            self.machine.add_keywords_from_list([_.lower() for _ in user_dict])
        else:
            self.machine = None

    def parse(self, text):

        ents = [(token.text, token.idx, token.idx + len(token.text))
                for token in self._model_spacy(text) if token.pos_ == 'NOUN']

        if self.machine:
            ents += self.machine[text.lower()]

        return [e for e, _, _ in ents]


class SentSimi:

    def __init__(self, device=None):
        self._model_sent = SentenceTransformer('distiluse-base-multilingual-cased-v1', device=device).eval()

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

    @torch.no_grad()
    def lookup(self, q, ks, k=5, batch_size=128):
        query = self._encode(q, batch_size=batch_size)
        key = self._encode(ks, batch_size=batch_size)

        confi = util.pytorch_cos_sim(query, key)

        n = confi.size(-1)
        _, idx = torch.topk(confi, min(k, n))

        return [ks[i] for i in idx.tolist()[0]]


class SentMaker:
    template = {'Antonym': '{s} and {o} are opposite',
                'AtLocation': '{s} is located at {o}',
                'CapableOf': '{s} is capable of {o}',
                'Capital': '{o} is the captial of {s}',
                'Causes': '{s} causes {o}',
                'CausesDesire': '{s} makes someone want {o}',
                'CreatedBy': '{s} is created by {o}',
                'DefinedAs': '{s} is defined as {o}',
                'DerivedFrom': '{s} is derived from {o}',
                'Desires': '{s} desires {o}',
                'DistinctFrom': '{s} is distinct from {o}',
                'EtymologicallyDerivedFrom': '{s} is derived from {o}',
                'EtymologicallyRelatedTo': '{s} is related to {o}',
                'Field': '{o} is the field of {s}',
                'FormOf': '{s} is an inflected form of {o}',
                'Genre': '{o} is the genre of {s}',
                'Genus': '{s} is the genus of {o}',
                'HasA': '{s} has {o}',
                'HasContext': '{s} is in the context of {o}',
                'HasFirstSubevent': '{s} begins with the event {o}',
                'HasLastSubevent': '{s} ends with the event {o}',
                'HasPrerequisite': 'to do {s}, one requires {o}',
                'HasProperty': '{s} can be characterized by having {o}',
                'HasSubEvent': '{s} includes the event {o}',
                'InfluencedBy': '{s} is influenced by {o}',
                'InstanceOf': '{s} is an instance of {o}',
                'IsA': '{s} is a {o}',
                'KnownFor': '{s} is known for {o}',
                'Language': '{s} is in language {o}',
                'Leader': '{o} is the leader of {s}',
                'LocatedNear': '{s} and {o} are found near each other',
                'MadeOf': '{s} is made of {o}',
                'MannerOf': '{s} is a specific way to do {o}',
                'MotivatedByGoal': '{s} is a step towards accomplishing the goal {o}',
                'Occupation': '{o} is the occupation of {b}',
                'PartOf': '{s} is a part of {o}',
                'Product': '{o} is the product of {s}',
                'ReceivesAction': '{o} can be done to {s}',
                'SimilarTo': '{s} is similar to {o}',
                'SymbolOf': '{s} symbolically represents {o}',
                'Synonym': '{s} and {o} have very similar meanings',
                'UsedFor': '{s} is used for {o}'}

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def lexicalize(self, triple):
        s, p, o = triple

        s = self.wnl.lemmatize(s)
        o = self.wnl.lemmatize(o)

        if p not in self.template:
            return ' '.join([s, p, o]).capitalize().replace('_', ' ')
        else:
            return self.template[p].format(s=s, o=o).capitalize().replace('_', ' ')

    template_zh = {'Antonym': '{s}和{o}是反义词',
                   'AtLocation': '{s}位于{o}',
                   'CapableOf': '{s}能够{o}',
                   'Capital': '{o}是{s}的首都',
                   'Causes': '{s}导致{o}',
                   'CausesDesire': '{s}让某人想要{o}',
                   'CreatedBy': '{s}是由{o}创造',
                   'DefinedAs': '{s}被定义为{o}',
                   'DerivedFrom': '{s}源自于{o}',
                   'Desires': '{s}想要{o}',
                   'DistinctFrom': '{s}区别于{o}',
                   'EtymologicallyDerivedFrom': '{s}源自于{o}',
                   'EtymologicallyRelatedTo': '{s}与{o}有关',
                   'Field': '{o}是{s}的领域',
                   'FormOf': '{s}是{o}的一种形式',
                   'Genre': '{o}是{s}一种类型',
                   'Genus': '{s}是{o}一种属',
                   'HasA': '{o}是{s}的一部分',
                   'HasContext': '{s}出现在{o}的上下文中',
                   'HasFirstSubevent': '{s}以{o}作为开始',
                   'HasLastSubevent': '{s}以{o}作为结束',
                   'HasPrerequisite': '为了达到{s},必须以{o}为前提',
                   'HasProperty': '{s}具有{o}的特点',
                   'HasSubEvent': '{s}中包含事件{o}',
                   'InfluencedBy': '{s}受{o}的影响',
                   'InstanceOf': '{s}是{o}的实例',
                   'IsA': '{s}是{o}',
                   'KnownFor': '{s}以{o}著称',
                   'Language': '{s}被语言{o}描述',
                   'Leader': '{o}是{s}的主导者',
                   'LocatedNear': '{s}和{o}出现在一起',
                   'MadeOf': '{s}由{o}组成',
                   'MannerOf': '{s}是一种达成{o}的方案',
                   'MotivatedByGoal': '{s}是实现{o}的一步',
                   'Occupation': '{o}的职业是{b}',
                   'PartOf': '{s}是{o}的一部分',
                   'Product': '{o}是{s}的产品',
                   'ReceivesAction': '{o}以便尝试着做{s}',
                   'SimilarTo': '{s}与{o}类似',
                   'SymbolOf': '{s}象征着{o}',
                   'Synonym': '{s}和{o}在语义上相似',
                   'UsedFor': '{s}被用于{o}'}

    def lexicalize_zh(self, triple):
        s, p, o = triple

        if p not in self.template:
            return ' '.join([s, p, o]).replace('_', ' ')
        else:
            return self.template_zh[p].format(s=s, o=o).replace('_', ' ')


def join_sents(sents, lang='en'):
    if lang == 'en':
        context = '. '.join(sents) + '.'
        context = context.replace('_', ' ')
    elif lang == 'zh':
        context = '。'.join(sents) + '。'
        context = context.replace('_', ' ')
    else:
        raise ValueError(lang)

    return context


if __name__ == '__main__':
    parser = SentParser(['eat', 'meat'])
    print(parser.parse('Lion eat meat.'))
