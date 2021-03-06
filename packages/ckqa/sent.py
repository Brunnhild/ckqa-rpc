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

        tokens = self._model_spacy(text)
        ents = [(token.text, token.idx, token.idx + len(token.text))
                for token in tokens if token.pos_ == 'NOUN' or token.pos_ == 'VERB']

        if self.machine:
            ents += self.machine[text.lower()]

        return [e for e, _, _ in ents]


class SentSimi:

    def __init__(self, device=None):
        self._model_sent = SentenceTransformer('/home/ubuntu/.cache/torch/sentence_transformers/sentence-transformers_distiluse-base-multilingual-cased-v1', device=device).eval()

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
    def lookup(self, q, ks, k=5, batch_size=128, threshold=0.3):
        if len(ks) == 0:
            return [], []

        query = self._encode(q, batch_size=batch_size)
        key = self._encode(ks, batch_size=batch_size)

        confi = util.pytorch_cos_sim(query, key)

        n = confi.size(-1)
        _, idx = torch.topk(confi, min(k, n))
        idx = idx.tolist()[0]
        confi = confi.tolist()
        idx = [i for i in idx if confi[0][i] > threshold]

        return [ks[i] for i in idx], [confi[0][i] for i in idx]


class SentMaker:
    template = {'Antonym':                   '{s} and {o} are opposite',
             'AtLocation':                '{s} is located at {o}',
             'CapableOf':                 '{s} is capable of {o}',
             'Capital':                   '{o} is the captial of {s}',
             'Causes':                    '{s} causes {o}',
             'CausesDesire':              '{s} makes someone want {o}',
             'CreatedBy':                 '{s} is created by {o}',
             'DefinedAs':                 '{s} is defined as {o}',
             'DerivedFrom':               '{s} is derived from {o}',
             'Desires':                   '{s} desires {o}',
             'DistinctFrom':              '{s} is distinct from {o}',
             'EtymologicallyDerivedFrom': '{s} is derived from {o}',
             'EtymologicallyRelatedTo':   '{s} is related to {o}',
             'Field':                     '{o} is the field of {s}',
             'FormOf':                    '{s} is an inflected form of {o}',
             'Genre':                     '{o} is the genre of {s}',
             'Genus':                     '{s} is the genus of {o}',
             'HasA':                      '{s} has {o}',
             'HasContext':                '{s} is in the context of {o}',
             'HasFirstSubevent':          '{s} begins with the event {o}',
             'HasLastSubevent':           '{s} ends with the event {o}',
             'HasPrerequisite':           'to do {s}, one requires {o}',
             'HasProperty':               '{s} can be characterized by having {o}',
             'HasSubEvent':               '{s} includes the event {o}',
             'InfluencedBy':              '{s} is influenced by {o}',
             'InstanceOf':                '{s} is an instance of {o}',
             'IsA':                       '{s} is a {o}',
             'KnownFor':                  '{s} is known for {o}',
             'Language':                  '{s} is in language {o}',
             'Leader':                    '{o} is the leader of {s}',
             'LocatedNear':               '{s} and {o} are found near each other',
             'MadeOf':                    '{s} is made of {o}',
             'MannerOf':                  '{s} is a specific way to do {o}',
             'MotivatedByGoal':           '{s} is a step towards accomplishing the goal {o}',
             'Occupation':                '{o} is the occupation of {o}',
             'PartOf':                    '{s} is a part of {o}',
             'Product':                   '{o} is the product of {s}',
             'ReceivesAction':            '{o} can be done to {s}',
             'RelatedTo':                 '{s} is related to {o}',
             'SimilarTo':                 '{s} is similar to {o}',
             'SymbolOf':                  '{s} symbolically represents {o}',
             'Synonym':                   '{s} and {o} have very similar meanings',
             'UsedFor':                   '{s} is used for {o}'}

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

    template_zh = {'Antonym':                   '{s}???{o}????????????',
             'AtLocation':                '{s}??????{o}',
             'CapableOf':                 '{s}??????{o}',
             'Capital':                   '{o}???{s}?????????',
             'Causes':                    '{s}??????{o}',
             'CausesDesire':              '{s}???????????????{o}',
             'CreatedBy':                 '{s}??????{o}??????',
             'DefinedAs':                 '{s}????????????{o}',
             'DerivedFrom':               '{s}?????????{o}',
             'Desires':                   '{s}??????{o}',
             'DistinctFrom':              '{s}?????????{o}',
             'EtymologicallyDerivedFrom': '{s}?????????{o}',
             'EtymologicallyRelatedTo':   '{s}???{o}??????',
             'Field':                     '{o}???{s}?????????',
             'FormOf':                    '{s}???{o}???????????????',
             'Genre':                     '{o}???{s}???????????????',
             'Genus':                     '{s}???{o}????????????????????????',
             'HasA':                      '{o}???{s}????????????',
             'HasContext':                '{s}?????????{o}???????????????',
             'HasFirstSubevent':          '{s}???{o}????????????',
             'HasLastSubevent':           '{s}???{o}????????????',
             'HasPrerequisite':           '????????????{s},?????????{o}?????????',
             'HasProperty':               '{s}??????{o}?????????',
             'HasSubEvent':               '{s}???????????????{o}',
             'InfluencedBy':              '{s}???{o}?????????',
             'InstanceOf':                '{s}???{o}?????????',
             'IsA':                       '{s}???{o}',
             'KnownFor':                  '{s}???{o}??????',
             'Language':                  '{s}?????????{o}??????',
             'Leader':                    '{o}???{s}????????????',
             'LocatedNear':               '{s}???{o}????????????',
             'MadeOf':                    '{s}???{o}??????',
             'MannerOf':                  '{s}???????????????{o}?????????',
             'MotivatedByGoal':           '{s}?????????{o}?????????',
             'Occupation':                '{o}????????????{o}',
             'PartOf':                    '{s}???{o}????????????',
             'Product':                   '{o}???{s}?????????',
             'ReceivesAction':            '{o}??????????????????{s}',
             'RelatedTo':                 '{s}???{o}??????',
             'SimilarTo':                 '{s}???{o}??????',
             'SymbolOf':                  '{s}?????????{o}',
             'Synonym':                   '{s}???{o}??????????????????',
             'UsedFor':                   '{s}?????????{o}'}

    def lexicalize_zh(self, triple):
        s, p, o = triple

        if p not in self.template:
            return ' '.join([s, p, o]).replace('_', ' ')
        else:
            return self.template_zh[p].format(s=s, o=o).replace('_', ' ')


def join_sents(sents, lang='en'):
    if len(sents) == 0:
        return ''
    if lang == 'en':
        context = '. '.join(sents) + '.'
        context = context.replace('_', ' ')
    elif lang == 'zh':
        context = '???'.join(sents) + '???'
        context = context.replace('_', ' ')
    else:
        raise ValueError(lang)

    return context


if __name__ == '__main__':
    parser = SentParser(['eat', 'meat'])
    print(parser.parse('Lion eat meat.'))
