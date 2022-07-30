# -*- coding: utf-8 -*-
import json
import math

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from tqdm import tqdm
from transformers import AutoTokenizer, BatchEncoding
from transformers import BertModel, BertConfig

relation_description = {'Antonym':                   'X和Y是反义词',
                        'AtLocation':                'X位于Y',
                        'CapableOf':                 'X有能力做Y',
                        'Capital':                   'Y是X的首都',
                        'Causes':                    'X导致Y的发生',
                        'CausesDesire':              'X使得某人想要Y',
                        'CreatedBy':                 'X是由Y创造',
                        'DefinedAs':                 'X被定义为Y',
                        'DerivedFrom':               'X源自于Y',
                        'Desires':                   'X想要Y',
                        'DistinctFrom':              'X区别于Y',
                        'EtymologicallyDerivedFrom': 'X在词源上源自于Y',
                        'EtymologicallyRelatedTo':   'X与Y在词源上有关',
                        'Field':                     'Y是X的领域',
                        'FormOf':                    'X是Y的一种形式',
                        'Genre':                     'X属于流派Y',
                        'Genus':                     'X是Y的一种动植物的属',
                        'HasA':                      'X拥有Y',
                        'HasContext':                'X出现在Y的上下文中',
                        'HasFirstSubevent':          'X以Y作为开始',
                        'HasLastSubevent':           'X以Y作为结束',
                        'HasPrerequisite':           '为了达到X,必须以Y为前提',
                        'HasProperty':               'X具有Y的特点',
                        'HasSubEvent':               'X中包含事件Y',
                        'InfluencedBy':              'X受Y的影响',
                        'InstanceOf':                'X是Y的实例',
                        'IsA':                       'X是Y的一种类型',
                        'KnownFor':                  'X以Y著称',
                        'Language':                  'X被语言Y描述',
                        'Leader':                    'Y是X的领导者',
                        'LocatedNear':               'X和Y一起出现',
                        'MadeOf':                    'X由Y组成',
                        'MannerOf':                  'X是一种达成Y的方案',
                        'MotivatedByGoal':           'X是实现Y的一步',
                        'Occupation':                'X的职业是Y',
                        'PartOf':                    'X是Y的一部分',
                        'Product':                   'Y是X的产品',
                        'ReceivesAction':            'X可以被Y',
                        'RelatedTo':                 'X与Y相关',
                        'SimilarTo':                 'X与Y类似',
                        'SymbolOf':                  'X象征着Y',
                        'Synonym':                   'X和Y在意思上相近',
                        'UsedFor':                   'X被用于Y'}


class CSKTokenizer:

    def __init__(self, pretrained_model_name_or_path):
        self.ptm_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

    def __call__(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)

    def tokenize(self, h_text, r_text, t_text, max_length=16, return_tensors='pt'):
        hr_input = self.ptm_tokenizer(r_text, h_text,  # r_text first to avoid h_text is too long
                                      padding='max_length',
                                      truncation=True,
                                      max_length=max_length,
                                      return_tensors=return_tensors)

        t_input = self.ptm_tokenizer(t_text,
                                     padding='max_length',
                                     truncation=True,
                                     max_length=max_length,
                                     return_tensors=return_tensors)

        inputs = BatchEncoding({

            't_input_ids':       t_input.input_ids,
            't_attention_mask':  t_input.attention_mask,
            't_token_type_ids':  t_input.token_type_ids,

            'hr_input_ids':      hr_input.input_ids,
            'hr_attention_mask': hr_input.attention_mask,
            'hr_token_type_ids': hr_input.token_type_ids})

        return inputs


class ModelModule(LightningModule):

    def __init__(self,
                 **module_args):
        super().__init__()
        self.save_hyperparameters(module_args)

        self.encoder = BertModel(BertConfig.from_pretrained('hfl/chinese-roberta-wwm-ext'))
        self.tokenizer = CSKTokenizer('hfl/chinese-roberta-wwm-ext')

        self.dropout = torch.nn.Dropout(self.hparams.dropout)

    def phi(self, hr, t):

        hr = F.normalize(hr, dim=1)
        t = F.normalize(t, dim=1)
        return hr @ t.t()

    def forward(self, input_ids, attention_mask, token_type_ids):

        output = self.encoder(input_ids, attention_mask, token_type_ids)
        if self.hparams.pool == 'cls':

            # simply taking the hidden state corresponding to the first token.
            vector = output.last_hidden_state[:, 0, :]

        elif self.hparams.pool == 'mean':
            last_hidden_state = output.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(last_hidden_state).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-8)
            vector = sum_embeddings / sum_mask
        else:
            raise ValueError(self.hparams.pool)

        vector = self.dropout(vector)

        return vector


class CompletionModel:
    def __init__(self, model_path):
        self.batch_size = 128
        self.candidate = json.load(open(f'{model_path}/release.json', 'r', encoding='utf-8'))
        self.model = ModelModule.load_from_checkpoint(f'{model_path}/release.ckpt')
        self.model.freeze()
        self.model = self.model.cuda()

    def predict(self, head='动物园', rel='AtLocation', is_inv=True):
        if is_inv:
            ts = self.candidate[rel]['h']
            rel = 'inv: ' + relation_description[rel]
        else:
            ts = self.candidate[rel]['t']
            rel = relation_description[rel]

        if len(ts) == 0:
            return {'h': head, 'r': rel, 't': []}

        inputs = self.model.tokenizer(head, rel, ts)

        # move to cuda
        inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            v_hr = self.model(inputs['hr_input_ids'], inputs['hr_attention_mask'], inputs['hr_token_type_ids'])

            v_t = []
            for i in tqdm(range(math.ceil(len(ts) / self.batch_size)), desc='Prepare v_t ...'):
                v_t_ = self.model(inputs['t_input_ids'][i * self.batch_size:(i + 1) * self.batch_size],
                             inputs['t_attention_mask'][i * self.batch_size:(i + 1) * self.batch_size],
                             inputs['t_token_type_ids'][i * self.batch_size:(i + 1) * self.batch_size])
                v_t.append(v_t_)
            v_t = torch.cat(v_t)
            score = self.model.phi(v_hr, v_t)

        value, indices = torch.topk(score, min(10, len(ts)))

        result = {'h': head, 'r': rel, 't': []}
        for v, i in zip(value[0], indices[0]):
            result['t'].append((ts[i], v.item()))

        return result
