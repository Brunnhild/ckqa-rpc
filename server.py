import sys
from urllib import parse
sys.path.append('service/gen-py')
from ckqa import CKQA
from ckqa.ttypes import Result, Tuple, Scale, CompletionResult
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

# -*- coding: utf-8 -*-
from packages.ckqa.kb import GConceptNetCS
from packages.ckqa.es import ES
from packages.ckqa.sent import SentParser, SentSimi, SentMaker, join_sents
from packages.ckqa.qa import MaskedQA, SpanQA, FreeQA
from packages.completion.api import CompletionModel
from packages.ckqa.v2cTry import v2cPrint
from HybridNet.main_process import process

# from wobert import WoBertTokenizer

from transformers import AutoModelForSequenceClassification, AutoTokenizer, T5TokenizerFast
import torch

from packages.choice.model import T5PromptTuningForConditionalGeneration
from packages.choice.standalone import Standalone

import os

from sentence_transformers import SentenceTransformer

from elasticsearch7.helpers import scan


class RPCHandler:
    def __init__(self) -> None:
        self.completion_model = CompletionModel('./packages/completion/release')
        self.es = ES()
        self.sbert_model = SentenceTransformer('/home/ubuntu/.cache/torch/sentence_transformers/sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2')

    def getLang(self, s):
        if len(s) == len(s.encode()):
            return 'en'
        else:
            return 'zh'

    def deduplicate(self, triple_list):
        all_set = set()
        res = []
        for triple in triple_list:
            if triple[0] == triple[2]:
                continue
            q = ''.join(triple)
            if q not in all_set:
                all_set.add(q)
                res.append(triple)
        return res

    def getMaskResultEnglish(self, q, includeNone, includeCSKG):
        # 解析问题中的实体
        # TODO: 优化自动机代码，提高词典的解析速度；当前为python代码，构建树形结构速度慢。
        parser = SentParser()
        entity = parser.parse(q.replace('[MASK]', ''))
        print('Parsing sentence:', entity)

        # 链接至常识图谱
        # TODO: 增加语义相似匹配代码，当前为字符匹配，默认为小写
        context = []
        for e in entity:
            context.extend(self.es.query(e, size=None))
        context = self.deduplicate(context)
        print('Context triple:', context)

        # 将检索到的三元组组合成自然语言
        maker = SentMaker()
        context = [maker.lexicalize(triple) for triple in context]
        print('Context sentence:', context)

        context_sim = SentSimi()
        context, _ = context_sim.lookup(q, context, k=10)
        print('Query-related sentence:', context)

        engine = MaskedQA('roberta-large')
        q = q.replace('[MASK]', engine.mask_token)
        context = join_sents(context)

        res = []
        if includeNone:
            result_without_context = engine(q, '')
            print(result_without_context)
            res.append(Result('none', result_without_context, ''))
        if includeCSKG:
            result = engine(q, context)
            print(result)
            res.append(Result('ckqa', result, context))

        return res

    def getMaskResultChinese(self, q, includeNone, includeCSKG):
        # 解析问题中的实体
        # TODO: 优化自动机代码，提高词典的解析速度；当前为python代码，构建树形结构速度慢。
        parser = SentParser(name='zh_core_web_sm', user_dict=[])
        entity = parser.parse(q.replace('[MASK]', ''))
        print('Parsing sentence:', entity)

        # 链接至常识图谱
        # TODO: 增加语义相似匹配代码，当前为字符匹配，默认为小写
        context = []
        for e in entity:
            context.extend(self.es.query(e, size=None))
        context = self.deduplicate(context)
        print('Context triple:', context)

        # 将检索到的三元组组合成自然语言
        maker = SentMaker()
        context = [maker.lexicalize_zh(triple) for triple in context]
        print('Context sentence:', context)

        context_sim = SentSimi()
        context, _ = context_sim.lookup(q, context, k=10)
        print('Query-related sentence:', context)

        engine = FreeQA('/mnt/ssd/wyt/transformers_models/bart-base-chinese')
        context = join_sents(context, lang='zh')

        res = []
        if includeNone:
            result_without_context = engine(q, '')
            print(result_without_context)
            res.append(Result('None', result_without_context, ''))
        if includeCSKG:
            result = engine(q, context)
            print(result)
            res.append(Result('CSKG', result, context))

        return res

    def getSpanResultChinese(self, q):
        # 解析问题中的实体
        # TODO: 优化自动机代码，提高词典的解析速度；当前为python代码，构建树形结构速度慢。
        parser = SentParser(name='zh_core_web_sm')
        entity = parser.parse(q)
        print('Parsing sentence:', entity)

        # 链接至常识图谱
        # TODO: 增加语义相似匹配代码，当前为字符匹配，默认为小写
        kb = GConceptNetCS('192.168.10.174')
        context = []
        for e in entity:
            context.extend(kb.query(e))
        print('Context triple:', context)

        # 将检索到的三元组组合成自然语言
        maker = SentMaker()
        context = [maker.lexicalize_zh(triple) for triple in context]
        print('Context sentence:', context)

        context_sim = SentSimi()
        context = context_sim.lookup(q, context, k=5)
        print('Query-related sentence:', context)

        engine = SpanQA('luhua/chinese_pretrain_mrc_roberta_wwm_ext_large')
        context = join_sents(context, lang='zh')
        result = engine(q, context)

        print(result)

        return [Result('ckqa', [result], context)]

    def getSpanResultEnglish(self, q):
        # 解析问题中的实体
        # TODO: 优化自动机代码，提高词典的解析速度；当前为python代码，构建树形结构速度慢。
        parser = SentParser()
        entity = parser.parse(q)
        print('Parsing sentence:', entity)

        # 链接至常识图谱
        # TODO: 增加语义相似匹配代码，当前为字符匹配，默认为小写
        kb = GConceptNetCS('192.168.10.174')
        context = []
        for e in entity:
            context.extend(kb.query(e))
        print('Context triple:', context)

        # 将检索到的三元组组合成自然语言
        maker = SentMaker()
        context = [maker.lexicalize(triple) for triple in context]
        print('Context sentence:', context)

        context_sim = SentSimi()
        context = context_sim.lookup(q, context, k=5)
        print('Query-related sentence:', context)

        engine = SpanQA('mrm8488/spanbert-finetuned-squadv2')
        context = join_sents(context)
        result = engine(q, context)

        print(result)

        return [Result('ckqa', [result], context)]

    def getMaskResult(self, query, includeNone, includeCSKG):
        q = parse.unquote(query)

        if self.getLang(q) == 'en':
            return self.getMaskResultEnglish(q, includeNone, includeCSKG)
        else:
            return self.getMaskResultChinese(q, includeNone, includeCSKG)

    def getSpanResult(self, query):
        # q = 'Lions like to eat [MASK].'
        q = parse.unquote(query)

        if self.getLang(q) == 'en':
            return self.getSpanResultEnglish(q)
        else:
            return self.getSpanResultChinese(q)

    def getTextQaResult(self, query, text):
        path_to_model = 'packages/choice/ipoie'
        max_step = 10
        device = -1

        tokenizer = T5TokenizerFast.from_pretrained(path_to_model)
        model = T5PromptTuningForConditionalGeneration.from_pretrained(path_to_model)

        standalone = Standalone(model=model, tokenizer=tokenizer, max_step=max_step, device=device)
        extraction = standalone.pipeline([text], batch_size=32)

        triples = map(lambda x: (x[0][1], x[0][0], x[0][2]), extraction[text].items())
        # 将检索到的三元组组合成自然语言
        maker = SentMaker()
        context = [maker.lexicalize(triple) for triple in triples]
        print('Context sentence:', context)

        context_sim = SentSimi()
        context, _ = context_sim.lookup(query, context, k=5)
        print('Query-related sentence:', context)

        engine = FreeQA('/mnt/ssd/wyt/transformers_models/bart-base-chinese')
        context = join_sents(context, lang='zh')
        result = engine(query, context)

        return [Result('Text', result, context)]

    def getExtraction(self, query):
        path_to_model = 'packages/choice/ipoie'
        max_step = 10
        device = -1

        tokenizer = T5TokenizerFast.from_pretrained(path_to_model)
        model = T5PromptTuningForConditionalGeneration.from_pretrained(path_to_model)

        standalone = Standalone(model=model, tokenizer=tokenizer, max_step=max_step, device=device)
        extraction = standalone.pipeline([query], batch_size=32)

        def get_embedding(items):
            return self.sbert_model.encode(items[1] + items[0] + items[2])

        res = map(lambda x: Tuple(x[0], x[1], get_embedding(x[0])), extraction[query].items())
        return list(res)

    def getEntailment(self, premise, hypothesises):
        # pose sequence as a NLI premise and label as a hypothesis
        nli_model = AutoModelForSequenceClassification.from_pretrained('/mnt/ssd/wyt/transformers_models/bart-large-mnli')
        tokenizer = AutoTokenizer.from_pretrained('/mnt/ssd/wyt/transformers_models/bart-large-mnli')

        res = []
        for hypothesis in hypothesises:
            # run through model pre-trained on MNLI
            with torch.no_grad():
                x = tokenizer.encode(premise, hypothesis, return_tensors='pt',
                                     truncation_strategy='only_first')
                logits = nli_model(x)[0]

                # we throw away "neutral" (dim 1) and take the probability of
                # "entailment" (2) as the probability of the label being true
                entail_contradiction_logits = logits[:, [0, 2]]
                probs = entail_contradiction_logits.softmax(dim=1)
                prob_label_is_true = probs[:, 1]

                res.append(prob_label_is_true.item())

        return res

    def get_cms(self,query,video):
        #cms,queries=v2cPrint(query,video)
        cms,queries=process(query,video)
        print("server cms:")
        print(cms)
        print("server queries:")
        print(queries)

        cms['query1'] = queries[0]
        cms['query2'] = queries[1]
        cms['query3'] = queries[2]
        cms['video'] = "video"+str(video)
        return cms

    def getScale(self):
        all_entities = {}
        all_entities_count = 0
        all_entities_count_cn = 0

        res = scan(
            client=self.es.es,
            index=self.es.index,
            query={
                'query': {
                    'match_all': {}
                }
            }
        )

        for item in res:
            source = item['_source']
            for ent in [source['subject'], source['object']]:
                if ent not in all_entities:
                    all_entities[ent] = True
                    all_entities_count += 1
                    if source['lang'] == 'zh':
                        all_entities_count_cn += 1

        return Scale(all_entities_count, all_entities_count_cn)

    def getCompletion(self, head, rel, isInv):
        res = []
        for item in self.completion_model.predict(head, rel, isInv)['t']:
            exist = self.es.exist(head, rel, item[0])
            res.append(CompletionResult(item[0], item[1], exist))
        return res

    def upsert(self, id, subject, relation, object):
        doc = {
            'subject': subject,
            'relation': relation,
            'object': object,
            'lang': self.getLang(f'{subject}{object}')
        }
        maker = SentMaker()
        if doc['lang'] == 'en':
            doc['query'] = maker.lexicalize((subject, relation, object))
        else:
            doc['query'] = maker.lexicalize_zh((subject, relation, object))
        doc['vector'] = self.sbert_model.encode(doc['query']).tolist()

        if id is not None and len(id) > 0:
            self.es.update(id, doc)
        else:
            self.es.insert(doc)


if __name__ == '__main__':
    pid = os.getpid()
    with open('./pid', 'w') as f:
        f.write(str(pid))

    handler = RPCHandler()
    processor = CKQA.Processor(handler)
    transport = TSocket.TServerSocket(host='0.0.0.0', port=8327)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)

    # You could do one of these for a multithreaded server
    # server = TServer.TThreadedServer(
    #     processor, transport, tfactory, pfactory)
    # server = TServer.TThreadPoolServer(
    #     processor, transport, tfactory, pfactory)

    print('Starting the server...')
    server.serve()
    print('done.')
