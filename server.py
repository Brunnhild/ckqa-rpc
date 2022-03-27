import sys
from urllib import parse
sys.path.append('service/gen-py')
from ckqa import CKQA
from ckqa.ttypes import Result, Tuple
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

# -*- coding: utf-8 -*-
from packages.ckqa.kb import GConceptNetCS
from packages.ckqa.sent import SentParser, SentSimi, SentMaker, join_sents
from packages.ckqa.qa import MaskedQA, SpanQA

# from wobert import WoBertTokenizer

from transformers import T5TokenizerFast

from packages.choice.model import T5PromptTuningForConditionalGeneration
from packages.choice.standalone import Standalone

import os

from sentence_transformers import SentenceTransformer


class RPCHandler:
    def __init__(self) -> None:
        pass

    def getLang(self, s):
        if len(s) == len(s.encode()):
            return 'en'
        else:
            return 'zh'

    def getMaskResultEnglish(self, q):
        # 解析问题中的实体
        # TODO: 优化自动机代码，提高词典的解析速度；当前为python代码，构建树形结构速度慢。
        parser = SentParser()
        entity = parser.parse(q.replace('[MASK]', ''))
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

        engine = MaskedQA('roberta-large')
        q = q.replace('[MASK]', engine.mask_token)
        context = join_sents(context)
        result = engine(q, context)
        result_without_context = engine(q, '')

        print(result_without_context)
        print(result)

        return [Result('none', result_without_context, ''), Result('ckqa', result, context)]

    def getMaskResultChinese(self, q):
        # 解析问题中的实体
        # TODO: 优化自动机代码，提高词典的解析速度；当前为python代码，构建树形结构速度慢。
        parser = SentParser(name='zh_core_web_sm')
        entity = parser.parse(q.replace('[MASK]', ''))
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

        engine = MaskedQA('hfl/chinese-roberta-wwm-ext')
        q = q.replace('[MASK]', engine.mask_token)
        context = join_sents(context, lang='zh')
        result = engine(q, context)
        result_without_context = engine(q, '')

        print(result_without_context)
        print(result)

        return [Result('none', result_without_context, ''), Result('ckqa', result, context)]

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

    def getMaskResult(self, query):
        q = parse.unquote(query)

        if self.getLang(q) == 'en':
            return self.getMaskResultEnglish(q)
        else:
            return self.getMaskResultChinese(q)

    def getSpanResult(self, query):
        # q = 'Lions like to eat [MASK].'
        q = parse.unquote(query)

        if self.getLang(q) == 'en':
            return self.getSpanResultEnglish(q)
        else:
            return self.getSpanResultChinese(q)

    # def getMaskWordResult(self, query):
    #     q = parse.unquote(query)
    #
    #     # 解析问题中的实体
    #     # TODO: 优化自动机代码，提高词典的解析速度；当前为python代码，构建树形结构速度慢。
    #     parser = SentParser(name='zh_core_web_sm')
    #     entity = parser.parse(q.replace('[MASK]', ''))
    #     print('Parsing sentence:', entity)
    #
    #     # 链接至常识图谱
    #     # TODO: 增加语义相似匹配代码，当前为字符匹配，默认为小写
    #     kb = GConceptNetCS('192.168.10.174')
    #     context = []
    #     for e in entity:
    #         context.extend(kb.query(e))
    #     print('Context triple:', context)
    #
    #     # 将检索到的三元组组合成自然语言
    #     maker = SentMaker()
    #     context = [maker.lexicalize_zh(triple) for triple in context]
    #     print('Context sentence:', context)
    #
    #     context_sim = SentSimi()
    #     context = context_sim.lookup(q, context, k=5)
    #     print('Query-related sentence:', context)
    #
    #     engine = MaskedQA('junnyu/wobert_chinese_plus_base', WoBertTokenizer)
    #     q = q.replace('[MASK]', engine.mask_token)
    #     context = join_sents(context, lang='zh')
    #
    #     result = engine(q, context)
    #     result_without_context = engine(q, '')
    #
    #     print(result_without_context)
    #     print(result)
    #
    #     return [Result('none', result_without_context, ''), Result('ckqa', result, context)]

    def getExtraction(self, query):
        path_to_model = 'packages/choice/ipoie'
        max_step = 10
        device = -1

        tokenizer = T5TokenizerFast.from_pretrained(path_to_model)
        model = T5PromptTuningForConditionalGeneration.from_pretrained(path_to_model)

        standalone = Standalone(model=model, tokenizer=tokenizer, max_step=max_step, device=device)
        extraction = standalone.pipeline([query], batch_size=32)

        def get_embedding(items):
            model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            return model.encode(items[1] + items[0] + items[3])

        res = map(lambda x: Tuple(x[0], x[1], get_embedding(x[0])), extraction[query].items())
        return list(res)


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
