import sys
from urllib import parse
sys.path.append('service/gen-py')
from ckqa import CKQA
from ckqa.ttypes import Result
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

# -*- coding: utf-8 -*-
from package.kb import GConceptNetCS
from package.sent import SentParser, SentSimi, SentMaker, join_sents
from package.qa import MaskedQA, SpanQA

from wobert import WoBertTokenizer

class CKQAHandler:
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

    def getMaskWordResult(self, query):
        q = parse.unquote(query)

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

        engine = MaskedQA('junnyu/wobert_chinese_plus_base', WoBertTokenizer)
        q = q.replace('[MASK]', engine.mask_token)
        context = join_sents(context, lang='zh')

        result = engine(q, context)
        result_without_context = engine(q, '')

        print(result_without_context)
        print(result)

        return [Result('none', result_without_context, ''), Result('ckqa', result, context)]


if __name__ == '__main__':
    handler = CKQAHandler()
    processor = CKQA.Processor(handler)
    transport = TSocket.TServerSocket(host='192.168.10.162', port=8327)
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
