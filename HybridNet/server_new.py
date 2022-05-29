import sys
from HybridNet.gen_py_new.v2c import v2c

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

#sys.path.append('../HybridNet')
from HybridNet.main_process import process
#sys.path.append('/HybridNet/')

class v2c_handler:
    def __init__(self):
        print("init v2c-server")
    def get_cms(self,query,video):
        #opt = parse_opt()
        #opt = vars(opt)
        #opt['cuda'] = True
        cms,queries=process(query,video)
        print("server cms:")
        print(cms)
        print("server queries:")
        print(queries)
        #result={}
        #result['cms'] = cms
        cms['query1'] = queries[0]
        cms['query2'] = queries[1]
        cms['query3'] = queries[2]
        cms['video'] = "video"+str(video)
        return cms

handler=v2c_handler()

processor = v2c.Processor(handler)

transport = TSocket.TServerSocket(port=9090)

t_factory = TTransport.TBufferedTransportFactory()
p_factory = TBinaryProtocol.TBinaryProtocolFactory()

server = TServer.TSimpleServer(processor, transport, t_factory, p_factory)
print('Starting the server...')

server.serve()
print('done.')

