# -*- coding: utf-8 -*-
from elasticsearch7 import Elasticsearch


class ES:
    def __init__(self):
        self.es = Elasticsearch(hosts='http://192.168.10.162:9200')
        self.index = 'cskg_combined'

    def query(self, entity, index=None, size=3000):
        resp = self.es.search(index=index if index is not None else self.index, size=size, query={
            'bool': {
                'should': [
                    {
                        'match': {
                            'subject': entity
                        }
                    },
                    {
                        'match': {
                            'object': entity
                        }
                    }
                ]
            }
        })
        return [
            [
                i['_source']['subject'],
                i['_source']['relation'],
                i['_source']['object']
            ] for i in resp['hits']['hits']
        ]
