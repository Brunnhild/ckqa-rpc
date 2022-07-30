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

    def exist(self, subject, relation, object, index=None):
        print({
            'subject': subject,
            'relation': relation,
            'object': object
        })
        resp = self.es.search(index=index if index is not None else self.index, query={
            'bool': {
                'must': [
                    {
                        'match': {
                            'subject': subject
                        }
                    },
                    {
                        'match': {
                            'relation': relation
                        }
                    },
                    {
                        'match': {
                            'object': object
                        }
                    }
                ]
            }
        })

        return len(resp['hits']['hits']) > 0

    def update(self, id, doc):
        self.es.update(index=self.index, id=id, body={
            'doc': doc
        })
        print(f'updated: {id}')

    def insert(self, doc):
        resp = self.es.index(index=self.index, document=doc)
        print(resp)
