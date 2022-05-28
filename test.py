# -*- coding: utf-8 -*-
from elasticsearch7 import Elasticsearch


class ES:
    def __init__(self):
        self.es = Elasticsearch(hosts='http://127.0.0.1:9200')

    def query(self, entity):
        resp = self.es.search(index='cskg_vector', query={
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
        return [{
            'subject': i['_source']['subject'],
            'relation': i['_source']['relation'],
            'object': i['_source']['object']
        } for i in resp['hits']['hits']]


if __name__ == '__main__':
    es = ES()
    print(es.query('狮子'))
