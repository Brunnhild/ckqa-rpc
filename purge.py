from elasticsearch7 import Elasticsearch
from elasticsearch7.helpers import scan, streaming_bulk


es = Elasticsearch('http://192.168.10.162:9200')


def generate_new_item():
    res = scan(
        client=es,
        index='cskg_vector_new',
        query={
            'query': {
                'match_all': {}
            }
        }
    )
    for item in res:
        source = item['_source']
        if source['subject'] != source['object']:
            yield source


if __name__ == '__main__':
    processed = 0
    for ok, action in streaming_bulk(
        client=es, index='cskg_vector_new_purge', actions=generate_new_item()
    ):
        processed += 1
        if processed % 10000 == 0:
            print('processed: %s' % processed)
