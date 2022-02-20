# -*- coding: utf-8 -*-
from package.gstore import GStore
import json

CST = ('AtLocation', 'CapableOf', 'Causes', 'CausesDesire', 'CreatedBy', 'Desires', 'HasA', 'HasFirstSubevent', 'DefinedAs',
       'HasLastSubevent', 'HasPrerequisite', 'HasProperty', 'HasSubEvent', 'InstanceOf', 'MadeOf', 'MotivatedByGoal',
       'UsedFor', 'PartOf', 'ReceivesAction')


class GConceptNetCS:
    """
    Use gStore as Backend
    """

    def __init__(self, ip='127.0.0.1', port=9000, usr='root', pwd='123456'):
        self.store = GStore(ip, port, usr, pwd)

    def query(self, entity):
        entity = entity.lower()

        q1 = 'select ?p ?o where { <' + entity + '> ?p ?o. }'
        q2 = 'select ?s ?p where { ?s ?p <' + entity + '>. }'

        a1 = self.store.query('cskg', q1)
        a1 = self._filter_results(entity, a1)

        a2 = self.store.query('cskg', q2)
        a2 = self._filter_results(entity, a2)

        return a1 + a2

    def _filter_results(self, entity, results):
        results = json.loads(results)
        return [(item['s']['value'] if 's' in item else entity,
                 item['p']['value'],
                 item['o']['value'] if 'o' in item else entity)
                for item in results['results']['bindings'] if item['p']['value'] in CST]