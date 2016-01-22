class Indexer(object):
    def __init__(self, first):
        self.next_id = first

    def next(self, result=None):
        '''
        If result is not None, then we return it unmodified.  Otherwise,
        we return the next ID and increment our internal counter.
        '''
        if result is None:
            result = self.next_id
            self.next_id += 1
        return result

    def clear(self):
        self.next_id = 1


class InnovationIndexer(object):
    def __init__(self, first):
        self.indexer = Indexer(first)
        self.innovations = {}

    def get_innovation_id(self, in_node_id, out_node_id):
        innovation_id = self.innovations.get((in_node_id, out_node_id))
        if innovation_id is None:
            innovation_id = self.indexer.next()
            self.innovations[in_node_id, out_node_id] = innovation_id

        return innovation_id
