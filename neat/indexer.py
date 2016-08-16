class Indexer(object):
    def __init__(self, first):
        self.next_id = first

    def get_next(self, result=None):
        '''
        If result is not None, then we return it unmodified.  Otherwise,
        we return the next ID and increment our internal counter.
        '''
        if result is None:
            result = self.next_id
            self.next_id += 1
        return result
