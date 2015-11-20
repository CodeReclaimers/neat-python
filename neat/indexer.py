

class Indexer(object):
    def __init__(self, first):
        self.next_id = first

    def next(self):
        result = self.next_id
        self.next_id += 1
        return result
