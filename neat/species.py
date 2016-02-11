class Species(object):
    """ A collection of genetically similar individuals."""

    def __init__(self, representative, ID):
        assert type(ID) == int
        self.representative = representative
        self.ID = ID
        self.age = 0
        self.members = []

        self.add(representative)

    def add(self, individual):
        individual.species_id = self.ID
        self.members.append(individual)
