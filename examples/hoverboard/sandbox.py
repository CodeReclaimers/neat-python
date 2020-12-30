class Fitness:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def __str__(self):
        return str((self.a,self.b))
    def __gt__(self, other):
        return self.a > other.a

fits = [Fitness(a,a*2) for a in range(10)]

print (fits)
print (max(fits))
