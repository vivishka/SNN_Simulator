from a import a

class C1(object):
    """docstring for C1."""

    def __init__(self):
        super(C1, self).__init__()
        self.dict = {}

    def get_dict(self):
        return self.dict

    def set_dict(self, dict):
        self.dict = dict


c1 = C1()
c2 = C1()

c1.set_dict({1: 'a', 2: 'b'})
c2.set_dict({3: 'c', 4: 'd'})

c1.dict = c2.dict

c2.dict = {5: 'e'}

print(c1.get_dict())
