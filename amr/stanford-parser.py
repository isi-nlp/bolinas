from amr import Dag
import re

'''
Parse stanford dependencies into DAGs.

@author: Daniel Bauer (dbauer)
@since: 2012-06-28
'''

rel_re = re.compile("(?P<rel>[^(]+)\((?P<parent>[^,]+), (?P<child>[^\)]+)\)")

class StanfordParse(Dag):
        

    def add_dependency(self,s):
        
        match = rel_re.match(s)
        parent = match.group("parent")
        child = match.group("child")
        rel = match.group("rel")
        print parent, child, rel
        self.add_triple(parent, rel, child)
    
    @classmethod
    def from_string(cls, s):
        dep_strings = [x.strip() for x in s.split("\n")]
        print dep_strings
        new = StanfordParse()
        for dep_s in dep_strings:
            if dep_s:    
                new.add_dependency(dep_s)
        new.roots = new.find_roots()
        return new


p = StanfordParse.from_string("""
det(House-2, The-1)
nsubj(voted-3, House-2)
root(ROOT-0, voted-3)
aux(boost-5, to-4)
xcomp(voted-3, boost-5)
det(wage-9, the-6)
amod(wage-9, federal-7)
amod(wage-9, minimum-8)
dobj(boost-5, wage-9)
prep(voted-3, for-10)
det(time-13, the-11)
amod(time-13, first-12)
pobj(for-10, time-13)
prep(time-13, since-14)
amod(1981-16, early-15)
pobj(since-14, 1981-16)
xcomp(voted-3, casting-18)
det(vote-22, a-19)
amod(vote-22, solid-20)
num(vote-22, 382-37-21)
dobj(casting-18, vote-22)
prep(vote-22, for-23)
det(measure-26, a-24)
nn(measure-26, compromise-25)
pobj(for-23, measure-26)
partmod(measure-26, backed-27)
prep(backed-27, by-28)
nn(Bush-30, President-29)
pobj(by-28, Bush-30)
                              """)
