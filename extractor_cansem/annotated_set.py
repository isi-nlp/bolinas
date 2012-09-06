import re, sys
from collections import defaultdict

from lib.amr.amr import Amr, Dag
from data_structures import AMRtriple,AMR,Edge,tibFormat

e2d_matcher = re.compile("([^ ]+) ([^ ]+) ([^\t ]+)[ \t]*([^\(])*\(([0-9]+),([0-9]+)\)")
e2s_matcher = re.compile("([^ ]+) ([^ ]+) ([^\t ]+)[ \t]*([^\-]*)-([0-9]+)")
e2sroot_matcher = re.compile("ROOT ([^-]+)-([0-9]+)")
deptree_matcher = re.compile("([^\(]+)\(([^-]+)-([0-9]+), ([^-]+)-([0-9]+)\)")

#align_matcher = re.compile("^([^:]+):([^:]+):([^-]+)-([0-9]+)$")

class AnnotatedSet():


    def __init__(self):
        '''
        Initialise an empty array of sentences
        '''
        self.data = {}
        self.number = 0

    @classmethod
    def loadText(cls,args,l):
        '''
        Load text only
        file.language.txt -> data['plain_text'][i]
        '''

        c = cls()
        try:
            f = open("%s/%s.%s.txt"%(args.dataBase,args.dataset,l))
            count = c.loadTxtFile(f,args.fromLine,args.toLine)
            print "Loaded %d sentences (text)" % count
            f.close()
        except IOError as e:
            print "Could not open text file"

        c.convertToS()
        print "Converted to Sentences format"

        return c


    @classmethod
    def loadTextAndAMR(cls,args,l):
        '''
        Load all the annotations we can find for use in training

        file.amr
        file.language.txt

        to
        data['tokens'][i]
        data['amr'][i]

        '''

        c = cls()

        try:
            f = open("%s/%s.amr"%(args.dataBase,args.dataset))
            count = c.loadAMR(f,args.fromLine,args.toLine) # get Daniel's AMR class and mine for each
            print "Loaded %d sentences (amr and d-amr)" % count
            c.number = count
            f.close()
        except IOError as e:
            print "Could not load AMR file"

        try:
            f = open("%s/%s.%s.txt"%(args.dataBase,args.dataset,l))
            count = c.loadTxtFile(f,args.fromLine,args.toLine)
            print "Loaded %d sentences (text)" % count
            f.close()
        except IOError as e:
            print "Could not open text file"

        c.delexAMR()
        print "Delexicalised AMRs"

        c.convertToS()
        print "Converted to Sentences format"

        return c

    @classmethod
    def loadTraining(cls,args,l):
        '''
        Load all the annotations we can find for use in training

        file.amr
        file.language.align
        file.language.txt
        file.language.deps
        file.language.syn
        file.language.tag

        to
        data['align'][i]
        data['tokens'][i]
        data['amr'][i]
        data['danielamr'][i]
        data['plain_text'][i]

        '''

        c = cls()

        try:
            f = open("%s/%s.amr"%(args.dataBase,args.dataset))
            count = c.loadAMR(f,args.fromLine,args.toLine) # get Daniel's AMR class and mine for each
            print "Loaded %d sentences (amr and d-amr)" % count
            c.number = count
            f.close()
        except IOError as e:
            print "Could not load AMR file"

        try:
            if args.align == "gold":
                print "%s/%s.%s.%s.align"%(args.dataBase,args.dataset,l,args.align)
                f = open("%s/%s.%s.%s.align"%(args.dataBase,args.dataset,l,args.align))
            else:
                f = open("%s/out/%s.%s.%s.align"%(args.dataBase,args.dataset,l,args.align))
            count = c.loadAlign(f,args.fromLine,args.toLine)
            print "Loaded %d sentences (alignments)" % count
            if c.data['amr']:
                (sc,ac) = c.applyAlign()
                print "Applied %d alignments on %d sentences" % (ac,sc)
            f.close()
        except IOError as e:
            print "Could not open alignment file"

        try:
            f = open("%s/%s.%s.txt"%(args.dataBase,args.dataset,l))
            count = c.loadTxtFile(f,args.fromLine,args.toLine)
            print "Loaded %d sentences (text)" % count
            f.close()
        except IOError as e:
            print "Could not open text file"

        c.delexAMR()
        print "Delexicalised AMRs"

        c.convertToS()
        print "Converted to Sentences format"

        return c

    def convertToS(self):

        self.sentences = []

        for i in range(len(self.data[self.data.keys()[0]])):

            self.sentences.append({})
            for key in self.data.keys():
                self.sentences[i][key] = self.data[key][i]


    def loadAMR(self,inFile,fLine=0,tLine=None):
        myamrs = []
        danielamrs = []
        plainamr = []
        count = 0
        for i in range(fLine):
            inFile.next()
        j = fLine
        if tLine == None: tLine = 100000000
        while j < tLine:
            try: line = inFile.next()
            except:
                break
            j += 1
            count += 1
            if count%500 == 0:  print "Loaded %d lines (AMR)"%(count)
            danielamr = Amr.from_string(line) #.stringify()
            myamr = AMR()
            for triple in danielamr.triples(instances=False):
                myamr.add_triple(AMRtriple(triple[0],triple[1],triple[2][0],1)) # Argh, Daniel changing underlying structure!
                #print triple[0],triple[1],triple[2][0]
            for root in danielamr.roots:
                #myamr.add_triple(AMRtriple('ROOT','ROOT',root,1))
                myamr.add_root(root)
            danielamrs.append(danielamr)
            myamrs.append(myamr)
        self.data['danielamr'] = danielamrs
        self.data['amr'] = myamrs
        assert (count == self.number or self.number == 0), "Fuc"
        return count

    def loadAlign(self,inFile,fLine=0,tLine=None):
        self.data['align'] = []
        count = 0
        for i in range(fLine):
            inFile.next()
        j = fLine
        if tLine == None: tLine = 100000000
        while j < tLine:
            try: line = inFile.next()
            except: return count
            j += 1
            if count%500 == 0:  print "Loaded %d lines (Alignments)"%(count)
            count += 1
            alignments = line.split()
            alignList = []
            for a in alignments:
                alignList.append(a)
            self.data['align'].append(alignList)
        assert (count == self.number or self.number == 0), "Fuc"
        return count

    def applyAlign(self,align=False):
        if not align:   align = self.data['align']
        sCount = 0
        aCount = 0
        for i in range(self.number):
            if i%500 == 0:  print "Applied %d of %d alignments"%(i,self.number)
            sCount += 1
            for a in align[i]:
                # Alignment: x9:prep-to:sell-01:x11-39
                fromN,rest = a.split(":",1)
                rest,tokPos = rest.rsplit("-",1)
                withN,toN = rest.rsplit(":",1)
                hasAligned = False
                for triple in self.data['amr'][i].triples:
                    if (triple.a,triple.b,triple.c) == (fromN,withN,toN):
                        triple.align.add(tokPos)
                        #triple.align.add("%s-%d" % (word, tokPos))
                        #triple.tags[int(toTok)] = s['tags'][int(tokPos)]
                        aCount += 1
                        hasAligned = True
                    elif triple.a == "root0" and (triple.a,triple.b) == (fromN,withN):
                        triple.align.add(tokPos)
                        aCount += 1
                        hasAligned = True
                if not hasAligned:
                    print "Alignment failure in sentence #%d: (%s,%s,%s,%s)"%(i,fromN,withN,toN,tokPos)
        return (sCount,aCount)

    def loadTxtFile(self,inFile,fLine=0,tLine=None):
        count = 0
        self.data['tokens'] = []
        self.data['plain_text'] = []
        for i in range(fLine):
            inFile.next()
        j = fLine
        if tLine == None: tLine = 100000000
        while j < tLine:
            try: line = inFile.next()
            except: return count
            j += 1
            self.data['plain_text'].append(tibFormat(line.strip()))
            if count%500 == 0:  print "Loaded %d lines (Text)"%(count)
            count += 1
            tokens = line.split()
            sentence = []
            for t in tokens:
                sentence.append(tibFormat(t))
            self.data['tokens'].append(sentence)
        return count

        
    def buildDepTree(self,s):

        #depTree = Dag()
        depParent = defaultdict(int)
        dTtriples = []
        dTroots = []
        for line in s['plain_typed'].splitlines():
            match = deptree_matcher.match(line)
            if match:
                depType = match.group(1)
                fromW = match.group(2)
                fromT = int(match.group(3))
                toW = match.group(4)
                toT = int(match.group(5))
                if depType != 'xsubj':
                    depParent[toT] = fromT
                    dTtriples.append((fromT,depType,toT))
                    #print (fromT,depType,toT)
                if depType == 'root':
                    dTroots.append(fromT)
        s['depTree'] = Dag.from_triples(dTtriples,dTroots)
        s['depParent'] = depParent

    def delexAMR(self):
        '''
        Takes the existing s['amr'] and builds the first step towards a delexicalised version.
        At this stage this requires an additional level of nodes s.t.:
    
        (a,*,b) --> (a,ARG0,b) --> (a,ARG0(boy),b)

        This transformation allows us to delexicalise rules to some degree

        '''

        self.data['delexAMR'] = []
        for i in range(self.number):

            self.data['delexAMR'].append(AMR())

            for oldtrip in self.data['amr'][i].triples:
                (a,b,c) = oldtrip
                argconcept = b.split(":",1)
                '''
                Hacks to deal with the non-AMR argumentless graphs
                '''
                if len(argconcept) == 2:
                    arg,concept = argconcept
                elif b == "root":     
                    arg = "ROOT"
                    concept = b
                else:
                    arg = "pseudoarg"
                    concept = b
                #if c in self.data['danielamr'][i].node_to_concepts.keys():
                #    concept = self.data['danielamr'][i].node_to_concepts[c]

                trip = AMRtriple(a,Edge(arg,concept),c,2)
                trip.align = oldtrip.align
                #trip.span = oldtrip.span
                #trip.tags = oldtrip.tags

                self.data['delexAMR'][i].add_triple(trip)

