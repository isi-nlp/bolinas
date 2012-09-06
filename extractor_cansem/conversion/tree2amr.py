import sys, re, math
from lib.amr.amr import Amr

from extractor_cansem.data_structures import tibFormat,galFormat
'''
ROOT_SL(ROOT_DL(want-01) ARG0_SO(ARG0_DL(boy) ARG1_SL(ARG1_DL(believe-01) ARG0_SO(ARG0_DL(girl) ARG1_DL(boy))))) # -13.066554
ROOT_SL(ROOT_DL(want-01) ARG0_SO(ARG0_DL(boy) ARG1_SL(ARG1_DL(believe-01) ARG0_SW(ARG0_DL(girl) ARG1_DL(boy))))) # -13.066554
'''

def evaluateLine(line,tripleDone,tripleStack,conceptDict,nodeCount,edgeCount,conceptCount):
    '''
    We assume a single start edge.
    Following the rules, we create additional edges and nodes
    During delexicalisation, all copies of a node will be delexicalised
    At a second delexicalisation of re-entrant nodes this is being checked for constitency
    There is some non-determinism with the ordering of SO rules:
        Create a copy of the current state. If inconsistency is found, return this copy for a new attempt
    '''

    currentBuf = ""
    resetStack = [] # Use this to store alternative restarting points. If we break the evaluation, we try again from there

    line = tibFormat(line,reverse=True)

    for i in range(len(line)):
        char = line[i]
        if char == "(" or char == ")" or char == " ":
            if not currentBuf or currentBuf == "#":
                continue

            match = currentBuf.rsplit("_",1)
            if len(match) == 2 and char == "(":
                '''
                We have a rule (i.e. ARG0_SL)
                Let's figure out which, and create the according nodes
                '''
                edge = match[0]
                rule = match[1]
                myEdge = tripleStack.pop()
                (myA,myLabel,myB) = myEdge
                if rule == "DL": # rename the edge label
                    newEdge = (myA,edge,myB)
                    tripleStack.append(newEdge)
                elif rule == "SL": # add a second edge
                    edge1 = (myA,myLabel,"Node%d"%nodeCount)
                    edge2 = ("Node%d"%nodeCount,"Edge%d"%edgeCount,myB)
                    nodeCount += 1
                    edgeCount += 1
                    tripleStack.append(edge2)
                    tripleStack.append(edge1)
                elif rule == "JH": # add a second edge
                    edge1 = (myA,myLabel,"Node%d"%nodeCount)
                    edge2 = (myB,"Edge%d"%edgeCount,"Node%d"%nodeCount)
                    nodeCount += 1
                    edgeCount += 1
                    tripleStack.append(edge2)
                    tripleStack.append(edge1)
                elif rule == "OL": # add a second edge. because of non-det create an alternative starting point
                    edge1 = (myA,myLabel,myB)
                    edge2 = (myA,"Edge%d"%edgeCount,"Node%d"%nodeCount)
                    nodeCount += 1
                    edgeCount += 1
                    tripleStack.append(edge2)
                    tripleStack.append(edge1)

                elif rule == "OR": # add a second edge. because of non-det create an alternative starting point
                    edge1 = (myA,myLabel,myB)
                    edge2 = (myA,"Edge%d"%edgeCount,"Node%d"%nodeCount)
                    nodeCount += 1
                    edgeCount += 1
                    tripleStack.append(edge1)
                    tripleStack.append(edge2)

                elif rule == "SW": # add a second edge
                    edge1 = (myA,myLabel,myB)
                    edge2 = (myA,"Edge%d"%edgeCount,myB)
                    edgeCount += 1
                    tripleStack.append(edge2)
                    tripleStack.append(edge1)
                elif rule == "LL": # add a child edge, stay put
                    edge1 = (myA,myLabel,myB)
                    edge2 = (myB,"Edge%d"%edgeCount,"Node%d"%nodeCount)
                    nodeCount += 1
                    edgeCount += 1
                    tripleStack.append(edge2)
                    tripleStack.append(edge1)
                elif rule == "CC": # add a child circled edge, stay put
                    edge1 = (myA,myLabel,myB)
                    edge2 = (myB,"Edge%d"%edgeCount,myB)
                    edgeCount += 1
                    tripleStack.append(edge2)
                    tripleStack.append(edge1)
                else:
                    print "Oh, noes: Unexpected Rule %s" % rule
                    print line
                    sys.exit(-1)

            else:
                '''
                This should only trigger if we have a word (i.e. we're within a delex rule)
                '''
                myEdge = tripleStack.pop()
                (myA,myLabel,myB) = myEdge
                concept = currentBuf

                if myB[0] == "N":
                    '''
                    Our current edge still has an abstract end node:
                        Replace abstract node with a new variable
                        Store variable and word in the conceptDict
                        Replace all other instances with that variable
                    '''
                    if concept[0] == '"':
                        #print "Delex Lexical"
                        #print tripleStack
                        #print (myA,myLabel,myB)
                        '''
                        If the 'concept' starts with an ", we know it's a literal, so no variable! 
                        The co-reference check must hence cause the AMR to be illegal by default!
                        '''
                        cVar = "v%d"%(conceptCount)
                        conceptCount += 1
                        newEdge = (myA,myLabel,cVar)
                        tripleDone.append(newEdge)
                        conceptDict[cVar] = concept

                        for i in range(len(tripleStack)):
                            (a,b,c) = tripleStack[i]
                            if myB == a or myB == c:
                                return (False,resetStack,False)
                    else:
                        cVar = "v%d"%(conceptCount)
                        conceptCount += 1
                        newEdge = (myA,myLabel,cVar)
                        tripleDone.append(newEdge)
                        conceptDict[cVar] = concept
                        for i in range(len(tripleStack)):
                            (a,b,c) = tripleStack[i]
                            if myB == a:
                                tripleStack[i] = (cVar,b,c)
                            if myB == c:
                                tripleStack[i] = (a,b,cVar)
                elif myB[0] == "v":
                    '''
                    The node has already been defined with a variable
                    If the variable describes the expected word, that's great and we do nothing
                    Otherwise this is a conflict: return False with the resetStack
                    '''
                    if conceptDict[myB] == concept:
                        tripleDone.append(myEdge)
                    else:
                        return (False,resetStack,False)
                else:
                    print "Unexpected word or input: %s" % currentBuf
                    sys.exit(-1)

            currentBuf = ""
        else:
            currentBuf += char

    assert len(tripleStack) == 0 or len(tripleDone) == 0, "FFFUUU"

    return (True,tripleDone,conceptDict)

def getTopAMR(inFile, skip):

    try:
        inf = open(inFile,'r')
    except IOError as e:
        print "File doesn't exist %s" % inFile
        return False,False

    try:
        for i in range(skip):
            inf.next()
    except:
        print "File doesn't contain enough rows"
        print "File: %s" % inFile
        return False,False

    for line in inf: 
        if line.strip() == "0": return (-1,False)
        # each attempt: (line,tripleDone,tripleStack,conceptDict,nodeCount,edgeCount,conceptCount)
        resetStack = [(line,[],[('Zero','Edge0','Node0')],{},1,1,0)]

        success = False
        while len(resetStack) > 0 and not success:
            # Attempt at solving thing. Need multiple attempts because of unknown SO ordering
            x = resetStack.pop()
            (success,result,conceptDict) = evaluateLine(*x)
            if not success:
                resetStack += result

        if success:
            if len(result) == 0:
                pass
                #print "one negative AMR"
            else:
                try:

                    myAMR = Amr.from_triples(result[1:],conceptDict,[result[0][2]]).to_concept_edge_labels()
                    x = myAMR.to_string()
                    y = Amr.from_string(x)
                    return y,line
                except:
                    pass
                    #print "illegal AMR - next"
        else:
            pass
            #print "one broken AMR" #, line

    print "all negative AMRs"
    return False,False

if __name__=="__main__":

    if len(sys.argv) < 1:
        print "ERROR, need a file name"
        sys.exit(-1)

    fName = sys.argv[1]

    print getTopAMR(fName)

