import sys
import re
import argparse
import subprocess
import os

from lib.amr.amr import Amr

from data_structures import tibFormat

from conversion.ghkm2tib import ghkm2tib
from canonical import CanonicalGrammar
from derivation_tree import DerivationTree

from interfaces import *
from filters import createDatabases, queryString2Tree, queryTree2String

def executeTraining(args,data):
    srilmMake(args)

    if args.javaGHKM:   parseFile = open(args.files['javaTree'],'w')
    else:               parseFile = open(args.files['cppTree'],'w')
    aFile = open(args.files['aFile'],'w')
    fFile = open(args.files['fFile'],'w')
    tfFile = open(args.files['tfFile'],'w')

    for s in data.sentences:
        cg = CanonicalGrammar(s)
        for d in cg.derivs_done:
            aGraph = DerivationTree.fromDerivation(d)
            if args.javaGHKM:   (parse,a) = aGraph.getGHKMtriple_Java()
            else:               (parse,a) = aGraph.getGHKMtriple_CPP()
            tibTree = aGraph.getTiburonTree()
            text = s["plain_text"].strip(' \t\n\r') #.lower()
            parseFile.write("%s\n"%parse)
            aFile.write("%s\n"%a)
            tfFile.write("%s\n%s\n"%(tibTree,text))
            fFile.write("%s\n"%text)

    parseFile.close()
    aFile.close()
    fFile.close()
    tfFile.close()

    if args.javaGHKM:
        print "Running GHKM Java rule extraction"
        runJavaGHKM(args)
        print "Converting GHKM rules to Tiburon format"
        ghkm2tib(args.files['ghkmFile'],args.files['tibRaw'],"Java")
    else:
        print "Running GHKM C++ rule extraction"
        runCPPGHKM(args)
        print "Converting GHKM rules to Tiburon format"
        ghkm2tib(args.files['ghkmFile'],args.files['tibRaw'],"C++")
    
    tibEMTraining(args)
    if args.xrsfilter: createDatabases(args)

def executeStr2Str(args,data1,data2,lang1,lang2,doSmatch=False):

    lm = srilmLoad(args,lang2)
    goldSentences = []
    inputSentences = []
    for i in range(len(data1.sentences)):
        inputSentences.append(data1.sentences[i]['plain_text'])
        goldSentences.append(tibFormat(data2.sentences[i]['plain_text'],reverse=True))
    genSentences = []

    fromK = 0
    toK = 0
    genSentences = []
    index = 0

    while toK < len(inputSentences):
        toK += args.tibStepSize
        treeList = tibString2AMRBatch(args,lang1,inputSentences[fromK:toK],index) # We only care about the tree
        genSentences += [srilmTop(args,lm,i) for i in tibTree2StringBatch(args,lang2,treeList,index)]
        fromK = toK
        index += 1

    score = runBleuBatch(args,goldSentences,genSentences,inputSentences)
    print "Average BLEU score %f"%(score['score'])


def executeStr2StrFiltered(args,data1,data2,lang1,lang2,doSmatch=False):

    outAMR = open(args.files['s2a_t_amr'],'w')
    lm = srilmLoad(args,lang2)

    if args.xrsfilter:  queryString2Tree(args,data1,lang1)

    # Generating Gold Sentences
    goldSentences = []
    for i in range(len(data1.sentences)):
        goldSentences.append(tibFormat(data2.sentences[i]['plain_text'],reverse=True))


    inputSentences = []

    # Going from String to Tree
    trees = []
    for index in range(len(data1.sentences)):
        inputSentences.append(data1.sentences[index]['plain_text'])
        s = data1.sentences[index]
        auto_amr,tree = tibString2AMR(args,lang1,s['plain_text'],index,args.xrsfilter)
        if tree:
            trees.append(tree)
            outAMR.write("%s\n"%auto_amr.to_string())
        else:       
            trees.append("EMPTY()")
            outAMR.write("\n")

    outAMR.close()
    # Second Query
    if args.xrsfilter:  queryTree2String(args,trees,lang2)

    # From Trees to Strings
    genSentences = []
    for index in range(len(trees)):
        genStrings = tibTree2String(args,lang2,trees[index],index,args.xrsfilter)
        if genStrings:  genSentences.append(srilmTop(args,lm,genStrings))
        else:           genSentences.append("")
        
    print "Running BLEU"
    score = runBleuBatch(args,goldSentences,genSentences,inputSentences)
    print "Average BLEU score %f"%(score['score'])


def executeStr2AMR(args,data,lang):
    print "String2AMR"

    outAMR = open(args.files['s2a_amr'],'w')
    outDeriv = open(args.files['s2a_deriv'],'w')

    if args.xrsfilter:
        print "Querying Database"
        queryString2Tree(args,data,lang)

    mean_p = 0.0
    mean_r = 0.0
    mean_f = 0.0
    count = 0
    for index in range(len(data.sentences)):
        s = data.sentences[index]
        print "String 2 AMR"
        auto_amr,tree = tibString2AMR(args,lang,s['plain_text'],index,args.xrsfilter)
        print "Smatching it"
        p,r,f = runSmatch(args,s['danielamr'],auto_amr)
        mean_p += p
        mean_r += r
        mean_f += f
        count += 1
        if auto_amr:
            outAMR.write("%s\n"%auto_amr.to_string())
            outDeriv.write("%s\n"%tree)
        else:
            outAMR.write("\n")
            outDeriv.write("\n")
        print "OVERALL RESULT"
        print "P:%f R:%f F:%f " % (mean_p/count, mean_r/count, mean_f/count) 
    outAMR.close()
    outDeriv.close()

def executeAMR2Str(args,data,lang):
    '''
    Takes AMRs and then tries to build string from them
    '''
    goldSentences = []
    genSentences = []

    lm = srilmLoad(args,lang)
    
    trees = []
    for index in range(len(data.sentences)):
        s = data.sentences[index]
        cg = CanonicalGrammar(s)
        goldSentences.append(tibFormat(s['plain_text'],reverse=True))
        if len(cg.derivs_done) == 1:
            deriv = cg.derivs_done[0]
            aGraph = DerivationTree.fromDerivation(deriv)
            tibTree = aGraph.getTiburonTree()
            trees.append(tibTree)
        else:
            trees.append("EMPTY()")

    if args.xrsfilter:
        print "Querying Database"
        queryTree2String(args,trees,lang)

    for index in range(len(trees)):
        genStrings = tibTree2String(args,lang,trees[index],index,args.xrsfilter)
        if genStrings:  genSentences.append(srilmTop(args,lm,genStrings))
        else:           genSentences.append("")
        
    score = runBleuBatch(args,goldSentences,genSentences)
    print "Average BLEU score %f"%(score['score'])

