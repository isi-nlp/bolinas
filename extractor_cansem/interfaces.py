
'''
Interface between the Canonical AMR Algorithm (km.py) and Michael Pust's xrsdb scripts
While we're at it: this contains interfaces to everything else, too

def runJavaGHKM(args):
def runCPPGHKM(args):
def tibString2AMR(args,lang,sentence,index=False):
def tibTree2String(args,lang,tree,index=False):
def tibString2AMRBatch(args,lang,sentences):
def tibTree2StringBatch(args,lang,trees):
def runBleuBatch(args,gold,hypotheses):
def runBleu(args,gold,hypothesis):
def runSmatch(gold_amr,auto_amr):
def generateXrsDB(args):
def tibEMTraining(args,lang):
'''

import sys
import os
import re
import argparse
import subprocess
import operator
import math
from collections import defaultdict

from kenlm import LanguageModel

from lib.amr.amr import Amr, Dag
from lib.amr.new_smatch import compute_smatch_hill_climbing as smatch
from lib.amr.new_smatch import get_random_start

from conversion.tree2amr import getTopAMR
from data_structures import tibFormat

from extra.command import Command


def runJavaGHKM(args):
    mem = "2g"
    ghkm_opts = "-fCorpus %s -eParsedCorpus %s -align %s -joshuaFormat false -maxLHS 200 -maxRHS 15 -MaxUnalignedRHS 15" % (args.files['fFile'],args.files['javaTree'],args.files['aFile'])
    java_opts="-Xmx%s -Xms%s -cp %s/ghkm-revised.jar:%s/lib/fastutil.jar"%(mem,mem,args.ghkmDir,args.ghkmDir)
    os.system("java %s edu.stanford.nlp.mt.syntax.ghkm.RuleExtractor %s > %s" % (java_opts,ghkm_opts,args.files['ghkmFile']))

def runCPPGHKM(args):
    #/home/kmh/Files/Tools/ghkm/xrs-extract/bin/extract -r asdfg.train.en -U 0 -O -x asdfg.train.cppr
    os.system("%s -r %s -U %s -O -x %s -l %s"%(args.cppghkmLoc,args.files['inRoot'],args.cppghkm_U,args.files['ghkmFile'],args.cppghkm_l))
    #os.system("%s -r %s -U %s -O -x %s -m %s"%(args.cppghkmLoc,args.files['inRoot'],args.cppghkm_U,args.files['ghkmFile'],args.cppghkm_m))

def tibString2AMR(args,lang,sentence,index,filtered=False):
    if filtered:    rulesFile = "%s.%i"%(args.files['xrss2tGrammar'][lang],index)
    else:           rulesFile = args.files['tibTrained'][lang]
    outTreeFile = "%s/%s.%s.%d.trees"%(args.tmpDir,args.experiment,args.function,index)

    amr = False
    i = 1
    numbers = [0,1,10,100,1000,2000]
    #try:
    while amr == False and i < len(numbers):
        print "Running Tiburon inverted (%d best)"%numbers[i]
        print "S2A (txt): %s" % sentence

        #cmd = 'echo """%s""" | %s %s - -k %d -o %s'%(sentence.replace('`','\`'),args.tiburonLoc,rulesFile,numbers[i],outTreeFile)
        #proc = subprocess.call(cmd,shell=True)

        sentenceFile = "%s.%d"%(args.files['s2a_tmp'],index)
        f = open(sentenceFile,'w')
        f.write("%s\n"%sentence)
        f.close()
        cmds = '/usr/bin/java -Xmx4g -jar %s.jar %s %s -k %d -o %s'%(args.tiburonLoc,rulesFile,sentenceFile,numbers[i],outTreeFile)
        cmd = cmds.split()
        print cmd
        timeoutproc = Command(cmd)
        success = timeoutproc.run(timeout=900)
        if success is False:
            print "Tiburon did not terminate in time"
            return False,False
        print "Converting Tree back to AMR"
        (amr,tree) = getTopAMR(outTreeFile,numbers[i-1])
        if amr == -1:
            print "Could not find an AMR"
            return False,False
        i += 1
    if not amr:
        print "Could not recover a well-formed AMR"
        return False,False
    #except:
    #    print "Tiburon did not complete -- error"
    #    return False,False
    print "S2A (amr): %s" % amr.to_string()
    print "S2A (tre): %s" % tree
    return amr,tree

def tibTree2String(args,lang,tree,index,filtered=False):
    #cat mnew/tree4.file | ./tiburon - ../stanford-ghkm-2010-03-08/out.tib -k 5
    if filtered:    rulesFile = "%s.%i"%(args.files['xrst2sGrammar'][lang],index)
    else:           rulesFile = args.files['tibTrained'][lang]

    cmd = [args.tiburonLoc,'-',rulesFile,'-k %d'%args.tib_sentenceK]
    proc = subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    output = proc.communicate("%s\n"%tree)[0].split('\n')
    clean = defaultdict(float)
    if proc.returncode == 0:
        for sentence in output:
            try:
                sentence = tibFormat(sentence,reverse=True)
                x,y = sentence.split('#')
                clean[x] += float(y)
            except:
                pass
    return clean

def srilmMake(args):

    textFile = "%s/%s.%s.txt"%(args.dataBase,args.dataset,args.lang[0])

    cmd = [args.srilmLoc,'-order','3','-unk','-text',textFile,'-tolower','-lm',args.files['lmModel'][args.lang[0]],'-write-vocab',args.files['lmVocab'][args.lang[0]]] #,'-write-binary-lm'] KenLM Python doesn't support binarised models
    proc = subprocess.call(cmd)

def srilmLoad(args,lang):
    lm = LanguageModel(args.files['lmModel'][lang])
    return lm

def srilmTop(args,lm,data):
    if len(data) == 0: return ""
    
    attempts = defaultdict(float)
    for s,p in data.iteritems():
        attempts[s] += p
    for s,p in attempts.iteritems():
        lmprob = lm.score(s.lower())
        #print s,math.log(p),lmprob
        attempts[s] = (1.0 - args.lmweight) * math.log(p) + args.lmweight * lmprob
        #print s,p
    return max(attempts.iteritems(), key=operator.itemgetter(1))[0]


def tibString2AMRBatch(args,lang,sentences,index):
    '''
    Same as above, but returns an array of AMRs / falses
    Cannot handle filtered results (obviously)
    '''
    rulesFile = args.files['tibTrained'][lang]
    outTmpFile = "%s/%s.s2a.batch.%d.out"%(args.tmpDir,args.experiment,index)
    inTmpFile  = "%s/%s.s2a.batch.%d.in"%(args.tmpDir,args.experiment,index)

    print "Dropping sentences in tmp file"
    f = open(inTmpFile,'w')
    for s in sentences: f.write("%s\n"%s.strip())
    f.close()

    print "Running Tiburon in batch mode"
    os.system('%s %s %s -k %d -o %s'%(args.tiburonLoc,rulesFile,inTmpFile,50,outTmpFile))
    print "Done"

    amrList = []
    for i in range(len(sentences)):
        print "Converting Tree %d/%d back to AMR"%(i,len(sentences))
        (amr,tree) = getTopAMR("%s.%d"%(outTmpFile,i),0)
        if amr == -1:
            amrList.append(False)
            print "Could not parse to AMR"
        else:
            if amr == False:
                print "All AMRs broken"
            else:
                print "Found an AMR"
            amrList.append(tree)
    return amrList

def tibTree2StringBatch(args,lang,trees,index):
    #cat mnew/tree4.file | ./tiburon - ../stanford-ghkm-2010-03-08/out.tib -k 5

    rulesFile = args.files['tibTrained'][lang]
    outTmpFile = "%s/%s.t2s.batch.%d.out"%(args.tmpDir,args.experiment,index)
    inTmpFile  = "%s/%s.t2s.batch.%d.in"%(args.tmpDir,args.experiment,index)

    print "Dropping trees in tmp file"
    f = open(inTmpFile,'w')
    for t in trees: 
        if t:   f.write("%s\n"%t.strip())
        else:   f.write("EMPTY()\n")
    f.close()

    print "Running Tiburon in batch mode"
    os.system('%s %s %s -k %d -o %s'%(args.tiburonLoc,inTmpFile,rulesFile,args.tib_sentenceK,outTmpFile))

    sentences = []
    for i in range(len(trees)):
        clean = defaultdict(float)
        f = open("%s.%d"%(outTmpFile,i),'r')
        for line in f:
            try:
                sentence = tibFormat(line,reverse=True)
                x,y = sentence.split('#')
                clean[x] += float(y)
            except:
                pass
        sentences.append(clean)
    return sentences

def runBleuBatch(args,gold,hypotheses,orig=False):

    print "Bleu Batch"
    fFile = "%s/%s.%s.gold"%(args.exDir,args.experiment,args.function)
    gFile = "%s/%s.%s.hypo"%(args.exDir,args.experiment,args.function)
    f = open(fFile,'w')
    g = open(gFile,'w')
    print "Writing to Gold/Hypo files"
    for i in range(len(gold)):
        f.write("%s\n"%gold[i].strip())
        g.write("%s\n"%hypotheses[i].strip())
        print "GOLD: ", gold[i].strip()
        print "HYP:  ", hypotheses[i].strip()
        if orig: print "FROM: ", orig[i].strip()
        print ""
    f.close()
    g.close()
    print "Done writing"
    xcmd = "perl %s %s < %s"%(args.bleuScorerLoc,fFile,gFile)
    print "Perl"
    proc = subprocess.Popen(xcmd,shell=True,stdout=subprocess.PIPE)
    result = proc.communicate()[0]
    print "Got a result"
    m = result.split()
    bleu = {}
    #BLEU = 62.80, 100.0/85.7/69.2/58.3 (BP=0.819, ratio=0.833, hyp_len=15, ref_len=18)
    try:
        bleu['score'] = float(m[2][:-1])
    except:
        bleu['score'] = "0.0 (sorry)"
    print result
    return bleu


def runBleu(args,gold,hypothesis):
    #perl multi-bleu.perl <(echo "This is me in a sailing chair .") < <(echo "This is me in a rocking chair .")
    ycmd = "perl %s <(echo \"\"\"%s\"\"\") < <(echo \"\"\"%s\"\"\")"%(args.bleuScorerLoc,gold,hypothesis)
    proc = subprocess.Popen(ycmd,shell=True,stdout=subprocess.PIPE,executable="/bin/bash")
    result = proc.communicate()[0]
    m = result.split()
    bleu = {}
    #BLEU = 62.80, 100.0/85.7/69.2/58.3 (BP=0.819, ratio=0.833, hyp_len=15, ref_len=18)
    bleu['score'] = float(m[2][:-1])
    print bleu
    return bleu

def runSmatch(args,gold_amr,auto_amr):
    '''
    Some pre-processing before handing over to Daniel's Smatch tool
    Either we have 
    :ARG0:concept edges
    or :concept edges without anything before - in this case, there is some manually added stuff: :ROOT and :pseudoarg
    need to remove these things
    '''

    if type(auto_amr) == Amr or type(auto_amr) == Dag:

        if args.dataset == "geoquery2" or args.dataset == "geoquery3": # No edges whatsoever
            z = auto_amr.clone()
            for par, rel, child in auto_amr.triples(instances = False):
                crel = tibFormat(rel,reverse=True)
                x = crel.split(":",1)
                if len(x) == 2 and (x[0] == 'ROOT' or x[0] == 'pseudoarg'):
                    z._replace_triple(par,rel,child, par, x[1].lower(), child)
                else:
                    z._replace_triple(par,rel,child, par, crel.lower(), child)
            test_amr = z
            amr_gold = gold_amr.clone()
            for par, rel, child in gold_amr.triples(instances = False):
                amr_gold._replace_triple(par,rel,child, par, rel.lower(), child)
        else:
            test_amr = auto_amr.clone()
            amr_gold = gold_amr.clone()


        print amr_gold.to_string(newline=False)
        print test_amr.to_string(newline=False)
        try:
            print "Running Smatch ..."
            p,r,f = smatch(amr_gold,test_amr,method=get_random_start)
            print "Success"
            print "P:%f R:%f F:%f " % (p, r, f)
            return p,r,f
        except:
            pass
    p,r,f = (0.0,0.0,0.0)
    print "P:%f R:%f F:%f " % (p, r, f)
    return p,r,f

def tibEMTraining(args):
    #~~/tiburon minitree minis mytrans -t 5 -o newtrans

    if not args.applyNPList:    args.files['tibTmp'] = args.files['tibTrained'][args.lang[0]]
    print "EM Training"
    #print '%s %s %s -o %s -t %d --prior 0.1'%(args.tiburonLoc,args.files['tfFile'],args.files['tibRaw'],args.files['tibTmp'],args.iterations)
    os.system('%s %s %s -o %s -t %d --prior 0.1'%(args.tiburonLoc,args.files['tfFile'],args.files['tibRaw'],args.files['tibTmp'],args.iterations))

    if args.applyNPList:    applyNP(args.files['wordlist'][args.lang[0]],args.files['tibTmp'],args.files['tibTrained'][args.lang[0]])


def applyNP(wordFile,inFile,outFile):

    wordlist = open(wordFile)
    codes = ['city','state','country','place','river','number']
    codes += ['cityid_city','stateid','cityid_state','countryid','placeid','riverid','numberid']
    groups = {}
    for code in codes:  groups[code] = []
    
    for line in wordlist:
        tree,word,code = line.strip().split('\t')
        groups[code].append((tree,word))

    inf = open(inFile,'r')
    ouf = open("%s.nps"%inFile,'w')

    for line in inf:
        for group in groups:
            for key,val in groups[group]:
                if key in line and val in line:
                    for k,v in groups[group]:
                        lx = line.replace(key,k).replace(val,v)
                        ouf.write("%s\n"%lx.strip())
    inf.close()
    ouf.close()

    orig = open(inFile,'r')
    newf = open("%s.nps"%inFile,'r')
    merg = open(outFile,'w')

    alllines = {}
    for line in orig:
        merg.write(line)
        alllines[line.split('#')[0]] = True
    for line in newf:
        if not line.split('#')[0] in alllines:
            merg.write(line)
            alllines[line.split('#')[0]] = True

    orig.close()
    newf.close()
    merg.close()
