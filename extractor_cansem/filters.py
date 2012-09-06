import subprocess

def createDatabases(args):
    '''
    We're done with training, let's create a rule database
    '''
    print "Creating XRS Databases"
    
    #echo "$PWD/sstring_xrsdb.sh small_grammar.raw sstring.xrsdb" | qsub -lwalltime=24:00:00,nodes=10 -qisi
    cmd = 'echo "%s/sstring_xrsdb.sh %s %s" | qsub -lwalltime=24:00:00,nodes=10 -qisi'%(args.xrsLoc,args.files['tibTrained'][args.lang[0]],args.files['xrsDBs2t'][args.lang[0]])
    print cmd
    proc1 = subprocess.call(cmd,shell=True)

    #echo "$PWD/sstring_xrsdb.sh small_grammar.raw sstring.xrsdb" | qsub -lwalltime=24:00:00,nodes=10 -qisi
    cmd = 'echo "%s/stree_xrsdb.sh %s %s" | qsub -lwalltime=24:00:00,nodes=10 -qisi'%(args.xrsLoc,args.files['tibTrained'][args.lang[0]],args.files['xrsDBt2s'][args.lang[0]])
    print cmd
    proc2 = subprocess.call(cmd,shell=True)

    #cmd = ["%s/stree_xrsdb.sh"%args.xrsLoc,args.files['tibTrained'][args.lang[0]],args.files['xrsDBt2s']]

    def hasNoStrings(sentence):
        try:
            x,y = sentence.split('->')
            rhs,x = y.split('#')
            words = rhs.split()
            for word in words:
                if "." not in word: return False
            return True
        except:
            return False
    
    f = open(args.files['tibTrained'][args.lang[0]],'r')
    t = open(args.files['xrsNoStr'][args.lang[0]],'w')

    f.next() # skip root
    for line in f:
        if hasNoStrings(line):
            t.write(line)

    return True


def queryString2Tree(args,data,lang):
    '''
    Create a S2T subset of the rules
    
    pipeline-0.9/bin/sent_to_lattice.pl < training.raw \
    | pipeline-0.9/bin/xrsdb_batch_retrieval -d sstring.xrsdb -i- -u decode -p sstring.out/rules. -s .gz
    '''
    f = open(args.files['xrsFilterStrings'],'w')
    for sentence in data.sentences:
        f.write("%s\n"%sentence['plain_text'])
    f.close()
    
    cmd = "%s/pipeline-0.9/bin/sent_to_lattice.pl < %s | %s/pipeline-0.9/bin/xrsdb_batch_retrieval -d %s -i- -u decode -p %s."%(args.xrsLoc,args.files['xrsFilterStrings'],args.xrsLoc,args.files['xrsDBs2t'][lang],args.files['xrss2tTmp'][lang])
    print cmd
    proc2 = subprocess.call(cmd,shell=True)

    for i in range(len(data.sentences)):
        f = open("%s.%d"%(args.files['xrss2tGrammar'][lang],i),'w')
        f.write("root\n")
        g = open(args.files['xrsNoStr'][lang],'r')
        for line in g:
            if not line.strip() == "root":
                f.write(line)
        g.close()
        h = open("%s.%d"%(args.files['xrss2tTmp'][lang],i+1),'r')
        for line in h:
            if not line.strip() == "root":
                f.write(line)
        h.close()
        f.close()

def queryTree2String(args,trees,lang):
    '''
    Create a T2S subset of the rules
    pipeline-0.9/bin/stree2lat < training.raw \
    | pipeline-0.9/bin/xrsdb_batch_retrieval -d stree.xrsdb -i- -u raw_sig -p 'stree.out/gram.' -s '.gz'
    '''

    f = open(args.files['xrsFilterTrees'],'w')
    for tree in trees:
        f.write("%s\n"%tree.strip())
    f.close()
    
    cmd = "%s/pipeline-0.9/bin/stree2lat < %s | %s/pipeline-0.9/bin/xrsdb_batch_retrieval -d %s -i- -u raw_sig -p %s."%(args.xrsLoc,args.files['xrsFilterTrees'],args.xrsLoc,args.files['xrsDBt2s'][lang],args.files['xrst2sTmp'][lang])
    print cmd
    proc2 = subprocess.call(cmd,shell=True)

    for i in range(len(trees)):
        f = open("%s.%d"%(args.files['xrst2sGrammar'][lang],i),'w')
        f.write("root\n")
        h = open("%s.%d"%(args.files['xrst2sTmp'][lang],i+1),'r')
        for line in h:
            if not line.strip() == "root":
                f.write(line)
        h.close()
        f.close()

