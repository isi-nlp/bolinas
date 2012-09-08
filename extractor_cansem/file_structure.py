import os
import errno

def fileNames(args,lang1,lang2):
    '''
    experiments                                                     <-- experimentRoot
    experiments/ontonotes-i50/                                      <-- exRoot
    experiments/ontonotes-i50/ontonotes-wsj-test-0-500/             <-- exDir
    experiments/ontonotes-i50/ontonotes-wsj-test-0-500/filters      <-- filterDir
    '''

    if not args.datasetRoot:    args.datasetRoot = args.dataset

    if args.cross > 0:  args.dataBase = "%s/%s/train/cv/%s%02d" % (args.dataRoot,args.datasetRoot,args.dataSub,args.cross-1)
    else:               args.dataBase = "%s/%s/%s" % (args.dataRoot,args.datasetRoot,args.dataSub)

    if args.toLine:
        thisTestDir = "%s-%s-%s%s-%d-%d" % (args.dataset,args.dataSub,lang1,lang2,args.fromLine,args.toLine)
    else:
        thisTestDir = "%s-%s-%s%s-all" % (args.dataset,args.dataSub,lang1,lang2)

    if args.experiment2 != args.experiment:
        thisTestDir += "_%s"%args.experiment2

    def saveMkdir(path):
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

    args.exRoot = "%s/%s" % (args.experimentRoot,args.experiment)
    saveMkdir(args.exRoot)
    args.exRoot2 = "%s/%s" % (args.experimentRoot,args.experiment2)

    args.exDir = "%s/%s/%s" % (args.experimentRoot,args.experiment,thisTestDir)
    saveMkdir(args.exDir)
    
    args.filterDir = "%s/%s/%s/filters" % (args.experimentRoot,args.experiment,thisTestDir)
    saveMkdir(args.filterDir)

    args.tmpDir = "%s/%s/%s/tmp" % (args.experimentRoot,args.experiment,thisTestDir)
    saveMkdir(args.tmpDir)
    
    if args.cross > 0:
        args.experiment = "%s_cv%02d"%(args.experiment,args.cross)
        args.experiment2 = "%s_cv%02d"%(args.experiment2,args.cross)
    
    files = {}

    # TRAINING

    # GHKM Rule Extraction Phase
    files['javaTree'] = "%s/%s.train.%s.ptb"%(args.exRoot,args.experiment,lang1)      # PTB Parses for Java GHKM
    files['cppTree'] = "%s/%s.train.%s.e-parse"%(args.exRoot,args.experiment,lang1)   # PTB Parses for CPP GHKM
    files['aFile'] = "%s/%s.train.%s.a"%(args.exRoot,args.experiment,lang1)           # Alignment file for GHKM
    files['fFile'] = "%s/%s.train.%s.f"%(args.exRoot,args.experiment,lang1)           # F-String file for GHKM

    files['ghkmFile'] = "%s/%s.train.%s.ghkm"%(args.exRoot,args.experiment,lang1)
    files['inRoot'] = "%s/%s.train.%s"%(args.exRoot,args.experiment,lang1)

    files['lmModel'] = {}
    files['lmModel'][lang1] = "%s/%s.train.%s.lm.lm"%(args.exRoot,args.experiment,lang1)           # SRILM Model
    files['lmModel'][lang2] = "%s/%s.train.%s.lm.lm"%(args.exRoot2,args.experiment2,lang2)           # SRILM Model
    files['lmVocab'] = {}
    files['lmVocab'][lang1] = "%s/%s.train.%s.lm.vb"%(args.exRoot,args.experiment,lang1)           # SRILM Vocab
    files['lmVocab'][lang2] = "%s/%s.train.%s.lm.vb"%(args.exRoot2,args.experiment2,lang2)           # SRILM Vocab

    # Tiburon EM Training
    files['tfFile'] = "%s/%s.train.%s.tree-f"%(args.exRoot,args.experiment,lang1)     # Tib/F for Tiburon EM training
    files['tibRaw'] = "%s/%s.train.%s.tib.raw"%(args.exRoot,args.experiment,lang1)
    files['tibTmp'] = "%s/%s.train.%s.tib.tmp"%(args.exRoot,args.experiment,lang1)
    files['tibTrained'] = {}
    files['tibTrained'][lang1] = "%s/%s.train.%s.tib.trained"%(args.exRoot,args.experiment,lang1)
    files['tibTrained'][lang2] = "%s/%s.train.%s.tib.trained"%(args.exRoot2,args.experiment2,lang2)

    # Filtering and XRSDB
    files['xrsDBs2t'] = {}
    files['xrsDBs2t'][lang1] = "%s/%s.%s.s2t.xrsdb"%(args.exRoot,args.experiment,lang1)
    files['xrsDBs2t'][lang2] = "%s/%s.%s.s2t.xrsdb"%(args.exRoot2,args.experiment2,lang2)
    files['xrsDBt2s'] = {}
    files['xrsDBt2s'][lang1] = "%s/%s.%s.t2s.xrsdb"%(args.exRoot,args.experiment,lang1)
    files['xrsDBt2s'][lang2] = "%s/%s.%s.t2s.xrsdb"%(args.exRoot2,args.experiment2,lang2)
    files['xrsNoStr'] = {}
    files['xrsNoStr'][lang1] = "%s/%s.%s.nostring"%(args.exRoot,args.experiment,lang1)
    files['xrsNoStr'][lang2] = "%s/%s.%s.nostring"%(args.exRoot2,args.experiment2,lang2)

    # NP Wordlist
    files['wordlist'] = {}
    files['wordlist'][lang1] = "%s/%s/extra/wordlist.%s.txt"%(args.dataRoot,args.datasetRoot,lang1)
    files['wordlist'][lang2] = "%s/%s/extra/wordlist.%s.txt"%(args.dataRoot,args.datasetRoot,lang2)

    # EVALUATION

    # Rule Filtering
    files['xrsFilterStrings'] = "%s/%s.%s%s.filter.txt"%(args.filterDir,args.experiment,lang1,lang2) # Sentences for filtering
    files['xrsFilterTrees'] = "%s/%s.%s%s.filter.tree"%(args.filterDir,args.experiment,lang1,lang2)  # Trees for filtering
    # temporary and final filtered grammar files
    files['xrst2sTmp'] = {}
    files['xrst2sTmp'][lang1] = "%s/%s.%s.%s.t2s.grammar.tmp"%(args.filterDir,args.experiment,lang1,args.function)
    files['xrst2sTmp'][lang2] = "%s/%s.%s.%s.t2s.grammar.tmp"%(args.filterDir,args.experiment,lang2,args.function)
    files['xrss2tTmp'] = {}
    files['xrss2tTmp'][lang1] = "%s/%s.%s.%s.s2t.grammar.tmp"%(args.filterDir,args.experiment,lang1,args.function)
    files['xrss2tTmp'][lang2] = "%s/%s.%s.%s.s2t.grammar.tmp"%(args.filterDir,args.experiment,lang2,args.function)
    files['xrst2sGrammar'] = {}
    files['xrst2sGrammar'][lang1] = "%s/%s.%s.%s.t2s.grammar.final"%(args.filterDir,args.experiment,lang1,args.function)
    files['xrst2sGrammar'][lang2] = "%s/%s.%s.%s.t2s.grammar.final"%(args.filterDir,args.experiment,lang2,args.function)
    files['xrss2tGrammar'] = {}
    files['xrss2tGrammar'][lang1] = "%s/%s.%s.%s.s2t.grammar.final"%(args.filterDir,args.experiment,lang1,args.function)    
    files['xrss2tGrammar'][lang2] = "%s/%s.%s.%s.s2t.grammar.final"%(args.filterDir,args.experiment,lang2,args.function)    

    # String 2 AMR
    files['s2a_amr'] = "%s/%s.s2a.%s.amr"%(args.exDir,args.experiment,lang1)         # Resultant AMRs
    files['s2a_deriv'] = "%s/%s.s2a.%s.deriv"%(args.exDir,args.experiment,lang1)     # Resultant Trees

    files['s2a_tmp'] = "%s/%s.s2a.%s.%s.tmp"%(args.exDir,args.experiment,lang1,args.function) #Temporary file for Tiburon (indexed)
    files['s2a_t_amr'] = "%s/%s.s2a.t.%s%s.amr"%(args.exDir,args.experiment,lang1,lang2)         # Resultant AMRs
    #files['a2s_tmp'] = "%s/%s.a2s.%s.%s.tmp"%(args.exDir,args.experiment,lang1,args.function) # Temporary file for Tiburon (indexed)

    # additionally:
    #if filtered:    rulesFile = "%s/%s_s2t_gram.%d.out"%(args.filterDir,args.experiment,index)
    #else:           rulesFile = "%s/%s.train.%s.tib.trained"%(args.exRoot,args.experiment,lang)
    #outTreeFile = "%s/%s.%s.%d.trees"%(args.tmpDir,args.experiment,args.function,index)

    # AMR 2 String
    files['a2s_txt'] = "%s/%s.a2s.%s.txt"%(args.exDir,args.experiment,lang1)

    #if args.function == "s2s":      rulesFile = "%s/%s_gen2str_gram.%d.out"%(args.filterDir,args.experiment,index)
    #elif args.function == "a2s":    rulesFile = "%s/%s_gold2str_gram.%d.out"%(args.filterDir,args.experiment,index)
    #else:                       rulesFile = "%s/%s.train.%s.tib.trained"%(args.exRoot,args.experiment,lang)

    # BATCH FILES
    #outTmpFile = "%s/%s.s2a.batch.%d.out"%(args.tmpDir,args.experiment,index)
    #inTmpFile  = "%s/%s.s2a.batch.%d.in"%(args.tmpDir,args.experiment,index)

    args.files = files

    return args



