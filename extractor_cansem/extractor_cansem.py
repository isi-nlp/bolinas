import argparse
import sys
import os

from annotated_set import loadData
from data_structures import CanonicalDerivation
from canonical_parser import CanonicalParser
from derivation_tree import DerivationTree

from conversion.ghkm2tib import ghkm2tib
#from lib.amr.dag import Dag

class ExtractorCanSem:

    def __init__(self):
        pass

    @classmethod
    def help(self):
        """
        Returns CanSem help message.
        """
        return ExtractorCanSem.main(ExtractorCanSem(),"--help")

    def main(self, *args):
        parser = argparse.ArgumentParser(description='CanSem Extraction Algorithm for SHRG',
                                         fromfile_prefix_chars='@',
                                         prog='%s extract-cansem'%sys.argv[0])

        parser.add_argument('nl_file', type=str, help="Natural Language File")
        parser.add_argument('mr_file', type=str, help="Meaning Representation File")
        parser.add_argument('alignment_file', type=str, help="Alignment File")
        #parser.add_argument('--tib_sentenceK', type=int, help='-k value for Tiburon Tree->Str', default="50")
        parser.add_argument('--ghkmDir', nargs='?', default='/home/kmh/Files/Tools/stanford-ghkm-2010-03-08', help="GHKM directory")
        parser.add_argument('--tiburonLoc', nargs='?', default='/home/kmh/Files/Tools/newtib/tiburon', help="Tiburon executable file")
        parser.add_argument('--suffix', nargs='?', default=False, help="Suffix for temporary and output files")

        args = parser.parse_args(args=args)

        if args.suffix == False:
            args.suffix = "test"
        args.parse_path = "%s.ptb"%args.suffix
        args.align_path = "%s.a"%args.suffix
        args.text_path = "%s.f"%args.suffix
        args.ghkm_path = "%s.ghkm"%args.suffix
        args.tib_path = "%s.tib"%args.suffix

        # load input data into AnnotatedSet

        data = loadData(args.nl_file,args.mr_file,args.alignment_file)

        derivations = []
        for sentence in data:
            # Extraction
            parser = CanonicalParser(sentence)
            if len(parser.derivs_done) > 0:
                derivations.append((sentence,parser.derivs_done[0]))

        print len(derivations)

        self.genGHKMfiles(args,derivations)

    def genGHKMfiles(self,args,derivations):

        parse_file = open(args.parse_path,'w')
        align_file = open(args.align_path,'w')
        text_file = open(args.text_path,'w')

        for s,d in derivations:
            x = DerivationTree.fromDerivation(d)
            parse,align = x.getGHKMtriple_Java()
            text = s["nl"].strip(' \t\n\r')
            parse_file.write("%s\n"%parse)
            align_file.write("%s\n"%align)
            text_file.write("%s\n"%text)

        parse_file.close()
        align_file.close()
        text_file.close()

        print "Running GHKM Java rule extraction"
        mem = "2g"
        ghkm_opts = "-fCorpus %s -eParsedCorpus %s -align %s -joshuaFormat false -maxLHS 200 -maxRHS 15 -MaxUnalignedRHS 15" % (args.text_path,args.parse_path,args.align_path)
        java_opts="-Xmx%s -Xms%s -cp %s/ghkm.jar:%s/lib/fastutil.jar -XX:+UseCompressedOops"%(mem,mem,args.ghkmDir,args.ghkmDir)
        os.system("java %s edu.stanford.nlp.mt.syntax.ghkm.RuleExtractor %s > %s" % (java_opts,ghkm_opts,args.ghkm_path))

        print "Converting GHKM rules to Tiburon format"
        ghkm2tib(args.ghkm_path,args.tib_path)
