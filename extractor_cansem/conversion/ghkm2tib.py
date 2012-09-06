import sys
import re
import math
from collections import defaultdict
from extractor_cansem.data_structures import galFormat,tibFormat

def ghkm2tib(inFile,outFile,inFormat="Java"):

    inf = open(inFile,'r')
    ouf = open(outFile,'w')
    ouf.write("root\n") ### not needed if filtering!
    '''
    FROM:
    ROOT@SL(ROOT@SL(x0:ROOT@DL x1:ARG1@DL) ARG0@SO(x2:ARG0@DL x3:ARG1@DL)) -> x0 "that" "the" x2 x1 x3 ||| 0.0074074073 0.01
    
    TO:
    q.ARG1_SL( ARG1_DL( z0:want-01 ) ARG0_SW( x1:ARG0_DL x2:ARG1_DL ) ) -> that prp.x1 wants prp.x2 

    I.e. add q. in front of things left and right.
    Encode as follows: 
    ROOT@SL@CODE -> ROOT_SL
    ROOT@SL@CODE -> code.ROOT_SL iff first element!
    
    x0:root@DL@code -> x0:root@DL and code.x0 for RHS
    '''    

    if inFormat == "Java":
        inf.next()
    else:
        inf.next()
        inf.next()
        inf.next()


    code1_matcher = re.compile("x([0-9]+):([^ @]+)@([^ @]+)@(.+)") # x0:ROOT@DL@q
    code2_matcher = re.compile("([^ @]+)@([^ @]+)@(.+)") # ROOT@DL@q
    var_matcher = re.compile("x([0-9]+)")

    str_matcher = re.compile("\"(.+)\"")

    cppprob_matcher = re.compile("fraccount=(.*)") # fraccount=0.125

    rules = defaultdict(float)

    for line in inf:
        line = galFormat(line,reverse=True)
        xDict = {}
        newLine = ""
        currentBuf = ""
        probJoint = "0.1"

        for char in line:
            if char == "(" or char == ")" or char == " ":
                #process currentToken
                # 1: x@x@x
                #print currentBuf
                match = code1_matcher.match(currentBuf)
                if match:
                    num = match.group(1)
                    rule = match.group(2)
                    arg = match.group(3)
                    code = match.group(4)
                    xDict[num] = code
                    currentBuf = "x%s:%s_%s" % (num,rule,arg)
                else:
                    match = code2_matcher.match(currentBuf)
                    if match:
                        rule = match.group(1)
                        arg = match.group(2)
                        code = match.group(3)
                        if newLine == "": # start of line
                            currentBuf = "%s.%s_%s" % (code,rule,arg)
                        else:
                            currentBuf = "%s_%s" % (rule,arg)

                    else:
                        match = var_matcher.match(currentBuf)
                        if match:
                            num = match.group(1)
                            currentBuf = "%s.x%s" % (xDict[num],num)

                    
                        else:
                            match = str_matcher.match(currentBuf)
                            if match:
                                currentBuf = match.group(1)

                            elif inFormat == "C++":
                                match = cppprob_matcher.match(currentBuf)
                                if match:
                                    probJoint = match.group(1)

                newLine += currentBuf
                newLine += char
                currentBuf = ""


            else:
                currentBuf += char

        newLine += currentBuf

        if inFormat == "Java":
            (line,probs) = newLine.split("|||")
            (probJoint,probCond) = probs.split()
        else:
            (line,rest) = newLine.split("###")
        #probJoint = math.log(float(probJoint))

        rules[line] += float(probJoint.strip())

    for key,val in rules.iteritems():
        ouf.write("%s # %s\n"%(key.rstrip(),val))
        
    inf.close()
    ouf.close()

if __name__=='__main__':

    if len(sys.argv) < 1:
        print "ERROR, need a file name"
        sys.exit(-1)

    fName = sys.argv[1]
    g2t(fName)
