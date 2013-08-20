Bolinas
=======

A toolkit for Synchronous Hyperedge Replacement Grammar.


Directory Structure
-------------------
aligner    - implementation of the DepDep alignment algorithm (Jones et al. 2012)
binarizer  - tool to binarize SHRG grammars
common     - data structures and utilities used across sub-modules
 > hgraph  - hypergraph representation and parser for hypergraph descriptions
doc        - additional documentation
examples   - example grammars and input files
extra      - additional tools to convert input graphs and grammars
extractor_cansem - CanSem rule extraction algorithm [1] 
extractor_synsem - SynSem rule extraction algorithm [1] 
lib         - libraries
LICENSE.txt - MIT license 
parser      - default HRG/String parser with explicit visit order on each SHRG rule
parser_td   - tree decomposition HRG parser [2] 


Getting Started
---------------

First, let's read in a Hyperedge Replacement Grammar and verify that it is 
syntactically correct:

$ ./bolinas examples/ambiguity.hrg
Loaded hypergraph grammar with 9 rules
$

Have a look at the grammar file examples/abiguity.hrg to understand the rule
format. Every rule has a LHS symbol, followed by "->" followed by a RHS 
hypergraph. The hypergraph format is documented in doc/hgraph_format.txt
Every rule is terminated with ";". A rule weight can follow the semicolon. 
Comments can appear at the end of a line following a "#". 

We can now parse our first graphs.

$ ./bolinas examples/ambiguity.hrg examples/ambiguity.graphs 
Loaded hypergraph grammar with 9 rules.
1(T$_1(9) U$_0(3(V$_0(5))))     #0.240000

2(U$_0(3(V$_0(5))))     #0.300000

This command reads the input grammar and apply it to the graphs in 
examples/ambiguity.graphs.
Bolinas prints the first best derivation structure for each graph, indicating
that this graph is indeed a member of the set of graphs described by the 
grammar.
Each line contains a derivation tree. For instance the first line means: 
-Replace the start symbol by rule 1
-Replace the nonterminal T with index 1 by rule 9
-Replace the nonterminal U with index 0 by rule 3
    - In the RHS graph fragment of rule 3 replace the nonterminal V 
      with index 0 by rule 5.

In fact, both graphs have multiple analyses and we can retrieve the k-best 

dbauer@dorian:~/bolinasdist/bolinas$ ./bolinas -k 3 examples/ambiguity.hrg examples/ambiguity.graphs 
Loaded hypergraph grammar with 9 rules.
1(T$_1(9) U$_0(3(V$_0(5))))     #0.240000
1(T$_1(9) U$_0(4(W$_0(6))))     #0.048000
1(T$_1(8) U$_0(4(W$_0(7))))     #0.028000

Found only 2 derivations.
2(U$_0(3(V$_0(5))))     #0.300000
2(U$_0(4(W$_0(6))))     #0.060000

We can also have bolinas return a derivation forest. In this case we specify 
a prefix for output files with -o. Bolinas will write a regular tree grammar encoding
the derivationf forest for each graph into a separate file. 

$ ./bolinas -ot forest -o out examples/ambiguity.hrg examples/ambiguity.graphs 
Loaded hypergraph grammar with 9 rules.
dbauer@dorian:~/bolinasdist/bolinas$ cat out_1.rtg 
_START
R1___0:_1,_0:x,_1:x,x -> 1(T$_1(R8___0:x,x) U$_0(R4___0:_1,_1:x))       #0.500000
R1___0:_1,_0:x,_1:x,x -> 1(T$_1(R9___0:x) U$_0(R3___0:_1,_1:x,x))       #0.500000
R1___0:_1,_0:x,_1:x,x -> 1(T$_1(R9___0:x) U$_0(R4___0:_1,_1:x,x))       #0.500000
R3___0:_1,_1:x,x -> 3(V$_0(R5___0:_1,_1:x))     #0.600000
R4___0:_1,_1:x -> 4(W$_0(R7___1:x))     #0.400000
R4___0:_1,_1:x,x -> 4(W$_0(R6___1:x,x)) #0.400000
R5___0:_1,_1:x -> 5     #1.000000
R6___1:x,x -> 6 #0.300000
R7___1:x -> 7   #0.700000
R8___0:x,x -> 8 #0.200000
R9___0:x -> 9   #0.800000

Finally the most common use case for Bolinas is graph translation. For instance, we
can translate hypergraphs into strings. To do this we specify that the output type
(-ot flag) should be a derived object. Bolinas will take each derivation from the 
set of k-best and apply the second right hand side of the SHRG to reconstruct the
string. 

$ ./bolinas -ot derived examples/basic_boygirl.shrg examples/basic_boygirl.graph 
Loaded hypergraph-to-string grammar with 7 rules.
the boy wants the girl to believe that he is wanted     #1.000000

the boy wants the girl to believe that he wants her     #1.000000

Finally, we may want to reverse this process and translate strings into hypergraphs.
We can use the same grammar, but use the "-r" flag to apply the SHRG right-to-left.

./bolinas -ot derived -r examples/basic_boygirl.shrg examples/basic_boygirl.string 
Loaded string-to-hypergraph grammar with 7 rules.
(x41. :arg0 (x40. :boy' ) :arg1 (x42. :arg0 (x11. :girl' ) :arg1 (x12. :arg1 x40. :want' ) :believe' ) :want' ) #1.000000

(x41. :arg0 (x40. :boy' ) :arg1 (x42. :arg0 (x22. :girl' ) :arg1 (x20. :arg0 x40. :arg1 x22. :want' ) :believe' ) :want' )      #1.000000

References
----------

[1] B. Jones*, J. Andreas*, D. Bauer*, K-M. Hermann*, K. Knight (2012): 
"Semantics-Based Machine Translation with Hyperedge Replacement Grammars", 
COLING. [*First authorship shared. Order on publication randomized.]

[2] D. Chiang, J. Andreas, D. Bauer, K-M. Hermann, B. Jones, K. Knight (2013):
"Parsing Graphs with Hyperedge Replacement Grammars", ACL.
