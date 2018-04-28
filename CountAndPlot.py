import numpy as np
import matplotlib.pylab as plt
import tools

# SCRIPT CONSTANTS

#main input file
nMainFile="SequencesBrutes.txt"
#corrections in the second row
nCorrect="Corrections.txt"

#COMPILING SEQUENCES
corpus,alphabet=tools.compile(nMainFile,nCorrect)
sizeCorpus=len(corpus)
aR=dict()
for i in range(len(alphabet)):
	aR[alphabet[i]]=i

#COUNTING
oldAutInc=tools.appAutNInc(aR,1000,1000)
oldAutRP=tools.appRPStar(aR,1000,1000)
autInc=tools.appAutNInc(aR)
autRP=tools.appRPStar(aR)

tau=np.linspace(0.0,0.2,20)
cRP=[tools.countPatterns(corpus,autRP,x) for x in tau]
cInc=[tools.countPatterns(corpus,autInc,x) for x in tau]

oldTau=np.linspace(0.0,0.5,10)
oldCInc=[tools.countPatterns(corpus,oldAutInc,x) for x in oldTau]
oldCRP=[tools.countPatterns(corpus,oldAutRP,x) for x in oldTau]

#DISPLAYING EXAMPLES

# tauTest=[(0.1,),]




#PLOTTING

plt.figure(0)

plt1=plt.subplot(1, 2, 1)
plt.plot(tau,cRP)
plt.plot(tau,cInc)

plt.title("Number of sequences from a pattern/choice of tau")
plt.xlabel('$\\tau$')

plt.ylabel('Number of sequences')
plt.legend(['(rp*)*','incr.'])

plt2=plt.subplot(1, 2, 2,sharey=plt1)
plt.plot(oldTau,oldCRP)
plt.plot(oldTau,oldCInc)

plt.title("Number of sequences from a pattern/choice of tau")
plt.xlabel('$\\tau$')

plt.ylabel('Number of sequences')
plt.legend(['(rp*)*','incr.'])



plt.show()
plt.close()
