import numpy as np
import time
import pickle
import MarkovModel as bw
import tools

# SCRIPT CONSTANTS

#main input file
nMainFile="SequencesBrutes.txt"
#corrections in the second row
nCorrect="Corrections.txt"
#save file for final Markov chains
nSave="ComputeBICMuriqui.txt"
# maximum order of a Markov chain
nOrder=25
# number of replications
nRep=10


# COMPILING FILES

corpus,alphabet=tools.compile(nMainFile,nCorrect)
	
# FITTING MARKOV CHAIN

hm=[[bw.HMM.randHMM(alphabet,j) for i in range(nRep)] for j in range(1,nOrder+1)]
n=len(hm)

# Measuring computation time
timestamp_init=time.time()
for i in range(n):
	for j in range(nRep):
		print("#Start BW:%f#"%(time.time()-timestamp_init))
		hm[i][j].baumwelch(corpus)
		print("#End BW#")

# dumping the results
with open(nSave,'wb') as f:
	pickle.dump(hm,f)


# STATISTICS OVER FINAL RESULTS
# Computing all log-likelihoods
logLik=np.full((n,nRep),0.0,dtype='float')
for i in range(n):
	for j in range(nRep):
		logLik[i][j]=hm[i][j].logL(corpus)
		
# Outputting best log-likelihood
for i in range(n):
	print("Best log-likelihood for order n°%i: %d"%(hm[i][0].n,np.max(logLik[i])))
	
# Outputting range interval of log-likelihood: if big, it suggests that the BW algorithm has sometimes converged to a local maximum significantly below the global maximum
for i in range(n):
	print("Range of log-likelihoods for order n°%i: %d"%(hm[i][0].n,np.max(logLik[i])-np.min(logLik[i])))
	
# Computing best BIC score
BIC=np.full((n,nRep),0.0,dtype='float')
for i in range(n):
	for j in range(nRep):
		BIC[i][j]=hm[i][j].bicScore(corpus)
bestOrder=np.argmax(np.max(BIC,axis=1))

# Outputting optimal Markov chain order according to BIC criterion
print("Best order: %i"%hm[bestOrder][0].n)



