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
nSave="ComputeBICMuriquiCorpusRP.txt"
# maximum order of a Markov chain
nOrder=25
# number of replications
nRep=10
# Taus
paramTau=[0.1,0.2,0.3]


# COMPILING FILES

corpusOrig,alphabetOrig=tools.compile(nMainFile,nCorrect)
aR=dict()
for i,letter in enumerate(alphabetOrig):
	aR[letter]=i
autoRP=tools.appRPStar(aR)

hmFin=[]
n=nOrder
logLik=np.full((len(paramTau),n,nRep),0.0,dtype='float')
BIC=np.full((len(paramTau),n,nRep),0.0,dtype='float')

for t,tau in enumerate(paramTau):
	corpus,alphabet=tools.selectSome(corpusOrig,autoRP,tau)
		
	# FITTING MARKOV CHAIN

	hm=[[bw.HMM.randHMM(alphabet,j) for i in range(nRep)] for j in range(1,nOrder+1)]


	# Measuring computation time
	timestamp_init=time.time()
	for i in range(n):
		for j in range(nRep):
			print("#Start BW:%f#"%(time.time()-timestamp_init))
			hm[i][j].baumwelch(corpus)
			print("#End BW#")

	hmFin.append(hm)
	# dumping the results
	with open(nSave,'wb') as f:
		pickle.dump(hmFin,f)


	# STATISTICS OVER FINAL RESULTS
	# Computing all log-likelihoods
	for i in range(n):
		for j in range(nRep):
			logLik[t][i][j]=hmFin[t][i][j].logL(corpus)
	# Computing BIC score
	
	for i in range(n):
		for j in range(nRep):
			BIC[t][i][j]=hmFin[t][i][j].bicScore(corpus)

bestOrder=np.full(len(paramTau),0.0,dtype='int')
for i,tau in enumerate(paramTau):
	print("TAU=",tau)

	# Outputting best log-likelihood
	for j in range(n):
		print("Best log-likelihood for order n°%i: %d"%(hmFin[i][j][0].n,np.max(logLik[i][j])))
		
	# Outputting range interval of log-likelihood: if big, it suggests that the BW algorithm has sometimes converged to a local maximum significantly below the global maximum
	for j in range(n):
		print("Range of log-likelihoods for order n°%i: %d"%(hmFin[i][j][0].n,np.max(logLik[i][j])-np.min(logLik[i][j])))
	
	# Retrieving index that achieves best order
	bestI=np.argmax(np.max(BIC[i],axis=1))
	bestOrder[i]=hmFin[i][bestI][0].n

	# Outputting optimal Markov chain order according to BIC criterion
	print("Best order: %i"%bestOrder[i])



