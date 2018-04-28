from MarkovModel import HMM
import numpy as np

alphabet = ['a','b']
sizeAlph = len(alphabet)

# HMM 1
# - generates alternations of large 'a' sequences and smaller 'a' sequences separated by single b's
def HMM1():
	
	nStates = 2

	init = np.full(nStates,0.0)
	init[0]=1.0

	trans = np.full((nStates,sizeAlph+1,nStates),0.0)
	end = np.full(nStates,0.0)

	# From initial state
	trans[0][0][0]=0.9
	trans[0][1][1]=0.05
	end[0]=0.05

	# From other state
	trans[1][0][1]=0.6
	trans[1][1][0]=0.35
	end[1]=0.05

	return HMM(alphabet,init,end,trans)