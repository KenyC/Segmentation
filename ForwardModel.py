from MarkovModel import HMM
import numpy as np

# A forward HMM is a Markovian process is an HMM with the following constraints
# - for all i,j>0, the probability of transitioning from i to j is zero if i >= j
# - state 0 is the only state that can exit
# - state 0 is start state

# Such a forward HMM can equivalently be represented with a PCFG of the form
# S -> US
# S -> epsilon
# U -> terminal symbols

class ForwardHMM(HMM):

	# This constructor is like the HMM, but it erases forbidden transitions
	def __init__(self,alphabet,m_end,trans):
		
		self.n = len(m_end)
		init = np.full(self.n, 0.0)
		init[0] = 1.0

		end = np.full_like(m_end,0.0)
		end[0] = m_end[0]

		trans = np.transpose(trans,(0,2,1))
		for i in range(1,self.n):
			for j in range(1,self.n):
				if i>=j:
					trans[i][j]*=0.0
		trans = np.transpose(trans,(0,2,1))

		super(ForwardHMM,self).__init__(alphabet,init,end,trans)

		self.normalize()

	def randHMM(alphabet, n):
		aux = HMM.randHMM(alphabet, n)
		print(aux.e)
		print(aux.t)
		return ForwardHMM(alphabet, aux.e, aux.t)

	# Segments a word
	# - word: word to be segmented
	# Returns a list of a sequence of strings corresponding to the words different segments
	def segment(self,word):
		_,path = self.viterbi(word)

		return self.auxSegment(path,word)

	# Given a path that generated a word, segment the word according to the path
	# Returns a list of a sequence of strings corresponding to the words different segments
	def auxSegment(self, path, word):
		path = path[1:]

		segments = []
		current = ""
		
		for idx, state in enumerate(path):
			current += word[idx]

			if state == 0:
				segments.append(current)
				current = ""

		return segments
	
	def genSegment(self,n=-1):
		if n<0:
			return self.auxSegment(*self.gen(path=True))
		else:
			return [self.auxSegment(path,word) for path,word in self.gen(n,path=True)]

	# Automatically discovers segments implicit in the representation of forward HMM
	# - sample (optional): number of samples taken for segmentation
	# - trim (optional): if positive, only keeps the elements that appear trim% of the time
	# Returns a list of pairs of segments and their empiric frequency
	def listSegment(self,sample = 100,trim = -1):
		corpus = self.gen(sample, path = True)

		d = dict()
		for path,word in corpus:
			l = self.auxSegment(path,word)
			for s in l:
				if s not in d:
					d[s]=1
				else:
					d[s]+=1
		
		freqTable = list(d.items())
		freqTable = sorted(freqTable, key = lambda x: x[1], reverse = True)
		sUnits = sum(f[1] for f in freqTable)
		freqTable = [(k,v/float(sUnits)) for (k,v) in freqTable]

		if trim > 0.0:
			
			i = 0
			freq = 0.0

			while freq<trim and i<len(freqTable):
				freq += freqTable[i][1]
				i+=1
			
			freqTable = freqTable[:i]

		return freqTable


