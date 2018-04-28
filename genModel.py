import numpy as np
# Segment generative model: implements a PCFG of the form
# S -> US
# S -> epsilon
# U -> terminal symbols

class SegmentGM:

	# Args:
	# - d: dictionary that maps segments to the probability of their apparition
	# - stop: probability of S -> epsilon
	def __init__(self,m_d,m_stop):
		self.d = m_d
		self.freqTable = list(self.d.items())
		self.segments = [s for s,p in self.freqTable]
		self.proba = np.array([p for s,p in self.freqTable])
		self.stop = m_stop

	def alphabet(self):
		return list(set().union(*tuple(set(w) for w in self.segments)))



	def gen(self,n=-1):
		if n<0:
			word = ""
			while True:
				if np.random.binomial(1,self.stop):
					return word
				else:
					word += np.random.choice(self.segments, p = self.proba)
		else:
			return [self.gen() for _ in range(n)]
