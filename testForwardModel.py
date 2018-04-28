from ForwardModel import ForwardHMM
from genModel import SegmentGM
import numpy as np

alphabet = ['a','b']
sizeAlph = len(alphabet)

# Generative model
stop = 0.1
units = {'a': 0.15, 'ab': 0.7,'b': 0.15}
gm = SegmentGM(units, stop)

# Generate corpus
size = 250
corpus = gm.gen(size)

# Forward model: random initialization
n = 3
fm = ForwardHMM.randHMM(alphabet, n)

def run_training():
	# Forward model: random initialization
	n = 3
	fm = ForwardHMM.randHMM(alphabet, n)

	# Learning phase
	fm.baumwelch(corpus)
	fm.aff()

	print("Final log-likelihood",fm.logL(corpus))

	predictUnits = fm.listSegment(trim=0.99)
	print("# Predicted Segments #")
	for w,p in predictUnits:
		print("\t- {} : {}".format(w,np.round(p,3)))

