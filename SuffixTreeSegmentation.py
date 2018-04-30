# This implements segmentation using another method, namely suffix trees
# Specifically, the procedure is the following:
# - find the most valuable substring in the corpus (if tie, pick the shortest one)
# value is defined as (number of occurences)*(length)
from suffix_trees.STree import STree

# In a later version, I will dispense of any assumption about morpheme size
maxl = 20

def bestScore(sTree,f):
	

	def traverseFunction(node, pathLabel=""):
		score = 0.0
		winner = ""
		if not node.is_leaf():
			print(pathLabel)
			score, winner = max(
				(traverseFunction(daughter,pathLabel + sTree._edgeLabel(daughter,node)) for daughter, _ in node.transition_links),
				key = lambda x: x[0])
			score, winner = max(
				((len(pathLabel)-1)*(node.nOcc-1),pathLabel),
				(score,winner),
				key = lambda x: x[0])

		return score,winner

	return traverseFunction(sTree.root)[1]



