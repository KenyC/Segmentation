import numpy as np
import random as r
from automaton import Automaton
from munkres import Munkres
# augment takes each letter of word "w" and add them to the alphabet "alphabet" only if the alphabet does not contain the letter already
def augment(w,alphabet):
	l=list(w)
	for c in l:
		if not c in alphabet:
			alphabet.append(c)
	

			
#COMPILING THE FILES AND THE ALPHABET
# There are two files: one contains most of the muriqi sequences
# The second file contains some corrections to the first file. 
# We first take all the lines from the first file, then for each line from the second file, if not empty, we replace the corresponding line the original file by the line from the second file.
#Putting the main file in
def compile(nMainFile,nCorrect):
	with open(nMainFile) as f:
		lines = [line.strip() for line in f]
	#Adding the corrections
	with open(nCorrect) as f:
		correct = [line.strip() for line in f]
		for i in range(len(correct)):
			if correct[i]!='':
				lines[i]=correct[i]
	#Deleting empty strings
	lines=np.asarray([line for line in lines if line.strip()!=''])

	#Compiling the alphabet
	alph=['r','p']
	for w in lines:
		augment(w,alph)
	return lines,alph

#This function selects a subset of the corpus that fit a certain pattern defined by "auto" et "tau"
def selectSome(corpus,auto,tau):
	lines=np.asarray([word for word in corpus if auto.distToPat(word)<tau*len(word)])

	#Compiling the alphabet
	alph=['r','p']
	for w in lines:
		augment(w,alph)
	return lines,alph	

def customCost(alphabet,req):
	nAlph=len(alphabet)

	indexP=alphabet['p']
	indexR=alphabet['r']

	#Delete costs
	delet=np.full(nAlph,req['delOther'],dtype='float')
	delet[indexP]=req['delP']
	delet[indexR]=req['delR']
	
	#Insert costs
	inser=np.full(nAlph,req['insOther'],dtype='float')
	inser[indexP]=req['insP']
	inser[indexR]=req['insR']
	
	#Substitution costs
	subst=req['sOtherToOther']*(np.ones(nAlph,dtype='float')-np.identity(nAlph,dtype='float'))
	for i in range(nAlph):
		subst[i][indexP]=req['sOtherToP']
		subst[indexP][i]=req['sPToOther']
		subst[i][indexR]=req['sOtherToR']
		subst[indexR][i]=req['sRToOther']
	subst[indexP][indexP]=0
	subst[indexR][indexR]=0

	return {"insert":inser,"delete":delet,"subst":subst}

def standardCost(alphabet,coutP=1000,coutR=1000):
	nAlph=len(alphabet)

	indexP=alphabet['p']
	indexR=alphabet['r']
	indexH=alphabet['h']

	#Distance costs : 'h' and 'r' are the same / 'p' costs costP to substitute, insert or delete, same for 'r'

	#Delete costs
	delet=np.ones(nAlph,dtype='float')
	delet[indexP]=coutP
	delet[indexR]=coutR
	delet[indexH]=coutR

	#Insert costs
	inser=np.ones(nAlph,dtype='float')
	inser[indexP]=coutP
	inser[indexR]=coutR
	
	#Substitution costs
	subst=np.ones(nAlph,dtype='float')-np.identity(nAlph,dtype='float')
	for i in range(nAlph):
		subst[i][indexP]=coutP
		subst[indexP][i]=coutP
	subst[indexP][indexP]=0
	subst[indexH][indexR]=0

	return {"insert":inser,"delete":delet,"subst":subst}

# Initialize the automaton that reconizes _(rp*)*_
# q0 (- q1 - q2) (- q3 - q4) ( - q5 - q6) - q7
def appRPStar(alphabet,coutP=1,coutR=1):
	nAlph=len(alphabet)
	nEtat=8
	r=np.zeros((nEtat,nEtat))
	p=np.zeros((nEtat,nEtat))
	acc=np.full(nEtat,False,dtype='bool')
	
	r[0][1]=1
	r[2][3]=1
	r[4][5]=1
	r[6][5]=1
	
	p[2][2]=1
	p[1][2]=1
	p[3][4]=1
	p[4][4]=1
	p[5][6]=1
	p[6][6]=1
	p[6][7]=1
	
	acc[6]=True
	acc[7]=True
	trans=np.zeros((nAlph,nEtat,nEtat))
	trans[alphabet['r']]=r
	trans[alphabet['p']]=p
	
	# Dealing with the mappings to the initial/final state
	for i in range(nAlph):
		trans[i][0][0]=1
		trans[i][7][7]=1
	trans[alphabet['r']][0][0]=0
	trans[alphabet['p']][0][0]=0
	trans[alphabet['r']][7][7]=0
	trans[alphabet['p']][7][7]=0
	
	
	return Automaton(trans,acc,alphabet,standardCost(alphabet,coutP,coutR))

# Initialize the automaton that reconizes _(rp)*(rp^2)*(rp^3)*(rp^4)*_
def appAutNInc(alphabet,coutP=1,coutR=1):
	nAlph=len(alphabet)
	nEtat=15
	r=np.zeros((nEtat,nEtat))
	p=np.zeros((nEtat,nEtat))
	acc=np.full(nEtat,False,dtype='bool')
	
	r[0][2]=1
	r[1][2]=1
	r[3][4]=1
	r[6][7]=1
	r[10][11]=1
	
	p[2][1]=1
	p[2][3]=1
	p[4][5]=1
	p[5][3]=1
	p[5][6]=1
	p[7][8]=1
	p[8][9]=1
	p[9][6]=1
	p[9][10]=1
	p[9][14]=1
	p[11][12]=1
	p[12][13]=1
	p[13][10]=1
	p[13][14]=1
	
	acc[14]=True
	trans=np.zeros((nAlph,nEtat,nEtat))
	trans[alphabet['r']]=r
	trans[alphabet['p']]=p
	
	# Dealing with the mappings to the initial/final state
	for i in range(nAlph):
		trans[i][0][0]=1
		trans[i][14][14]=1
	trans[alphabet['r']][0][0]=0
	trans[alphabet['p']][0][0]=0
	trans[alphabet['r']][14][14]=0
	trans[alphabet['p']][14][14]=0
	
	return Automaton(trans,acc,alphabet,standardCost(alphabet,coutP,coutR))

# Count the number of words that conform to a pattern up to tau
# "auto": the automaton that represents the good strings
# "tau": the percentage of mistakes allowed in a string
# "sigma": standard deviation of the local shuffle
def countPatterns(data,auto,tau=0.3,display=False):
	n=0
	for s in data:
		if auto.distToPat(s)<tau*len(s):
			if display:
				print(s)
			n+=1
	return n

# Finds the best match between strings of two lists in terms of a given edit distance
# "words1" and "words2" are two word lists
# "d" is a distance function
def matchStrings(words1,words2,d):
	m=Munkres()
	distMatrix=[[d(word1,word2) for word2 in words2] for word1 in words1]
	indices=m.compute(distMatrix)
	return [(words1[row],words2[column]) for row,column in indices]