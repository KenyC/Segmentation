from ForwardModel import ForwardHMM
from string import ascii_lowercase

filePath = "data/word_alpha.txt"
alphabet = list(ascii_lowercase)

corpus = []
with open(filePath,'r') as f:
    corpus = [line[:-1] for line in f]

# Forward model: random initialization
n = 5
fm = ForwardHMM.randHMM(alphabet, n)

fm.baumwelch(corpus)
print(fm.gen(10))
        
