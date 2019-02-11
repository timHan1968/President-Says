"""
This python script constructs unigram and bigram language models and then
calculates the probabilities for the language models

Authors: Sena Katako, Vivian Gao, Xianyi Han
"""

import math
from random import uniform, randint
import csv
import sys

BEGINNING_OF_SENTENCE = "<s>"
END_OF_SENTENCE = "</s>"
UNKNOWN_WORD = "<UNK>"
#For postprocessings
STOPPERS = ['.', '?', '!']
IREGULARS = [',', '’']

class UnigramLM:

	"""
	The unigram language model class......
	"""

	def __init__(self, filepath):

		"""
		The constructor initializes the corpus to be stored in the unigram model.
		It creates a dictionary of tokens and value and also counts the total
		number of words in the corpus.

		Input
		-----
		- filepath: The path of the file to be read in
		"""

		sentences = readFile(filepath)
		self._tokens, self._num_words = makeUnigrams(sentences)
		self._tokensUnk, self._removedList = makeUnkUni(self._tokens)


	def calWordProbUnk(self, word):
		"""
		A function that returns probability of a given word. With
		unknown words implemented, the funciton works regardless of
		the given word's presence in the training set.

		Input
		-------
		word: a given word in the form of string

		output
		-------
		returns the probability value (not log) of the word

		"""

		if word not in self._tokensUnk:
			value = self._tokensUnk.get(UNKNOWN_WORD, 0)
		else:
			value = self._tokensUnk.get(word, 0)
		return value/self._num_words


	def makeUnigramProbTable(self):

		"""
		Returns the log probability table of the unigram. Does not include
		Unknown Words as the table is intended for random sentence generation.

		Output
		------
		probTable: a dictionary of {unigram: log probability}
		"""

		probTable = {k: math.log(v/self._num_words) for k,v in self._tokens.items()}

		return probTable


	def generateRandomUnigramSentences(self, seed = "", generate = True):

		"""
		This function calculates the probability of a bigram relative
		to all the bigrams that share the same start word.

		Input
		------
		- seed: the incomplete sentence to start the generation with
		- generate: determines whether a random unigram sentence should be formed

		Output
		------
		- sentence: the randomly generated sentence

		"""
		table = self.makeUnigramProbTable()

		sentence = seed.strip()

		max = 20
		count = len(sentence.split())

		while generate:

			n = uniform(0,1)
			interval = 0

			for k,v in table.items():

				interval += math.exp(v)
				if interval >= n:

					if k == END_OF_SENTENCE:
						if len(sentence) != 0:
							generate = False
					elif k not in STOPPERS and k != BEGINNING_OF_SENTENCE:
						sentence = " ".join([sentence, k])
						count += 1

					if (count == max) or generate == False:
						generate = False
						sentence = "".join([sentence, "."])

					break

		for s in STOPPERS:
			sentence = sentence.replace((' ' + s + ' '), " ")

		for r in IREGULARS:
			sentence = sentence.replace((' ' + r), r)

		return sentence.strip()


	def unigramPerplexity(self, setOfWords):

		"""
		A function that returns the model's perplexity value using
		[setOfWords] as the development set.

		Input
		------
		- setOfWords: The development set as a list of word tokens.

		Output
		------
		- perplexity: The calculated perplexity.
		"""
		total = 0
		n = 0

		for word in setOfWords:
			prob = self.calWordProbUnk(word)
			total += -1 * math.log(prob)
			n += 1
		perplexity = math.exp((1 / n)* total)

		return perplexity


class BigramLM(UnigramLM):

	"""
	The bigram language model class......
	"""

	def __init__(self, filepath):

		"""
		The constructor initializes the corpus to be stored in the bigram model.
		It inherits from the Unigram Language Model and also creates a dictionary
		of bigrams and counts.

		Input
		-----
		- filepath: The path of the file to be read in
		"""
		super().__init__(filepath)

		sentences = readFile(filepath)
		self._separatedbigrams = makeBigrams(sentences)

		self._bigramsUnk = makeUnkBi(sentences, self._removedList)


	def calBiProbLap(self, firstWord, secondWord):
		"""
		a function that returns the probabililty of a given bigram. With
		both unknown word and add-one smoothing implemented, the function
		works for any given bigram.

		Input
		------
		- firstWord: First word of a given bigram
		- secondWord: Second word of a given bigram

		Output
		------
		- prob: the probability of the bigram
		"""
		v = len(self._tokensUnk) #Number of tokens

		#Possible Edge Case?
		if firstWord == END_OF_SENTENCE:
			return 0

		if firstWord not in self._tokensUnk:
			firstWord = UNKNOWN_WORD

		if secondWord not in self._tokensUnk:
			secondWord = UNKNOWN_WORD

		if secondWord not in self._bigramsUnk[firstWord]:
			num = 1
		else:
			num = self._bigramsUnk[firstWord][secondWord] + 1

		den = self._tokensUnk[firstWord] + v

		return num/den


	def makeBigramProbTable(self):

		"""
		Returns the log probability table of a Bigram. Does not include
		Unknown Words or implement smoothing as the function is intended
		for random sentence generation.

		Output
		------
		probTable: a dictionary of {bigrams: log probability}
		"""

		probTable = {}

		for k,v in self._separatedbigrams.items():

			probTable[k] = {}
			for s,t in v.items():
				probTable[k][s] = math.log(t/self._tokens[k])

		return probTable


	def generateRandomBigramSentences(self, seed=""):

		"""
		This function returns a randomly generated sentence that starts from
		and completes the given string [seed]

		Input
		------
		- seed: the incomplete sentence start the genration with
		- generate: determines whether a random unigram sentence should be formed

		Output
		------
		- sentence: the randomly generated sentence

		"""

		table = self.makeBigramProbTable()

		sentence = seed.strip()
		prev_word = ""

		n = uniform(0,1)
		interval = 0
		max = 20
		counter = len(sentence.split())

		if sentence == "":
			#Choose a random word to start
			for k, v in table[BEGINNING_OF_SENTENCE].items():
				interval += math.exp(v)

				if interval >= n:
					sentence = k
					prev_word = k
					counter += 1
					break
		else:
			#Assume seed has known words only
			prev_word = sentence.split()[-1]


		while END_OF_SENTENCE not in sentence and counter < max:
			interval = 0
			tokens = table[prev_word]
			n = uniform(0,1)

			for k,v in tokens.items():

				interval += math.exp(v)

				if interval >= n or k == END_OF_SENTENCE:
					sentence = " ".join([sentence, k])
					prev_word = k
					counter += 1
					break

		#PostProcessings:
		sentence = sentence.replace(" </s>", "")

		if sentence[-1:] in STOPPERS:
			sentence = sentence.replace(sentence[-2:], sentence[-1:])
		else:
			#When sentence ends due to reaching maximum length
			sentence = "".join([sentence, "."])

		for r in IREGULARS:
			sentence = sentence.replace((' ' + r), r)

		return sentence


	def bigramPerplexity(self, setOfWords):

 	    """
 	    The function returns the perplexity value using [setOfWords] as the
 	    development set.

 	    Input
 	    ------
 	    - setOfWords: The development set as a list of word tokens.

 	    Output
 	    ------
 	    - perplexity: The calculated perplexity value.
 	    """

 	    total = 0
 	    n = 0

 	    for i in range(1, len(setOfWords)):

 	    	prevWord = setOfWords[i-1]
 	    	currentWord = setOfWords[i]

 	    	if prevWord != END_OF_SENTENCE:
 	    		prob = self.calBiProbLap(prevWord, currentWord)
 	    		total += -1 * math.log(prob)
 	    		n += 1

 	    perplexity = math.exp((1 / n)* total)

 	    return perplexity
#Class Mehtods Ends Here


def readFile(filepath):

	"""
	Reads a text file and returns a list of paragraphs

	Input
	------
	filepath: the file path

	Output
	-------
	text: a list of the sentences in the text file
	"""

	with open(filepath, "r", encoding = "utf-8") as fp:
		text = fp.readlines()
	fp.close()
	return text


def addMacro(words, addBegin):
	"""
	Take in a list of words then preprocss it by adding proper macroes
	like BEGINNING_OF_SENTENCE and END_OF_SENTENCEself.

	Input
	-------
	- words: A list of strings that contain all tokens of a line in txt
	- addBegin: A boolean value indicating whether the sentence starter
	should be inserted at beginning of the given line.

	Output
	--------
	- words: The original [words] but with proper macros inserted
	- addBegin: Whether the next line begins as a new sentence.
	"""
	if addBegin:
		words.insert(0, BEGINNING_OF_SENTENCE)
		addBegin = False

	r = len(words)

	for i in range(r):
		if words[i] in STOPPERS:
			words.insert(i+1, END_OF_SENTENCE)
			if (i+2) == len(words):
				#Already at end of line
				addBegin = True
			elif words[i+2] != '”':
				words.insert(i+2, BEGINNING_OF_SENTENCE)

	#Continue to work if [line] has multiple sentences
	while(r != len(words)):
		new_r = len(words)
		for i in range(r, new_r):
			if words[i] in STOPPERS:
				words.insert(i+1, END_OF_SENTENCE)
				if (i+2) == len(words):
					#Already at end of line
					addBegin = True
				elif words[i+2] != '”':
					words.insert(i+2, BEGINNING_OF_SENTENCE)
		r = new_r

	return words, addBegin


def readDevelop(filepath):
	"""
	Read a development set text file and returns a list of words
	with proper preprocessing

	"""
	with open(filepath, "r", encoding = "utf-8") as fp:
		lines = fp.readlines()

	ls = []
	addBegin = True

	for line in lines:

		words = line.strip().split()
		words, addBegin = addMacro(words, addBegin)
		ls.extend(words)

	fp.close()

	return ls


def makeUnigrams(sentences):

	"""
	Returns a dictionary of unigram tokens and the total number of
	words in the corpus.

	Input
	------
	sentences: a list of all sentences in the corpus

	Output
	-------
	tokens: a dictionary of all tokens and their counts in the corpus
	numWords: an integer representing the total number of words in the corpus
	"""

	tokens = {}
	numWords = 0
	addBegin = True

	for line in sentences:

		words = line.strip().split()

		words, addBegin = addMacro(words, addBegin)

		n = len(words)
		numWords += n

		for word in words:

			if word not in tokens:
				tokens[word] = 1
			else:
				tokens[word] += 1

	return tokens, numWords


def makeBigrams(sentences):

	"""
	Returns a dictionary of bigrams and their counts

	Input
	------
	sentences: a list of all sentences in the corpus

	Output
	-------
	pairs: a dictionary of all bigrams and their counts in the corpus
	"""

	bi_separated = {}
	addBegin = True

	for line in sentences:

		words = line.strip().split()

		words, addBegin = addMacro(words, addBegin)

		for i in range(len(words[:-1])):

			preceedingWord = words[i]
			currentWord = words[i+1]

			#Makes the separated case
			if preceedingWord != END_OF_SENTENCE:
				if preceedingWord not in bi_separated.keys():
					bi_separated[preceedingWord] = {words[i+1]:1}
				else:
					if currentWord not in bi_separated[preceedingWord]:
						bi_separated[preceedingWord][currentWord] = 1
					else:
						bi_separated[preceedingWord][words[i+1]] += 1

	return bi_separated


def makeUnkUni(tokens):

    """
    Returns a Unigram dictonary that includes Unknown words and a
	list of the words removed as Unknowns

    Input
	------
	tokens: a dictionary of all tokens and their counts in corpus without
    managing unknowns

	Output
	-------
	tokensUnk: a dictionary of all tokens and their counts in corpus after
    handling unknowns
    removedList: a list of words removed as Unkowns
    """
    tokensUnk = tokens.copy()

    tokensUnk[UNKNOWN_WORD] = 0
    removed = 0
    threshold = 2  #words with frequency < [threshold] is replaced by Unknown
    max = 20  #Maximum number of words replaced as Unknown
    removedList = []

    for k, v in list(tokensUnk.items()):

        if v < threshold and k != UNKNOWN_WORD:

            removedList.append(k)
            tokensUnk[UNKNOWN_WORD] += v
            removed += 1
            tokensUnk.pop(k, None)

        if removed >= max:
            break

    return tokensUnk, removedList


def makeUnkBi(sentences, removedList):

	"""
	A modified version of [makeBigrams] that handles Unknowns

	Input
	------
	sentences: a list of all sentences in the corpus
    removedList: a list of words removed as unknowns

	Output
	-------
	bi_separated: a dictionary of all bigrams and their counts in the corpus
	"""
	bi_separated = {}
	addBegin = True

	for line in sentences:

		words = line.strip().split()

		#appends the start word and end word symbols
		words, addBegin = addMacro(words, addBegin)

		for index, word in enumerate(words):
			if word in removedList:
				words[index] = UNKNOWN_WORD

		for i in range(len(words[:-1])):

			preceedingWord = words[i]
			currentWord = words[i+1]

			#Makes the separated case
			if preceedingWord != END_OF_SENTENCE:
				if preceedingWord not in bi_separated.keys():
					bi_separated[preceedingWord] = {words[i+1]:1}
				else:
					if currentWord not in bi_separated[preceedingWord]:
						bi_separated[preceedingWord][currentWord] = 1
					else:
						bi_separated[preceedingWord][words[i+1]] += 1

	return bi_separated


def readTest(filepath):
	"""
	The function reads a test file and processes it into a list
	of lines (list of words). Also added sentence begginer and stopper
	tokens.
	"""
	lines = readFile(filepath)
	unseenSpeech = []
	addBegin = False

	for line in lines:
		words = line.strip().split()
		words, addBegin = addMacro(words, addBegin)
		addBegin = False
		unseenSpeech.append(words)

	return unseenSpeech


def seqProbability(model, line):
	"""
	The function takes in a BigramLM object and use it to calculate the
	probability of a given [line] which is a list of words.
	"""
	table = model.makeBigramProbTable()
	for word in line:
		if word == line[0]:
			if word in table['<s>']:
				prob = math.exp(table['<s>'][word])
			else:
				prob = model.calBiProbLap('<s>',word)
			sequenceProb = prob
			previous=word
		else:
            #find P(word|previous word)
			if previous != '</s>':
				if previous in table and word in table[previous]:
					prob = math.exp(table[previous][word])
				else:
					prob = model.calBiProbLap(previous, word)
				sequenceProb = sequenceProb * prob
			previous = word
	return sequenceProb


def speechClassify(obamaModel,trumpModel,unseenSpeech):
    """
    The function takes in the obama and trump Bigram model and returns
    the speech classification of a given script in csv format.
    """
    lineNumber = 0
    prediction = None

    with open('LMprediction.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'Prediction'])
        for line in unseenSpeech:
            obamaProb = seqProbability(obamaModel, line)
            trumpProb = seqProbability(trumpModel, line)
            if obamaProb > trumpProb:
                prediction = 0
            else:
                prediction = 1
            writer.writerow([lineNumber, prediction])
            lineNumber += 1
    f.close()


def printUniTable(model, outputFile):
	"""
	The function that prints the Unsmoothed Unigram Prob Table into
	a txt file.
	"""
	table = model.makeUnigramProbTable()
	with open(outputFile, 'w', encoding = "utf-8") as f:
		f.write("Unigram  Probability\n")
		f.write("---------------------------------------\n")
		for k, v in table.items():
			f.write(k + '    ' + str(math.exp(v)) + '\n')
		f.write("---------------------------------------\n")
	f.close()


def printBiTable(model, outputFile):
	"""
	The function that prints the Unsmoothed Bigram Prob Table into
	a txt file.
	"""
	table = model.makeBigramProbTable()
	with open(outputFile, 'w', encoding = "utf-8") as file:
		file.write("Bigram    Probability\n")
		file.write("---------------------------------------\n")
		for k, v in table.items():
			previous = k
			for s, f in table[previous].items():
				bigram = previous + ' ' + s
				file.write(bigram + '    ' + str(math.exp(f)) + '\n')
		file.write("---------------------------------------\n")
	file.close()


def main(option='demo'):

	file_o = "Assignment1_resources/train/obama.txt"
	file_t = "Assignment1_resources/train/trump.txt"

	dev_o = readDevelop("Assignment1_resources/development/obama.txt")
	dev_t = readDevelop("Assignment1_resources/development/trump.txt")

	xx_o = UnigramLM(file_o)
	xx_t = UnigramLM(file_t)

	tt_o = BigramLM(file_o)
	tt_t = BigramLM(file_t)

	if option == "trumpsays":
		print(tt_t.generateRandomBigramSentences())
	elif option == "obamasays":
		print(tt_o.generateRandomBigramSentences())
	else:
		print("Unigram Obama says:")
		print(xx_o.generateRandomUnigramSentences())
		print("And perplexity is:")
		print(xx_o.unigramPerplexity(dev_o))
		print()

		print("Unigram Trump says:")
		print(xx_t.generateRandomUnigramSentences())
		print("And perplexity is:")
		print(xx_t.unigramPerplexity(dev_t))
		print()

		print("Bigram Obama says:")
		print(tt_o.generateRandomBigramSentences())
		print("And perplexity is:")
		print(tt_o.bigramPerplexity(dev_o))
		print()

		print("Bigram Trump says:")
		print(tt_t.generateRandomBigramSentences())
		print("And perplexity is:")
		print(tt_t.bigramPerplexity(dev_t))
		print()
		


if __name__ == "__main__":

	if len(sys.argv) > 1:
		option = sys.argv[1]
		main(option)
	else:
		main()
