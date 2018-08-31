'''
Alphabetical list of part-of-speech tags used in the Penn Treebank Project:
Number
	
Tag
	
Description
1. 	CC 	Coordinating conjunction
2. 	CD 	Cardinal number
3. 	DT 	Determiner
4. 	EX 	Existential there
5. 	FW 	Foreign word
6. 	IN 	Preposition or subordinating conjunction
7. 	JJ 	Adjective
8. 	JJR 	Adjective, comparative
9. 	JJS 	Adjective, superlative
10. 	LS 	List item marker
11. 	MD 	Modal
12. 	NN 	Noun, singular or mass
13. 	NNS 	Noun, plural
14. 	NNP 	Proper noun, singular
15. 	NNPS 	Proper noun, plural
16. 	PDT 	Predeterminer
17. 	POS 	Possessive ending
18. 	PRP 	Personal pronoun
19. 	PRP$ 	Possessive pronoun
20. 	RB 	Adverb
21. 	RBR 	Adverb, comparative
22. 	RBS 	Adverb, superlative
23. 	RP 	Particle
24. 	SYM 	Symbol
25. 	TO 	to
26. 	UH 	Interjection
27. 	VB 	Verb, base form
28. 	VBD 	Verb, past tense
29. 	VBG 	Verb, gerund or present participle
30. 	VBN 	Verb, past participle
31. 	VBP 	Verb, non-3rd person singular present
32. 	VBZ 	Verb, 3rd person singular present
33. 	WDT 	Wh-determiner
34. 	WP 	Wh-pronoun
35. 	WP$ 	Possessive wh-pronoun
36. 	WRB 	Wh-adverb 
'''



import sys
import platform
import nltk

from nltk.tokenize import PunktSentenceTokenizer, word_tokenize
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
from nltk.corpus import state_union

exampleText = "beautifully running in the United Kingdom, UK. Number 1 I have been paying you or ecoweb hosting for 3 months  and I do not understand when I log in it takes me to the customer area. Silently talking I bought your deal when you were advertising for the advanced hosting at $4.99 pounds but it asks me to pick my hosting which I already have, as I mentioned. The other thing is I believed I had transferred my domain about two months ago which was theplrproducts4u.com but it says it can not find my domain and I have to pay to transfer that domain. I do not know if the problem is I had just transferred the domain but did not build a site with it. I do not know if you guys removed it because I have not done anything with it . But I figured that if I was paying my monthly fee there would not be a problem. I know that the last time that I had logged in which might be a while(maybe a month ago I think but you guys might have a better idea ) I saw my domain proliferate. I think that was the word that my domain co. said that I could take a couple of days and then go back to check it make sure it had done so.I know it proliferated at I was able to log in and saw my domain but now its' gone. So if I have done something wrong you can explain because as of this moment I have not decided to keep the account or cancel it. Also are you a different company,  because when I tried to log in Eco hosting as oppose to Eco web (which is your company) they stated I did not exist or they could not find me in their accounts.Any that's it for now and thank you. You know something strange that happened when you indicated to write a ticket, as I was writing theplrproducts.com showed up in your subject line automatically but when I logged in to my site the above happened which means it did not go to my advanced hosting site and it did not show my domain. But was requesting for me to pay to transfer my domain but I was able to log in but it just goes to the customer area."


# exampleSentences = sent_tokenize(exampleText)
# print(word_tokenize(exampleText))

#stop words and work/sentence tokenisation

#stopWords = set(stopwords.words("english"))
# filteredSentence = []

# for i in sent_tokenize(exampleText):
#     words = word_tokenize(i)
#     for w in words:
#         if w not in stopWords:
#             filteredSentence.appenr"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""d(w)

# print(filteredSentence)

#stemming

# ps = PorterStemmer()

# for w in filteredSentence:
#     print(ps.stem(w))

# stopWords = set(stopwords.words("english"))
# filteredSentence = []

trainText = state_union.raw("2005-GWBush.txt")

customSentTokenizer = PunktSentenceTokenizer(trainText)
tokenized = customSentTokenizer.tokenize(exampleText)

#stop words
# for i in tokenized:
#     words = word_tokenize(i)
#     for w in words:
#         if w not in stopWords:
#             filteredSentence.append(w)


# ##working chunking and chinking
# def processContent():
# 	try:
# 		for i in tokenized:
# 			words = nltk.word_tokenize(i)
# 			tagged = nltk.pos_tag(words)

# 			#print(tagged);

# 			chunkGram = r"""Chunk: {<.*>+}
# 			                        }<VB.?|IN|DT>+{"""

# 			chunkParser = nltk.RegexpParser(chunkGram)
# 			chunked = chunkParser.parse(tagged)

# 			print(chunked)
# 			chunked.draw()

# 	except Exception as e:
# 		print(str(e))

# processContent()


def processContent():
	try:
		for i in tokenized:
			words = nltk.word_tokenize(i)
			tagged = nltk.pos_tag(words)

			namedEnt = nltk.ne_chunk(tagged)

			namedEnt.draw()
			#print(namedEnt)

	except Exception as e:
		print(str(e))

processContent()