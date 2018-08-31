#lematising

import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize
from nltk.stem import WordNetLemmatizer

exampleText = "beautifully running in the United Kingdom, UK. Number 1 I have been paying you or ecoweb hosting for 3 months  and I do not understand when I log in it takes me to the customer area. Silently talking I bought your deal when you were advertising for the advanced hosting at $4.99 pounds but it asks me to pick my hosting which I already have, as I mentioned. The other thing is I believed I had transferred my domain about two months ago which was theplrproducts4u.com but it says it can not find my domain and I have to pay to transfer that domain. I do not know if the problem is I had just transferred the domain but did not build a site with it. I do not know if you guys removed it because I have not done anything with it . But I figured that if I was paying my monthly fee there would not be a problem. I know that the last time that I had logged in which might be a while(maybe a month ago I think but you guys might have a better idea ) I saw my domain proliferate. I think that was the word that my domain co. said that I could take a couple of days and then go back to check it make sure it had done so.I know it proliferated at I was able to log in and saw my domain but now its' gone. So if I have done something wrong you can explain because as of this moment I have not decided to keep the account or cancel it. Also are you a different company,  because when I tried to log in Eco hosting as oppose to Eco web (which is your company) they stated I did not exist or they could not find me in their accounts.Any that's it for now and thank you. You know something strange that happened when you indicated to write a ticket, as I was writing theplrproducts.com showed up in your subject line automatically but when I logged in to my site the above happened which means it did not go to my advanced hosting site and it did not show my domain. But was requesting for me to pay to transfer my domain but I was able to log in but it just goes to the customer area."

lemmatizer = WordNetLemmatizer()
trainText = state_union.raw("2005-GWBush.txt")

customSentTokenizer = PunktSentenceTokenizer(trainText)
tokenized = customSentTokenizer.tokenize(exampleText)

def processContent():
	try:
		for i in tokenized:
			words = nltk.word_tokenize(i)
			for w in words:
				print(lemmatizer.lemmatize(w))
				print(w)
	except Exception as e:
		print(str(e))

processContent()