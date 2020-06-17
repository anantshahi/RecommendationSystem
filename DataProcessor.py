from nltk import *

from textblob import TextBlob
from nltk.corpus import stopwords
import re
import string


#class user input is made to take the input from user, currently a string, later changed to file input
#self._userstring is a protected raw string 


class user_input:
    
   def user(self):
        self._userstring = input("Enter whatever string you want to enter ").lower()
        print("string in lower case:",self._userstring)



'''
class datacleaning consists of all the required layered processing, all the variables ae protected
and this module inherits class user_input  

'''



class datacleaning(user_input):

    """
conversion function  converts the user string from user input class into two formats:-
1) It stores the user string into the protected variable 'self._string1' for nltk processing
2)It stores the user string into the protected variable 'self._string2' for TextBlob processing 



    """


    def conversion(self):
        self._string1 = self._userstring #self._name = name  #protected attribute 
        self._string2 = TextBlob(self._userstring)  #self._name = name  #protected attribute 
        print(self._string1,"\n")
        print(self._string2)


    
    

   

    """
this module will remove the digits, if mistakingly present in the string 

    """



    def digits(self):
     
        DigitsList = ['1', '2', '3', "4","5","6","7","8","9","0"] 

# initializing test string 
        self._digitstring = self._string1

# printing original string 
        print ("Original String : " + self._digitstring) 


 
        for i in DigitsList : 
	        self._digitstring = self._digitstring.replace(i, '') 

# printing resultant string 
        print ("Resultant list is : " + str(self._digitstring)) 





        # this spl char function will remove any special character present in the string 



    def spl_char(self):
         

# initializing bad_chars_list 
        bad_chars = [';', ':', '!', "*","@","#","%","$","^","&","{","}"] 

 
        

# printing original string 
        print ("Original String : " + self._digitstring) 

 
 
        for i in bad_chars :
            self._splcharstring = self._digitstring.replace(i,"")
            self._digitstring=self._splcharstring

        #printing resultant string 
        print ("Resultant list is : " + str(self._splcharstring)) 




# this module will remove extra white spaces from the string 

    def extra_white_spaces(self):
        i = " "
        for i in self._splcharstring:
            self._xtrawspace = self._splcharstring.replace("  "," ")
            self._splcharstring = self._xtrawspace
        print(self._xtrawspace)

 
 #this module wil remove any spelling errors form the string 
 
    def spellchecking(self):
        self._SpellString = TextBlob(self._xtrawspace)
        self._SpellCorrectedString = self._SpellString.correct()
        print("Corrected string is: " + str(self._SpellCorrectedString))



#this module tags the parts of speech using textblob 

    def pos_tagging(self):
        self._ResultTag =  self._SpellCorrectedString.pos_tags
        print(self._ResultTag)
        



#this module removes the stop words from the sring and prepares for further processing 

    def stop_words(self):
         self._StopWords = set(stopwords.words('english'))
         self._match = ""
         
         self._match = self._match.join(self._SpellCorrectedString)
         print(self._match)

         self._tokens = word_tokenize(self._match)
         print(self._tokens)
         self._result = [i for i in self._tokens if not i in self._StopWords]
         print("i removed the stop words, result is:",self._result)

         self._StopWordsRemovedString = " "
         self._StopWordsRemovedString = self._StopWordsRemovedString.join(self._result)
         

        

        
        


#this module tags the relation between named enitites 

    def named_entity_recognition(self):
        print(ne_chunk(pos_tag(word_tokenize(self._StopWordsRemovedString))))


    
#this module starts to tokenize, count words for freq dist and sentences 

    def tokenize(self):
        self._t1 = TextBlob(self._StopWordsRemovedString).words #breaks into words 
        print(self._t1)

        self._t2= TextBlob(self._StopWordsRemovedString).sentences #breaks into sentences
        print(self._t2) 

        self._t3= TextBlob(self._StopWordsRemovedString).word_counts #wordcount, it will be used as frequency distribution 
        print(self._t3)




#this module stems and tokenize the string obtained, list item from this has to be carried forward into next module


    def stemming_func(self):
        self._stemmer= PorterStemmer()
        self._ntokens = word_tokenize(self._StopWordsRemovedString)
        self._StemList = []
        for i in self._ntokens:
            self._val = self._stemmer.stem(i)
            self._StemList.append(self._val)

        print(self._StemList)
        
    
    
    

    
    def lemme_func(self):
        self._lemmatizer = WordNetLemmatizer()
        self._ntokens = word_tokenize(self._StopWordsRemovedString)

        self._Lemmelist = []

        for i in self._ntokens:
            self._Lemmelist.append(self._lemmatizer.lemmatize(i))
            
        print(self._Lemmelist)
        

    


#CLASS OBJECTS DECLARATION  

mandatory = user_input()
dataflow = datacleaning()


#   IT IS MANDATORY TO FOLLOW THE ORDER 
dataflow.user()

dataflow.conversion()

dataflow.digits()
dataflow.spl_char()
dataflow.extra_white_spaces()
dataflow.spellchecking()
dataflow.pos_tagging()
dataflow.stop_words()
dataflow.named_entity_recognition()
dataflow.tokenize()
dataflow.stemming_func()
dataflow.lemme_func()




    





