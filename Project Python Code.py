# Using NLTK package as a toolkit for working with NLP in Python. 
# It provides various text processing libraries with a lot of test datasets.
#nltk.download()
#pip install 
#pip install beautifulsoup4

import nltk

'''
----------------- Corpus -----------------
'''

import os, os.path # os.path module is always the path module suitable for the operating system Python is running on.


# A function that takes raw corpus, then creates a .txt file with the raw corpus inside it.
# The function takes 3 parameters:
# 1st as the name of the txt file wanting to create.
# 2nd as the raw corpus.
# 3rd as the function desired to be done to the .txt file. Ex. write or append.
def to_txt(create, write, function):
    
    general_path = 'C:/Users/ragid/Downloads/MS' # files path.
    
    txt_path = os.path.join(general_path, create) # Combining both the general_path and the name of the file to get the final path.
    
    create_txt = open(txt_path, function, encoding = "utf-8") # Open the txt file for writing and reading.
  
    create_txt.write(write) # wrting into the txt file.
    
    close_txt = create_txt.close() # Close the txt file.
    
    return close_txt


# Function that takes a url and extract only the text paragraphs from it. Then send the corpus to to_txt function to write it into a txt file.
def web_corpus(url):
    
    # urllib is a package that collects several modules for working with URLs.
    # urllib.request for opening and reading URLs
    from urllib import request

    # Some links might have an issue to access, due to that we do (try: except:).
    try:
        url_response = request.urlopen(url) # Opening the URL.
        url_rd = url_response.read().decode("utf8") # Read the whole content of the url and decode it.


        # URL BeautifulSoup.
        # BeautifulSoup is a package for parsing HTML and XML documents. 
        # It creates a parse tree for parsed pages that can be used to extract data from HTML.
        from bs4 import BeautifulSoup
    
        url_text_raw = BeautifulSoup(url_rd, "html.parser") # "html.parser" is a structured markup processing tool used to parse HTML files.
    
        url_text_para = url_text_raw.find_all("p") # Only selecting data with tag p (paragraph).
    
        clean_text = "" # A variable that stores the clean corpus from the web page.
        
        # for loop which stores the corpus from the web in a variable.
        for i in url_text_para:
            clean_text = clean_text + i.get_text()
        
        
        # In some cases, the sentences taken from the websites are connected. The following will separate each sentence.
        clean_text1 = clean_text.split('.') # Separating each sentence.
        
        clean_text2 = "" # A variable that stores the clean corpus from the web page.
        
        # for loop which store the corpus from the web in a variable.
        for i in clean_text1:
            clean_text2 = clean_text2 + i + '. '
        
    
        # URL txt File Creation and Passing Corpus.
        to_txt('url_text_raw.txt', clean_text2, "a") # Passing values to to_txt function.
  
    except:
        print("An exception occurred")
    

# Function that takes a corpus and tokenize it into lists of tokenized sentences and every sentence tokenized into words.
def tokenize_corpus(corpus):
    
    # First, convert all the words to lowercase letters.
    sent_token_lowercase = corpus.lower()
    
    
    # Second, tokenize the corpus into sentences.
    # Third, tokenize the sentences into words.
    from nltk.tokenize import sent_tokenize, word_tokenize
    import string # A function that accepts a string name with dot separators.
    
    table = str.maketrans('', '', string.punctuation) # Remove punctuation from each word.
    word_token = [t.translate(table) for t in sent_tokenize(sent_token_lowercase)] # Apply removing punctuations from each sentence.
    word_token = [word_tokenize(t) for t in word_token] # Tokenize each sentence into words.
    word_token = [[word for word in word_token if word.isalpha() or word.isnumeric()] for word_token in word_token] # Remove remaining tokens that are not alphabetic or numeric.
    
    return word_token



'''
----------------- Spell Correction -----------------
'''

from collections import Counter # Counter() function is used to count the occurrences of each word in the corpus.

# Reading the txt File.
# Wordlist from https://github.com/mertemin/turkish-word-list (Total words: 78,233) + Turkish alphapets + numbers from 0 to 10k
url_data_load = nltk.data.load("words.txt") # Used to load NLTK resource files, such as corpora, grammars, and saved processing objects.

# Tokenization of a txt file to get a list of words to use as a dictionary.
# nltk.tokenize is a Package for dividing strings into lists of substrings.
# word_tokenize is a function that splits a given sentence into words.
from nltk.tokenize import word_tokenize
token = word_tokenize(url_data_load)
dictionary = Counter(token)


# A function that creates a set containing the subset of each word that appears in the dictionary.
def known(word):
    word_subset = set(w for w in word if w in dictionary)
    return word_subset


# A function that gets the probability of each word.
def word_probability(word):
    word_sum = sum(dictionary.values()) # Summing all the number of words occurring in the dictionary.
    word_prob = dictionary[word] / word_sum # Dividing the word by all the words in the dictionary to get it's probability.
    return word_prob


# A function that do a first edit. All edits that are one edit away from a word.
def edits1(word):   
    letters    = 'abcçdefgğhıijklmnoöprsştuüvyz' # A variable that holds all the alphabetic letters.
    splits     = [(word[:i], word[i:])  for i in range(len(word) + 1)] # Splitting each word multiple times by deleting the first letter in each iteration.
    deletes    = [L + R[1:]               for L, R in splits if R] # Using the splits variable to get all the possible combinations of the word, then delete the extra letters to get all the possible combinations.
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1] # A transposition is to swap two adjacent letters.
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters] # A replacement is to change one letter to another and this will be done to each letter in the word.
    inserts    = [L + c + R               for L, R in splits for c in letters] # An insertion is to add a letter. And this will be done before the word, between each two letters, and after the word.
    
    # Creating a set containing all edited strings (whether words or not) that can be made with one simple edit.
    edits1 = set(deletes + transposes + replaces + inserts)
    return edits1


# A function that selects a candidate by generating possible spelling corrections for a word.
def candidates(word): 
    candidate = (known([word]) or known(edits1(word)) or [word])
    
    candidate_list = list(candidate)
    
    return candidate, candidate_list


# A function that selects the most probable spelling correction for a word.
def correction(word):
    from tabulate import tabulate # A module which is used to print tabular data in nicely formatted tables.
    
    table = [] # Table to show results.
    col_names = ["Misspell", "Suggestions", "Correct Spell"] # Table header.
    all_words = [] # List containing all words from the array provided.
    misspell_word = [] # List containing the misspelled words.
    suggestions = [] # List containing suggested words for correction.
    correct = [] # List containing correct words.
    total_correct = [] # List containing only correct words.
    counter = 0
    counter2 = 0
    
    # nested for loop to access the nested word list and replace each word with its spell correction.
    for i in range(len(word)):
        for j in range(len(word[i])):
            
            all_words.append(word[i][j]) # Now it contain all the words in the list.
            word[i][j] = max(candidates(word[i][j])[0], key = word_probability) # Replace the old List misspell words with correct ones. max() function returns the largest item in an iterable.
 
            if all_words[counter] == word[i][j]: # Check if the word is correct or not.
                total_correct.append(all_words[counter])
             
            elif all_words[counter] != word[i][j]: # Check if the word is correct then ignore it, if it's wrong the proceed.
                misspell_word.append(all_words[counter]) # Add the misspelled words to a new list.
                
                suggestions.append((word[i][j], f"{word_probability(word[i][j]):.9f}")) # Add the suggested words for correction to a new list.

                correct.append(word[i][j]) # Add the correct words to a new list.
                
                table.append([misspell_word[counter2], suggestions[counter2], correct[counter2]]) # Creating an array that holds the values to display as a table.
                counter2 += 1
            counter += 1

    print(tabulate(table, headers = col_names, tablefmt = "fancy_grid")) # Printing the spell correction results as a table.
    
    print('- Total words:', counter)
    print('- Total Correct words:', len(total_correct))
    print('- Total Misspelled words:', len(misspell_word))
                
    return word



'''
----------------- Unigram, Bigram and Trigram -----------------
'''

# Function that takes a tokenized corpus and creates a ngrams for it using the NLTK package.
def ngram(corpus, ngrams):
    
    '''
    ##### Data Preparing #####
    '''

    '''
    # Importing two libraries from NLTK package:
    # 1st bigrams and trigrams, to turn a text into them. 
    # 2nd pad_both_ends, to add special “padding” symbols to the sentence which are:
      <s>" at the start of the sentence AND "</s>" at the end of the sentence.
    '''
    from nltk.util import bigrams, trigrams
    from nltk.lm.preprocessing import pad_both_ends
    
    ngram = 0 # Variable containing the number of ngrams.
    
    if ngrams == 'bigrams':
        ngram = 2
        ngram_list = []
        # for loop that goes over every sentence in the corpus and turn it into bigrams. Every sentence is inside a list.
        for i in range(len(corpus)):
            ngram_list.append(list(bigrams(pad_both_ends(corpus[i], n = ngram)))) # n is the n-grams, which in this case 2 is bigrams.
           
        
    if ngrams == 'trigrams':
        ngram = 3
        ngram_list = []
        # for loop that goes over every sentence in the corpus and turn it into trigrams. Every sentence is inside a list.
        for i in range(len(corpus)):
            ngram_list.append(list(trigrams(pad_both_ends(corpus[i], n = ngram)))) # n is the n-grams, which in this case 3 is trigrams.


    '''
    # Importing a library from NLTK package:
    # flatten, to combine the sentences into one flat stream of words.
    '''
    from nltk.lm.preprocessing import flatten
    
    # Turning all the bigrams or trigrams sentences lists into 1 list.
    flatten_ngram = list(flatten(pad_both_ends(sent, n = ngram) for sent in corpus))

    
    
    '''
    ##### Model Training #####
    '''
    
    '''
    # Importing a library from NLTK package:
    # padded_everygram_pipeline, to do all the previous steps in 1 step for training our model.
    # As to avoid re-creating the text in memory, both train and vocab are lazy iterators.
      They are evaluated on demand at training time.
    '''
    from nltk.lm.preprocessing import padded_everygram_pipeline
    
    # padded_everygram_pipeline return: iterator over text as ngrams, iterator over text as vocabulary data.
    train, vocab = padded_everygram_pipeline(ngram, corpus)
    
    
    '''
    # Importing a library from NLTK package:
    # MLE, to implement the Maximum Likelihood Estimator (MLE).
    '''
    from nltk.lm import MLE
    lm = MLE(ngram) # Specify the highest ngram order to instantiate it.


    # fitting the model with the iterator over text as ngrams and the iterator over text as vocabulary data.
    # The vocabulary helps us handle words that have not occurred during training by using: lm.vocab.lookup()
    lm.fit(train, vocab)


    '''
    ##### Using a Trained Model #####
    '''
    lm_counts = lm.counts # Used for counting unigrams, bigrams and trigrams.
    lm_score = lm.score # Used for providing the Maximum Likelihood Estimator for unigrams, bigrams and trigrams.
    
    return ngram_list, flatten_ngram, lm_counts, lm_score



# Function that takes a sentence and the Maximum Likelihood Estimator (MLE) of how many times two sequential words appeared and returning a table shape showing the ngram of words.
def ngram_table(sent, lm_counts):
    
    ngram_count = []
    # Nested for loop that goes over every sentence in the corpus and finds the MLE of every time two sequential words appear.
    for i in range(len(sent)):
        for j in range(len(sent)):
            ngram_count.append(lm_counts[[sent[i]]][sent[j]]) # This provides a convenient interface to access counts for ngrams.
    

    from tabulate import tabulate
    # Printing the ngram table.
    table = [[sent[0], ngram_count[0], ngram_count[1], ngram_count[2], ngram_count[3], ngram_count[4], ngram_count[5], ngram_count[6], ngram_count[7], ngram_count[8], ngram_count[9]],
             [sent[1], ngram_count[10], ngram_count[11], ngram_count[12], ngram_count[13], ngram_count[14], ngram_count[15], ngram_count[16], ngram_count[17], ngram_count[18], ngram_count[19]],
             [sent[2], ngram_count[20], ngram_count[21], ngram_count[22], ngram_count[23], ngram_count[24], ngram_count[25], ngram_count[26], ngram_count[27], ngram_count[28], ngram_count[29]],
             [sent[3], ngram_count[30], ngram_count[31], ngram_count[32], ngram_count[33], ngram_count[34], ngram_count[35], ngram_count[36], ngram_count[37], ngram_count[38], ngram_count[39]],
             [sent[4], ngram_count[40], ngram_count[41], ngram_count[42], ngram_count[43], ngram_count[44], ngram_count[45], ngram_count[46], ngram_count[47], ngram_count[48], ngram_count[49]],
             [sent[5], ngram_count[50], ngram_count[51], ngram_count[52], ngram_count[53], ngram_count[54], ngram_count[55], ngram_count[56], ngram_count[57], ngram_count[58], ngram_count[59]],
             [sent[6], ngram_count[60], ngram_count[61], ngram_count[62], ngram_count[63], ngram_count[64], ngram_count[65], ngram_count[66], ngram_count[67], ngram_count[68], ngram_count[69]],
             [sent[7], ngram_count[70], ngram_count[71], ngram_count[72], ngram_count[73], ngram_count[74], ngram_count[75], ngram_count[76], ngram_count[77], ngram_count[78], ngram_count[79]],
             [sent[8], ngram_count[80], ngram_count[81], ngram_count[82], ngram_count[83], ngram_count[84], ngram_count[85], ngram_count[86], ngram_count[87], ngram_count[88], ngram_count[89]],
             [sent[9], ngram_count[90], ngram_count[91], ngram_count[92], ngram_count[93], ngram_count[94], ngram_count[95], ngram_count[96], ngram_count[97], ngram_count[98], ngram_count[99]]]
    df = tabulate(table, headers = [sent[0], sent[1], sent[2], sent[3], sent[4], sent[5], sent[6], sent[7], sent[8], sent[9]], tablefmt = "plain")
    print(df, '\n')
             
    

# Function that takes a sentence and the Maximum Likelihood Estimator (MLE) of how many times a word appeared and returning a table shape showing the unigram of words.
def unigram_table(sent, lm_counts):
    
    unigram_count = []
    # for loop that goes over every word in the corpus and finds the MLE of every time a word appear.
    for i in range(len(sent)):
        unigram_count.append(lm_counts[sent[i]]) # This provides a convenient interface to access counts for unigrams.


    from tabulate import tabulate
    # Printing the unigram table.
    table = [[unigram_count[0], unigram_count[1], unigram_count[2], unigram_count[3], unigram_count[4], unigram_count[5], unigram_count[6], unigram_count[7], unigram_count[8], unigram_count[9]]]
    df = tabulate(table, headers = [sent[0], sent[1], sent[2], sent[3], sent[4], sent[5], sent[6], sent[7], sent[8], sent[9]], tablefmt = "plain")
    print(df, '\n')



# Function that takes a sentence and the Maximum Likelihood Estimator (MLE) of how many times two sequential words appeared and returning a table shape showing the ngram of each pair of words probability.
def probability_table(sent, lm_score):
    
    ngram_count = []
    # Nested for loop that goes over every sentence in the corpus and finds the MLE of every time two sequential words appear.
    for i in range(len(sent)):
        for j in range(len(sent)):
            ngram_count.append(lm_score(sent[j], [sent[i]]))


    from tabulate import tabulate
    # Printing the ngram table.
    table = [[sent[0], ngram_count[0], ngram_count[1], ngram_count[2], ngram_count[3], ngram_count[4], ngram_count[5], ngram_count[6], ngram_count[7], ngram_count[8], ngram_count[9]],
             [sent[1], ngram_count[10], ngram_count[11], ngram_count[12], ngram_count[13], ngram_count[14], ngram_count[15], ngram_count[16], ngram_count[17], ngram_count[18], ngram_count[19]],
             [sent[2], ngram_count[20], ngram_count[21], ngram_count[22], ngram_count[23], ngram_count[24], ngram_count[25], ngram_count[26], ngram_count[27], ngram_count[28], ngram_count[29]],
             [sent[3], ngram_count[30], ngram_count[31], ngram_count[32], ngram_count[33], ngram_count[34], ngram_count[35], ngram_count[36], ngram_count[37], ngram_count[38], ngram_count[39]],
             [sent[4], ngram_count[40], ngram_count[41], ngram_count[42], ngram_count[43], ngram_count[44], ngram_count[45], ngram_count[46], ngram_count[47], ngram_count[48], ngram_count[49]],
             [sent[5], ngram_count[50], ngram_count[51], ngram_count[52], ngram_count[53], ngram_count[54], ngram_count[55], ngram_count[56], ngram_count[57], ngram_count[58], ngram_count[59]],
             [sent[6], ngram_count[60], ngram_count[61], ngram_count[62], ngram_count[63], ngram_count[64], ngram_count[65], ngram_count[66], ngram_count[67], ngram_count[68], ngram_count[69]],
             [sent[7], ngram_count[70], ngram_count[71], ngram_count[72], ngram_count[73], ngram_count[74], ngram_count[75], ngram_count[76], ngram_count[77], ngram_count[78], ngram_count[79]],
             [sent[8], ngram_count[80], ngram_count[81], ngram_count[82], ngram_count[83], ngram_count[84], ngram_count[85], ngram_count[86], ngram_count[87], ngram_count[88], ngram_count[89]],
             [sent[9], ngram_count[90], ngram_count[91], ngram_count[92], ngram_count[93], ngram_count[94], ngram_count[95], ngram_count[96], ngram_count[97], ngram_count[98], ngram_count[99]]]
    df = tabulate(table, headers = [sent[0], sent[1], sent[2], sent[3], sent[4], sent[5], sent[6], sent[7], sent[8], sent[9]], tablefmt = "plain")
    print(df, '\n')
    


# Function that takes a sentence and the Maximum Likelihood Estimator (MLE) of how many times two sequential words appeared and returning the probability of the sentence.
def sent_probability(sent, lm_score, ngrams):
    
    sent_probability = []
    j = 0
    prob_mul = 1
    
    # for loop that goes over every sentence in the corpus and finds the MLE of every time two sequential words appear from the sentence.
    for i in range(len(sent)):
        if j < (len(sent) - 1):
            sent_probability.append(lm_score(sent[j + 1], [sent[i]]))
            if ngrams == 'bigrams':
                print('P(', sent[j + 1], "|", sent[i], ') = ', sent_probability[i])
            if ngrams == 'trigrams':
                print('P(', sent[j + 1], "|", sent[i - 1], sent[i], ') = ', sent_probability[i])
            j += 1
            prob_mul *= sent_probability[i] # Multiplication of every sentences' probability.
            
    print('\n', 'The probability of the sentence', sent, 'is: ', prob_mul)
        


# A function which takes urls from csv file and extract the text content, then write them into a txt file.
def link_to_txt():
    import csv # An import and export format for spreadsheets and databases.
    
    # Reading a csv file containing newspapers links for the corpus.
    with open('links.csv', 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        list_of_csv = list(csv_reader) # Storing the links from csv file into a list.

    
    # Accessing the lniks list containig the urls.
    webcsv = []
    for i in range(len(list_of_csv)):
        for j in range(len(list_of_csv[0])):
            webcsv.append(list_of_csv[i][j])
    
    
    # Using the web_corpus function to store the corpus obtained from the links into a txt file.
    numlink = 1
    for i in range(len(webcsv)):
        web_corpus(webcsv[i])
        print (numlink)
        numlink += 1
        


# Main function
def main():

    # Calling the link_to_txt function which takes urls from csv file and extract the text content, then write them into a txt file.
    link_to_txt()

    # Reading the txt File.
    url_data_load = nltk.data.load("url_text_raw.txt") # Used to load NLTK resource files, such as corpora, grammars, and saved processing objects.
    
    
    # Calling the tokenize_corpus function by passing the corpus as a parameter.
    word_token = tokenize_corpus(url_data_load)
    
    print("Words Tokenized")
    
    # Calling the correction function to correct the misspelled words and replace them with the correct words.
    spell_correction = correction(word_token)
    
    print('- Total sentences in the corpus:', len(spell_correction))
    
    print('-------------------------------------------- \n')
    
    
    # Calling the ngram function by passing the spell_correction and the ngrams as parameters.
    lm_counts = ngram(spell_correction, 'bigrams')
    

    # Experimenting with these words.
    sent = ['bugün', 'hava', 'iyi', 'çiçek', 'toplamak', 've', 'yürümek', 'için', 'parka', 'gideceğim']


    print('- Raw Ngram Counts of 10 Words. \n')
    
    print('*** Ngrams Counts Table ***', '\n')
    # Calling the ngram_table function to print the ngram table by passing a sentence and lm_counts as parameters.
    ngram_table(sent, lm_counts[2])
    
    
    print('-------------------------------------------- \n')
    
    
    print('*** Unigram Counts Table ***', '\n')
    # Calling the unigram_table function to print the unigram table by passing a sentence and lm_counts as parameters.
    unigram_table(sent, lm_counts[2])
    
    
    print('-------------------------------------------- \n')
    
    
    print('*** Ngram Probability Counts Table ***', '\n')
    # Calling the probability_table function to print the probability table by passing a sentence and lm_counts (score) as parameters.
    probability_table(sent, lm_counts[3])
    
    
    print('-------------------------------------------- \n')
    
    
    print('*** Sentence Probability Counts ***', '\n')
    
    
    print('Sentence 1:')
    sent1 = 'bugün üniversiteye gittim ve sonra ödevimi yapmak için eve gittim.'
    word_token1 = tokenize_corpus(sent1)
    flatten_bigram1 = ngram(word_token1, 'bigrams')
    
    # Calling the sent_probability_table function to print the sentence probability by passing a sentence and lm_counts (score) as parameters.
    sent_probability(flatten_bigram1[1], lm_counts[3], 'bigrams')
    
    
    print('\n')
    
    
    print('Sentence 2:')
    sent2 = 'bu yıl üniversiteden mezun olacağım ve çok mutluyum'
    word_token2 = tokenize_corpus(sent2)
    flatten_bigram2 = ngram(word_token2, 'bigrams')
    
    # Calling the sent_probability_table function to print the sentence probability by passing a sentence and lm_counts (score) as parameters.
    sent_probability(flatten_bigram2[1], lm_counts[3], 'bigrams')
    
    
main()