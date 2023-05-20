REQUIREMENTS:

Python 3 (Programming Language)
nltk package (Python Library)
tabulate (Python Library)
beautifulsoup4 (Python Library)

--------------------------------------------------------------
FILES:

1) Final Project Code.py
2) ReadMe.txt
3) Final Project Report.docx
4) links.csv
5) words.txt

Note: place all these files under 1 folder.

--------------------------------------------------------------

STEPS TO RUN THE CODE:

1) Open any programming environment that supports Python (Anaconda Spyder is preferable).
2) Open the file named as: Final Project Code.py
3) If NLTK library is not installed on your device, uncomment the 3rd line of the code to install. After finishing, comment it back using "#" hashtag symbol.
   For further information you can see the official documentations on how to install: https://www.nltk.org/data.html
4) If tabulate is not installed on your device, uncomment the 4th line of the code to install. After finishig, comment it back using "#" hashtag symbol.
   For further information you can see the official documentations on how to install: https://pypi.org/project/tabulate/
5) If beautifulsoup4 is not installed on your device, uncomment the 5th line of the code to install. After finishig, comment it back using "#" hashtag symbol.
   For further information you can see the official documentations on how to install: https://www.geeksforgeeks.org/beautifulsoup-installation-python/
6) On line 23, change the path based on the location of your files.
7) Press the Run file button (F5 in Spyder) to run the code.

--------------------------------------------------------------

HOW IT WORKS:

1) After running the Final Project Code.py file, the links from Links.csv file will be taken and used to extract the texts from them and then does the writing of these
texts into a new txt file called url_text_raw.txt
2) Then the corpus from url_text_raw.txt file will be tokenized and spell corrected.
3) Lastly, it will be applied on the nltk package for Unigrams, Bigrams and Trigrams.