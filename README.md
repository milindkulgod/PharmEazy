# PharmEazy

## A Text Search Module using Python

The dataset used for this has been taken from [Kaggle](https://www.kaggle.com/jessicali9530/kuc-hackathon-winter-2018/home).
This dataset consists of patient reviews for numerous drugs that have been prescribed for the symptoms that they have experienced, and how effective they have been.

The code was coded initially on Jupyter Notebook to have a stepwise visual response of the code flow.
The code was finalized on Notepad++.

Any other text editor will do just fine as long as they support the Python libraries that are being used.

For the application, Flask was used.

**Steps to run the flask app locally**
<body>
<pre>
1.Install flask in your local environment.

2.Set the FLASK_APP environment variable to app.py and run command **flask run** </b> 

or </b>

Directly run the app.py file by the command **python app.py**

3.Open **localhost:5000** in the web browser to see and interact with the app.
</pre>
</body>



**Data Preprocessing:**


In order to create a text search module, first we need to have the appropriate dataset that we intend to work on.
On obtaining the data set, the dataset has to be processed, and brought to a comprehendable form.

Preprocessing of data of a large data set is a task, especially when it contains entries more than 100k. In the dataset that I have used, there are html codes that are present in place of special characters. For this, the unescape function of the html library has been used.

Text search will be performed  on the reviews column of the dataset, which contains long length of words. These words are separated from the string with the help of Python's split function.

In order to make the search more generalized, all characters have been converted into lowercase letters.

Next, we use the nltk library, which is probably one of the most useful libraries for natural language processing in Python. We import the stopwords function from the corpus of the module. We can utilize this to eliminate the stopwords that are present in the data. On completing this, we use the lemmatizer funtion in order to identify the root of the words, broadening the spectrum for the search. 


**Word Bank Creation:**


In order to make sure the search engine covers all the words that are present in the dataset, we have to create a word bank, which contains all the unique words that are present. The data structure that can be used for this is a python dictionary. This makes use of hash indexing, which is quick.

For each review, i.e document, we have to create a posting list, and this is possible by calculating the TF-IDF of the document.
(Term Frequency - Inverse Document Frequency).

**Query Analysis and Processing:**


We take an input from the user through a web application and process the query by calculating the word weight. On doing so, we retrieve the top k results that are required. The cosine similarity is calculated.

 If a review is not there in the first k elements, we will utilize weight in the kth element as the upper-bound on weight in the vector. thus finding the upper-bound score.
