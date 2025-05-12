# nlp-exercise-2--aspect-based-sentiment-analysis-solved
**TO GET THIS SOLUTION VISIT:** [NLP Exercise 2- Aspect-Based Sentiment Analysis Solved](https://www.ankitcodinghub.com/product/nlp-exercise-2-aspect-based-sentiment-analysis-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;91100&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;NLP Exercise 2- Aspect-Based Sentiment Analysis Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
<div class="page" title="Page 1">
<div class="layoutArea">
<div class="column">
&nbsp;

1 Introduction

The goal of this exercise is to implement a classifier to predict aspect-based polarities of opinions in sentences. The classifier assigns a polarity label to every triple &lt;aspect_category, aspect_term, sentence&gt;. The polarity labels are: positive, negative and neutral.

An example of a (small) dataset containing only 2 instances

Each line contains 5 tab-separated fields: the polarity of the opinion, the aspect category on which the opinion is expressed, a specific target term, the character offsets of the term (start:end), and the sentence in which that opinion is expressed.

For instance, in the first line the opinion about the SERVICE#GENERAL aspect, which is associated to the term “wait staff”, is negative.

In the example of the second line, the sentence is the same but the opinion is about a different aspect and a different target term, and is positive.

There are 12 different aspects categories:

AMBIENCE#GENERAL DRINKS#PRICES DRINKS#QUALITY DRINKS#STYLE_OPTIONS FOOD#PRICES

FOOD#QUALITY FOOD#STYLE_OPTIONS LOCATION#GENERAL RESTAURANT#GENERAL RESTAURANT#MISCELLANEOUS RESTAURANT#PRICES SERVICE#GENERAL

</div>
</div>
<div class="layoutArea">
<div class="column">
negative SERVICE#GENERAL Wait staff 0:10 Wait staff is blantently unappreciative of your business but its the best pie on the UWS! positive FOOD#QUALITY pie 74:77 Wait staff is blantently unappreciative of your business but its the best pie on the UWS!

</div>
</div>
</div>
<div class="page" title="Page 2">
<div class="layoutArea">
<div class="column">
The training set has this format (5 fields) and contains 1503 lines, i.e. 1503 opinions.

</div>
</div>
<div class="layoutArea">
<div class="column">
File: traindata.csv

The classifier should be learned only from this training set.

A development dataset is distributed to help you set up your classifier and assess its performance. It has the same format as the training dataset.

File: devdata.csv (376 lines, i.e. 376 opinions).

We will perform the final evaluation by measuring the accuracy of your classifier on a test dataset that is not distributed. The majority class of the dev set is about 70% (positive labels), and will be used as a baseline.

</div>
</div>
<div class="layoutArea">
<div class="column">
2

</div>
<div class="column">
How to proceed

<ol>
<li>Install and use python &gt;= 3.8.x (required). Besides the standard python modules, you can use the following:
<ol>
<li>PyTorch &gt;= 1.10.0</li>
<li>HuggingFace transformers, version 4.16.x.</li>
<li>scikit-learn &gt;=0.24.0</li>
<li>pandas &gt;= 1.3.0</li>
<li>nltk &gt;= 3.6.0</li>
<li>gensim &gt;= 4.1.2</li>
<li>stanza == 1.3.0</li>
</ol>
</li>
<li>You can work on the project in groups of 4 max.</li>
<li>Download the exercise2.zip file and uncompress it to a dedicated root folder. The root folder
will contain 3 subfolders:

<ol>
<li>data: contains traindata.csv and devdata.csv</li>
<li>src: contains 2 python files: tester.py, classifier.py</li>
<li>resources : (empty) where you can put your resource files if needed</li>
</ol>
</li>
<li>Implement your classifier by completing the “Classifier” class template in src/classifier.py, containing the following 2 methods:
<ol>
<li>The train method takes a training data file as input and trains the model</li>
<li>The predict method takes a data file (e.g. devdata.csv) and should return a python list of predicted labels. The returned list contains the predicted labels in the same
order as the corresponding examples in the input file
</li>
</ol>
</li>
<li>You can create new python files in the src subfolder, if needed to implement the classifier.</li>
<li>You can train your model using gpu (assume max gpu memory == 11GB)</li>
<li>To check and test your classifier, cd to the src subfolder and run tester.py. It should run
without errors, training the model on traindata.csv and evaluating it on devdata.csv, and

reporting the accuracy measure.
</li>
<li>Please do not modify tester.py! Your program must run successfully without having to
modify this file.
</li>
<li>The exact content of the deliverable is described in section 3 of this document</li>
</ol>
</div>
</div>
</div>
<div class="page" title="Page 3">
<div class="layoutArea">
<div class="column">
3

</div>
<div class="column">
<ol start="10">
<li>Your project deliverable must be a unique zip file (a compressed folder). No gz, or other compression format.</li>
<li>The name of the zip file must consist of the family names of the authors of the deliverable. Example: Clouseau_Holmes_Velasquez.zip</li>
<li>The zip file size should not exceed 3 MBs.</li>
<li>Send the zip file by email to: salah.ait-mokhtar@naverlabs.com</li>
</ol>
Deliverable Content

</div>
</div>
<div class="layoutArea">
<div class="column">
When uncompressed, the main folder must contain the following elements:

Element Description

resources (optional): containing specific resources if used (e.g. polarity lexicons)

Note:

<ul>
<li>– &nbsp;Pleasemakesurethatwhenyoucdtothesrcsubfolderandlaunch tester.py (unmodified!) with the python interpreter, it runs without errors: it trains the classifier on the train set and evaluates it on the dev dataset, outputting the average accuracy.</li>
<li>– &nbsp;You can use any type of models, including non-deep models. However, note that classification methods based on pre-trained, transformer-based language models usually yield better accuracy.</li>
<li>– &nbsp;You can also use deep models enriched with explicit or implicit linguistic features: lemmatization, shape, POS tags, and/or parsing dependency relations using the stanza parser (c.f. lecture on parsing), polarity lexicons, etc.</li>
</ul>
</div>
</div>
<table>
<tbody>
<tr>
<td>
<div class="layoutArea">
<div class="column">
README.txt

</div>
</div>
</td>
<td>
<div class="layoutArea">
<div class="column">
A plain text file that should contain a couple of paragraphs describing:

<ol>
<li>Names of the students who contributed to the deliverable (max=4)</li>
<li>A couple of paragraphs describing your final system (type of
classification model, feature representation, resources etc.)
</li>
<li>The accuracy that you get on the dev dataset.</li>
</ol>
</div>
</div>
</td>
</tr>
</tbody>
</table>
<table>
<tbody>
<tr>
<td>
<div class="layoutArea">
<div class="column">
src

</div>
</div>
</td>
<td>
<div class="layoutArea">
<div class="column">
A subfolder containing ALL the python source files required to train and run your classifier using the unmodified tester.py : this file is used to run and evaluate your classifier.

</div>
</div>
</td>
</tr>
</tbody>
</table>
</div>
