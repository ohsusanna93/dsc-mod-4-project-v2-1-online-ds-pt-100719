Final Project Submission

* Student name: Susanna Han
* Student pace: part time
* Scheduled project review date/time: 07/05/2020
* Instructor name: James Irving
* Blog post URL: 

# Introduction

We are building a model that can predict the sentiment of a tweet based on its content. Giving insight to the manufactures of Google and Apple hardware. The tweet reviews are on the following products and services.


- IPhone
- Ipad
- Apple Apps (Iphone/Ipad)
- Other Apple product or service 


- Google
- Android
- Android Apps
- Other Google product or service 


Using a Natural Language Processing model allows us to analyze text data, which makes analyzing the score of the 9,093 product tweets possible. Finding the correlation and important features of the positive and negative feedback helps provide insight to the what makes up a positive review and negative review.


The data set we used has three columns which includes the tweet (the review), which product the review is referring to, and whether or not the review was positive, negative, or neutral. 



Below are all libraries and programs used in building our models:


```python
import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
from nltk import FreqDist
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import regexp_tokenize

from wordcloud import WordCloud
import string
import re
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from IPython.display import Image  

import warnings
warnings.filterwarnings('ignore')
```

    [nltk_data] Downloading package wordnet to
    [nltk_data]     /Users/susannahan/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!


Functions:


```python
def evaluate_model(y_test,y_pred,X_test,clf=None,
                  scoring=metrics.recall_score,verbose=False,
                   figsize = (10,4),
                   display_labels=('Negative','Neutral','Positive')):
    """
    Arguments needed to evaluate the model are y_test, y_pred, x_test, 
    the model, and display labels.
    
    Evaluate_model prints out the precision, recall, and f1-score score. As well as accuracy, 
    macro average, and weighted average.
    
    Below, a Confusion Matrix visual shows the percentage of how accurate the model fit both predicted and actual
    values. 
    
    """
    ## Classification Report / Scores 
    print(metrics.classification_report(y_test,y_pred))
    # plots Confusion Matrix
    metrics.plot_confusion_matrix(clf,X_test,y_test,cmap="Blues",
                                  normalize='true', 
                                  display_labels = display_labels)
    #plt.title('Confusion Matrix')
    plt.show()

    try: 
        df_important = plot_importance(clf)
    except:
        df_important = None
        
```


```python
def plot_importance(tree_clf, top_n=20,figsize=(10,8)):
    """ Arguments needed to plot an importance bar graph is the model, number of features to display, and 
    desired figsize for the graph. 
    
    This function displays a bar graph of top 10 important features from most to least important."""
    
    #calculates which feature was used the most in the model.
    df_importance = pd.Series(tree_clf.feature_importances_,vectorizer.get_feature_names())
    
    #sorts 20 important features data in ascending order
    df_importance.sort_values().tail(10).plot(
        kind='barh', figsize=figsize)

    #graph labels
    
    #plt.title('Top Important Features')
    plt.xlabel('Features Importance')
    plt.ylabel('Features')


    plt.show() 

    return df_importance
```

# Observations

First, we import the data and look through the dataset and make observations of what to change in the dataset to build a good model.


```python
df = pd.DataFrame(pd.read_csv('tweets.csv', encoding = 'unicode_escape'))
#importing the dataset
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_text</th>
      <th>emotion_in_tweet_is_directed_at</th>
      <th>is_there_an_emotion_directed_at_a_brand_or_product</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>.@wesley83 I have a 3G iPhone. After 3 hrs twe...</td>
      <td>iPhone</td>
      <td>Negative emotion</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@jessedee Know about @fludapp ? Awesome iPad/i...</td>
      <td>iPad or iPhone App</td>
      <td>Positive emotion</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@swonderlin Can not wait for #iPad 2 also. The...</td>
      <td>iPad</td>
      <td>Positive emotion</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@sxsw I hope this year's festival isn't as cra...</td>
      <td>iPad or iPhone App</td>
      <td>Negative emotion</td>
    </tr>
    <tr>
      <th>4</th>
      <td>@sxtxstate great stuff on Fri #SXSW: Marissa M...</td>
      <td>Google</td>
      <td>Positive emotion</td>
    </tr>
  </tbody>
</table>
</div>



One of the first observations we see is that the column names are very long and make it more difficult to recall and work with. Therefore, they were modified and renamed to tweets, product, and emotion as shown below.


```python
df.rename(columns = {'tweet_text':'tweets', 'emotion_in_tweet_is_directed_at': 'product', 
              'is_there_an_emotion_directed_at_a_brand_or_product': 'emotion'}, inplace=True)
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9093 entries, 0 to 9092
    Data columns (total 3 columns):
     #   Column   Non-Null Count  Dtype 
    ---  ------   --------------  ----- 
     0   tweets   9092 non-null   object
     1   product  3291 non-null   object
     2   emotion  9093 non-null   object
    dtypes: object(3)
    memory usage: 213.2+ KB



```python
df.shape
#shows the shape of our dataframe - 9093 rows and 3 columns.
```




    (9093, 3)



### Finding missing values.


```python
print (f'Product Missing Values:', df['product'].isna().sum())
# 5802 out of 9093 rows are missing product. 
print ('Tweets Missing Values:',df['tweets'].isna().sum())
print ('Emotion Missing Values:',df['emotion'].isna().sum())
```

    Product Missing Values: 5802
    Tweets Missing Values: 1
    Emotion Missing Values: 0


We have decided to keep all missing values in the product column to add more data to the correlation between tweet and emotion. The one missing row from the tweets column is dropped because it adds no value to the dataset.


```python
df['tweets'].dropna(inplace=True)
```


```python
df['tweets'].isna().sum()
```




    0



Now that we have taken care of the missing values we look into all the categories in each columns. In the "emotion" column we see that there are 4 categories and the " I can't tell " category only having 1% of the data. Therefore, dropping that category as it does not provide much information needed and is only a small portion of the dataset. 


```python
df['emotion'].value_counts(normalize=True)
#shows percentage of each value in column. 
#neutral - 59%
#positive - 33%
#negative - 6%
#unknown - 2%
```




    No emotion toward brand or product    0.592654
    Positive emotion                      0.327505
    Negative emotion                      0.062686
    I can't tell                          0.017156
    Name: emotion, dtype: float64




```python
#drop I can't tell from column 'emotion'
df.drop(df.loc[df['emotion']=="I can't tell"].index, inplace=True)
```

### Data Distribution


```python
df['emotion'].value_counts()
#All "I can't tell" rows has successfully been dropped from the column.
```




    No emotion toward brand or product    5389
    Positive emotion                      2978
    Negative emotion                       570
    Name: emotion, dtype: int64




```python
y= df["emotion"].value_counts()
ax = sns.barplot(y.index, y.values, palette="hls")
ax.set_title('Number of Neutral, Positive, and Negative Emotion in Data')
ax.set(xlabel='Emotion', ylabel='Number of Data')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.show();
```


![png](output_27_0.png)


We are able to see the imbalance in our dataset as there are many more neutral tweets than there are positive and negative. Therefor we are going to separate the data into it's individual dataset to extract information from what is given.

# Processing Data

After cleaning out the dataset we started to clean the text in the tweets column to properly train the model to process the given text. We use word_tokenize to separate each word and punctuation to more accurately get rid of empty spaces/words when using stopwords. 


```python
corpus = df['tweets']
```


```python
corpus[:10]
```




    0    .@wesley83 I have a 3G iPhone. After 3 hrs twe...
    1    @jessedee Know about @fludapp ? Awesome iPad/i...
    2    @swonderlin Can not wait for #iPad 2 also. The...
    3    @sxsw I hope this year's festival isn't as cra...
    4    @sxtxstate great stuff on Fri #SXSW: Marissa M...
    5    @teachntech00 New iPad Apps For #SpeechTherapy...
    6                                                  NaN
    7    #SXSW is just starting, #CTIA is around the co...
    8    Beautifully smart and simple idea RT @madebyma...
    9    Counting down the days to #sxsw plus strong Ca...
    Name: tweets, dtype: object



## Stop Words List

Stopwords is a list of common words that do not add meaning to a sentence. A 'more_punc' list was created to add to the stopwords list that were common in the data texts that didn't add any value.


```python
stopwords_list = stopwords.words('english')

more_punc = ['--',"'",'...','\\','.','%',',','sxsw','link','mention','rt','austin',
            'rise_austin','link','{','}','-','&',';','iphone','apple','ipad',
            'quot','also','marissa','google','app','#sxsw', '@mention']

stopwords_list+=string.punctuation
stopwords_list.extend(more_punc)
```

### Positive Emotion Data


```python
positive_df = df.loc[df['emotion']=="Positive emotion"].value
```


```python
positive_df[:5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweets</th>
      <th>product</th>
      <th>emotion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>@jessedee Know about @fludapp ? Awesome iPad/i...</td>
      <td>iPad or iPhone App</td>
      <td>Positive emotion</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@swonderlin Can not wait for #iPad 2 also. The...</td>
      <td>iPad</td>
      <td>Positive emotion</td>
    </tr>
    <tr>
      <th>4</th>
      <td>@sxtxstate great stuff on Fri #SXSW: Marissa M...</td>
      <td>Google</td>
      <td>Positive emotion</td>
    </tr>
    <tr>
      <th>7</th>
      <td>#SXSW is just starting, #CTIA is around the co...</td>
      <td>Android</td>
      <td>Positive emotion</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Beautifully smart and simple idea RT @madebyma...</td>
      <td>iPad or iPhone App</td>
      <td>Positive emotion</td>
    </tr>
  </tbody>
</table>
</div>




```python
pos_corpus = positive_df['tweets'].head()
```


```python
pos_corpus[:10]
```




    1    @jessedee Know about @fludapp ? Awesome iPad/i...
    2    @swonderlin Can not wait for #iPad 2 also. The...
    4    @sxtxstate great stuff on Fri #SXSW: Marissa M...
    7    #SXSW is just starting, #CTIA is around the co...
    8    Beautifully smart and simple idea RT @madebyma...
    Name: tweets, dtype: object



We used regular expression to find all urls, hashtags, retweets, and mention patterns to be replaced with '', which are all common tweet texts. 


```python
string = ','.join(str(v) for v in pos_corpus)
patterns = [r"(http?://\w*\.\w*/+\w+)", #urls
            r'\#\w*', #hashtags
            r'RT [@]?\w*:', #retweets
            r'\@\w*', #mentions
            r"(?=\S*['-])([a-zA-Z'-]+)"] #contraction words
            
clean_pos_corpus = re.sub('|'.join(patterns), '', string)
#joins all patterns and replaces them with ''.
```


```python
clean_pos_corpus
```




    ' Know about  ? Awesome iPad/iPhone app that  likely appreciate for its design. Also,  giving free Ts at , Can not wait for  2 also. They should sale them down at ., great stuff on Fri : Marissa Mayer (Google), Tim  (tech books/conferences) &amp; Matt Mullenweg (Wordpress), is just starting,  is around the corner and  is only a hop skip and a jump from there, good time to be an  fan,Beautifully smart and simple idea RT   wrote about our  iPad app for ! '




```python
pos_tokens = nltk.word_tokenize(clean_pos_corpus)
#tokenize the new corpus 
```


```python
pos_tokens[:10]
```




    ['Know',
     'about',
     '?',
     'Awesome',
     'iPad/iPhone',
     'app',
     'that',
     'likely',
     'appreciate',
     'for']




```python
pos_tokenized = [word.lower() for word in pos_tokens if word.lower() not in stopwords_list]
#make the text lowercase and remove words from the stopwords list created above.
```


```python
pos_tokenized[:10]
```




    ['know',
     'awesome',
     'ipad/iphone',
     'likely',
     'appreciate',
     'design',
     'giving',
     'free',
     'ts',
     'wait']



Then we figured out the frequency distribution for all the stopped_tokens as well as the bigrams to see the top most common words used in the text.


```python
pos_freq = FreqDist(pos_tokenized)
pos_freq.most_common(40)
```




    [('know', 1),
     ('awesome', 1),
     ('ipad/iphone', 1),
     ('likely', 1),
     ('appreciate', 1),
     ('design', 1),
     ('giving', 1),
     ('free', 1),
     ('ts', 1),
     ('wait', 1),
     ('2', 1),
     ('sale', 1),
     ('great', 1),
     ('stuff', 1),
     ('fri', 1),
     ('mayer', 1),
     ('tim', 1),
     ('tech', 1),
     ('books/conferences', 1),
     ('amp', 1),
     ('matt', 1),
     ('mullenweg', 1),
     ('wordpress', 1),
     ('starting', 1),
     ('around', 1),
     ('corner', 1),
     ('hop', 1),
     ('skip', 1),
     ('jump', 1),
     ('good', 1),
     ('time', 1),
     ('fan', 1),
     ('beautifully', 1),
     ('smart', 1),
     ('simple', 1),
     ('idea', 1),
     ('wrote', 1)]




```python
list(nltk.bigrams(pos_tokenized[:30]))
#paired words
```




    [('know', 'awesome'),
     ('awesome', 'ipad/iphone'),
     ('ipad/iphone', 'likely'),
     ('likely', 'appreciate'),
     ('appreciate', 'design'),
     ('design', 'giving'),
     ('giving', 'free'),
     ('free', 'ts'),
     ('ts', 'wait'),
     ('wait', '2'),
     ('2', 'sale'),
     ('sale', 'great'),
     ('great', 'stuff'),
     ('stuff', 'fri'),
     ('fri', 'mayer'),
     ('mayer', 'tim'),
     ('tim', 'tech'),
     ('tech', 'books/conferences'),
     ('books/conferences', 'amp'),
     ('amp', 'matt'),
     ('matt', 'mullenweg'),
     ('mullenweg', 'wordpress'),
     ('wordpress', 'starting'),
     ('starting', 'around'),
     ('around', 'corner'),
     ('corner', 'hop'),
     ('hop', 'skip'),
     ('skip', 'jump'),
     ('jump', 'good')]



### Negative Emotion Data


```python
negative_df = df.loc[df['emotion']=="Negative emotion"]
```


```python
negative_df[:5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweets</th>
      <th>product</th>
      <th>emotion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>.@wesley83 I have a 3G iPhone. After 3 hrs twe...</td>
      <td>iPhone</td>
      <td>Negative emotion</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@sxsw I hope this year's festival isn't as cra...</td>
      <td>iPad or iPhone App</td>
      <td>Negative emotion</td>
    </tr>
    <tr>
      <th>17</th>
      <td>I just noticed DST is coming this weekend. How...</td>
      <td>iPhone</td>
      <td>Negative emotion</td>
    </tr>
    <tr>
      <th>38</th>
      <td>@mention  - False Alarm: Google Circles Not Co...</td>
      <td>Google</td>
      <td>Negative emotion</td>
    </tr>
    <tr>
      <th>64</th>
      <td>Again? RT @mention Line at the Apple store is ...</td>
      <td>NaN</td>
      <td>Negative emotion</td>
    </tr>
  </tbody>
</table>
</div>




```python
neg_corpus = negative_df['tweets']
```


```python
neg_corpus[:10]
```




    0      .@wesley83 I have a 3G iPhone. After 3 hrs twe...
    3      @sxsw I hope this year's festival isn't as cra...
    17     I just noticed DST is coming this weekend. How...
    38     @mention  - False Alarm: Google Circles Not Co...
    64     Again? RT @mention Line at the Apple store is ...
    67     attending @mention iPad design headaches #sxsw...
    68     Boooo! RT @mention Flipboard is developing an ...
    92     What !?!? @mention  #SXSW does not provide iPh...
    103    Know that &quot;dataviz&quot; translates to &q...
    170    Seriously #sxsw? Did you do any testing on the...
    Name: tweets, dtype: object




```python
string = ','.join(str(v) for v in neg_corpus)
patterns = [r"(http?://\w*\.\w*/+\w+)",
            r'\#\w*',
            r'RT [@]?\w*:',
            r'\@\w*',
            r"(?=\S*['-])([a-zA-Z'-]+)"]
clean_neg_corpus = re.sub('|'.join(patterns), '', string)
```


```python
clean_neg_corpus
```




    '. I have a 3G iPhone. After 3 hrs tweeting at , it was dead!  I need to upgrade. Plugin stations at ., I hope this  festival  as crashy as this  iPhone app. ,I just noticed DST is coming this weekend. How many iPhone users will be an hour late at SXSW come Sunday morning?  ,   False Alarm: Google Circles Not Coming Now\x89ÛÒand Probably Not Ever?  {link}    ,Again? RT  Line at the Apple store is insane.. ,attending  iPad design headaches  {link},Boooo! RT  Flipboard is developing an iPhone version, not Android, says  ,What !?!?    does not provide iPhone chargers?!?   changed my mind about going next year!,Know that &quot;dataviz&quot; translates to &quot;satanic&quot; on an iPhone.  just . ,Seriously ? Did you do any testing on the mobile apps? Constant iPad crashes causing lost schedules, and no sync for WP7.,ipad2 and ...a conflagration of doofusness.  {link},You spent $1,000+ to come to SXSW. \n\n already used iPad 1. \n\nThe wait is a couple city blocks. \n\nWhy?   {}, up to 2 iPad 2s seen in the wild. Both people say it is fast, but the still pics are terrible. ,If iPhone alarms botch the timechange, how many  freak? Late to flights, missed panels, behind on bloody marys...,I meant I also wish I  at   stupid iPhone!,Found the app kyping my  geolocation &amp; not releasing when in background. Need a patch,   ,Of course Apple built a temp store in Austin.  Texas. They understand the concept of corralling cattle  ,\x89ÛÏ Apple is opening up a temporary store in downtown Austin for  and the iPad 2 launch&quot; oh YAY more traffic.,&quot;The Apple store at the mall on Sunday is 10x as crowded as this. This line is fake. I just need a fucking dongle.&quot; Genius let me in. ,iPad news apps  last  at  {link},Overheard at  interactive: &quot;Arg! I hate the iphone! I want my blackberry back&quot; ,\x89ÛÏ at : &quot;apple comes up with cool technology no  ever heard of because they  go to conferences&quot;\x89Û\x9d,overheard at MDW (and  second it) &quot;halfway through my iPhone battery already and I  even boarded the plane to &quot; ,they took away the lego pit but replaced it with a recharging station ;)  and i might check prices for an iphone  crap samsung android,. Bad Apple: shows up late, Qs the process,  ideas, leaves early. Can even be &; too creative&quot; or busy  ,Trying to balance the power of power needs on iPhone vs iPad at . This 3G iPad sucks it out quick. Might have go airplane mode.,My iPhone battery  keep up with my tweets! Thanks Apple.    ,\x89ÛÏ Best thing  heard this weekend at  &quot;I gave my iPad 2 money to  relief. I  need an iPad 2.&quot;  ,Google vs Bing on . Bing has a shot at success w/ structured search.  Potentially higher margin CPA model vs . ,iPad 2 is coming out at , guess  pretty desperate to give it attention.,Shipments daily  follow   4 updates RT   Apple Store seems to be out of  iPad2s. ,IPhone is dead. Find me on the secret batphone .,Why Barry Diller thinks iPad only content is nuts    {link} ,In Japan: Docomo introduced mobile apps six years before Apple came out with app store.  ,Jeez guys dunno about an Apple  over a  Gym. Do u realize how  that is?  ,I need to find a better  stream to follow. The inane iPad 2 tweets are surely not what SouthBy is about,Austin is getting full, and  is underway.  I can tell because my iPhone is an intermittent brick. ,Length of Google penalty based on severity of breach of webmaster guidelines. i.e.white text on white bgr might get 30 day pen.  ,\x89ÛÏ Google to Launch Major New Social Network Called Circles, Possibly Today {link} \x89Û\x9d \n never beat myspace.,Hey  how bout donating all this $ your spending on the new  to Japan. Do you REALLY need that thing?,. I have a 3G iPhone. After 3 hrs tweeting at , it was dead!  I need to upgrade. Plugin stations at .,. I have a feeling  will be the worst place to try &amp; get an iPad 2 everyone there will be trying to get one. ,I thought I would use my iPad a lot during , but I  even touched it once.  Hmmzies.,google is interested in location based tech for indoor venues  businesses, convention centers etc.  Tech needs to improve first.  ,the first iPad  even exist here last year and I already feel like  pulling out an antique everytime I use my iPad  ,shoot, my ipad will not display any search results :( will have to go through questions later  ,The  iPhone app is mocking me. Like I have the money to be there  COME ON.,The  iPhone app is one of the worst  had to use in a very long time.,Disliking iPhone twitter auto shortening links for me. ,my iPhone is overheating. why are there so many british sounding people in texas? ,My iPhone is wilting under the stress of being at .,more that just location, PixieEngine! RT  Google says the future is location, location, location: {link}  ,iPhone, I know this  week will be tough on your  battery, but so help me Jeebus if you keep correcting my curse words.,Google to Launch Major New Social Network Called Circles (Updated) {link} *Not launched at , but soon. Should I care?,Google to launch product! Wait, no launch, but product exists.  Wait, product does not exist!  {link},God,  like being at   have iMac, MacBook, iPhone and BlackBerry all staring at me. Enough! Time to read a book  remember those?,ha, seems like it &gt; \x89ÛÏ The only news at SXSWi was  temporary iPad store?  ?\x89Û\x9d,compiling my  list in one google doc is taking a lot longer than i thought... so many parties. so many good musicians.,Day 1 and my charger kicked the bucket. Where the heck is the Apple store  within walking distance? ,With just my iPhone and iPad on me, I feel very unequipped compared to everyone else. Dang! ,Hm? Do we need another 1? RT  Google to Launch Major New Social Network Called Circles, Possibly Today {link} ,Google is not launching anything during , just partying {link} ,&quot;All of  is nothing but a teeming sea of iPhone addicts too busy Twittering to ever engage with one anoth\x89Û_ (cont) {link},Not a fan of a new  trend  audience sharing opinions via holding up an iPad. Not here to listen to you. ,Tomlinson of TX Observer says Apple subscription data holding is biggest impediment to success.  , to Open  Shop at SXSW {link} &; concept but I  spend my  waiting for the ,  packing a point by showing iphone fragmentation  ,I left my pocket guide at the hotel. I  know how  going to cope. What does that say about the usability of iPad/iPhone app? ,. Problem with Google Living Stories was the process of creating content  change  was just an interface.  ,BRILLIANT READ: Attention Marketers and Media Professionals: the iPad  save you by   {link}  ,L.A.M.E.  RT  &quot;...by the law of averages, better than Buzz&quot; RT  &quot;Google Circles will be ______ &quot;  ,New  rule: no more ooing and ahing over your new iPad. We get it. Its not a big deal. Everybody has one now... :),My iPhone says it  connect to the Internet even though  wifi works great on my computer. Any suggestions?,Q: Why do social sites like Delicious often have better results than Google or Bing?  ,Just read that  will be at  selling iPad 2 on release day. Good lord, the vortex of smugness that day may be unbearable.,\x89ÛÏ The 10 most dangerous IPhone apps.  {link},So I went the whole day w/out my laptop &amp; just used my iPad 1. Things I missed: FTP, cloudapp, fast typing, &amp; skype.  ,You think there would be a  app for blackberry. Just when I gave iPhone the finger! Guess  have to carry around an ipad. LAME,People in this Apple store do not...smell great.   ,Thank you to  for letting me test drive a car to the Apple store. Now lets hope that they can fix my phone!! ,So we get to see google fail at social on another day RT  Okay, no Google Circles debuting at  today, Trajan: Google has destroyed the &lt;title&gt; tag  websites SEO them. Open Graph Protocol added a clean title tag instead ,Do i know u? Why u selling to me! Weve never talked. I now Hate your product! RT  check out Heyo 4 iPhone  ,Apple pop up store... line is ridic.   {link},Trying to update software (4.0) on iPhone to download  app. But so far no luck  wonder if  because phone is from Mexico.,Def could use an iPad for   today. Tweeting via iPhone sorta pretty much sux.,Dense una vuelta por   para ver la gran diferencia..RT  &quot;The revolution will be clumsily translated by Google.&quot;,\x89÷¼ Why Are We Better Than ? \x89÷_ {link} \x89ã_      ,if   so cocky, perhaps a separate ipad2 store at  would have been a good idea. guess they  need the extra marketing;),Disgusted with my  battery life. Already down to 11% at 3:30 pm while my blackberry is going strong. ,Just when you thought &quot;social&quot;  get more overblown at , Google may be announcing &quot;Circles&quot; today: {link},   is the classiest fascist company in America. Really elegant.  ,Lost iPad, corrupted iPhone, and crazy hotels. Good times.   AUS  {link},At midday, iPhone at . This outlet, blocked by an immobile booth, serves no purpose but to taunt me.  {link},SXSW 2011: Novelty of iPad news apps fades fast among digital delegates {link} via   XIPAD,So true!!! RT   lost its way by caring too much for the business vs. the     ,SHITE APP. :( RT  New  for  now in the App Store includes UberGuide to  sponso\x89Û_ (cont) {link},I know   because my iPhone has started lying to me about its signal strength.,  nexus s on  is 10x more useful than my iphone4 on AT&amp;T at . Is it because too many ppl have ?,... get Twitter searches to update in Hootsuite or Tweetdeck on iPad. ,    Paper phones?  Means an iPad will likely be useless at  as well. , &; worked at Google for over 11 years, so  seen a lot of evil.&quot;  ,At   iPad Design Headaches  avoiding the pitfalls of the new design challenges,Disaster = iPhone died in middle of  ...  function. heading to apple store.,The iPad 1 is so March 11, 2011 4:59PM PST. , 2011: Novelty of  news apps fades fast among digital delegates {link} via , 2011: Novelty of iPad news apps fades fast among digital delegates {link}, 2011: The  and  smackdown in all its bloody banality {link}, 2011: The  and  smackdown in all its bloody banality {link} via  ,Not even 10am here in Austin and my iPhone batt is at 54%  shit  ,Apple is a more selfish brand than Microsoft but it has served its brand well., about to talk at  on mistakes building   app,fucking mac users..... {link}  CWebb iPad Grant Hill,: iPad Design Headaches (Take Two Tablet and Call Me in the Morning) {link},Google Brain apologies for randy ads popping up with search results; algorithm skewed over weekend by randy queries from  attendees.,2+ hour wait at the makeshift Apple store for iPad 2.  , make sense to limit content to a specific platform, if  on the iPad it should be on the web  interview w/  ,Guy Kawasaki &quot;I believe in God because there is no other explanation for the continuous survival of Apple over the years.&quot;. LOL ,GroupMe talks about Apple App Store approval woes  there would be a big group of people that would agree with that!  , not all Apple love here at . Heard quite a bit of grumbling about holding back features on iPad 1 so people would have to buy v2.,Why does all the  meetups here in  are when  at work. Well at least there is the PS meetup ,  no NFC in  bc of standardization while  will have it , by an ipad 2 for the  camera, here is what you get: {link} , Android needs a way to group apps like you can now do with iPad/iPod.  ,Lonely Planet releases free Austin city guide iPhone app for . Like there  enough apps cluttering up your handset., another google social failure? ,Is there any way of deleting an app that  stop loading on the iPhone? Draining my power, but no  to cancel. Help me, .,A major  iOS update the day before ? What genius project manager came up with that brilliant idea? , Apple  store at  was 5 blocks long! ,Queue at Apple  Store at  still long!,Come on ! Why cant I have  without ? Must it be all or nothing?  , arg. It  load on me iPhone. Not via the app store or the site. Just trying to  it and am feelin so out of touch.,had to charge the  every 6hours here on . running it on lowest possible screen brightness. apple, fix your basics!,Is the Flash discussion still relevant?   iPad design headaches,Guy just asked  Mayer: it can take a year to remove deadly routes from Google Maps, such as through Death Valley ,NYT, WSJ at  ask: Is there a future for branded, native news apps on the iPad? {link} / no there is NOT!!, battery  last long under  usage,Marissa Mayer: Google maps should have better customer service, quicker responses.  ,Nothing says DOUCHE CAKE like walking around with a IPAD like  an Iphone ,really amazing to see so many people putting up with  Notes app on the iPad at ,really amazing to see so many people putting up with  Notes app on the iPad at :  {link}, Best thing  heard at  &quot;I gave my iPad 2 money to  relief. I  need an iPad 2.(  ,Too quotable &gt; RT \x89ÛÏ &quot;Apple is the most elegant fascist company in America.&quot;  ,Will somebody write an app that lets me consistently type &quot;SHIT&quot; on my goddamn iphone? Hey  ... jump on that.,Should I buy an iPad for  at the makeshift  Apple Store on launch day? Fuck no, I respect him too much... ,The walk by Lady Bird Lake was lovely, but Google Maps travel times are not to be trusted. ,The iPad calendar = The Frankeninterface. Designed as one interface metaphor but uses another.  , :  &quot;No one uses Windows voluntarily.&quot; &quot;Apple is greatest collection of egomaniacs in history.&quot; haha, :  &quot;Sell your dream. Steve Jobs  say: iPhone is $188 of parts+AT&amp;T, made by ppl in suicidal Chinese,Just launched the  Apple Store at .  our &quot;vintage&quot; store format: Mostly iPods and snarky employees. Ah, the good old days.,Stupid technology! You always fail at important times! At  w/ an iPhone, laptop AND Blackberry and the only thing working is my ., official: people using the iPad 2 to take photos just look weird. ,Pretty excited for my iPhone to stop working  &amp;T,Trying 2 convince  2 watch the launch of google Social network at  . I  know there was going to be comedy?!,The data crunch at  is crippling Google Voice. Back to regular texting we go...,True,expect more insidious approaches, like Apple subscriptions. RT  {link} . Disgraceful.   ,  when bldg app for AppStore, need wholistic mktg awareness, p.r. strategy to drive adoption  just rely on Apple.  ,I composed a tweet so acerbic and cynical about ipad users that my phone crashed and dumped all my apps.   ,Google and Bing meet at  after their lovers tiff last month but it seems to have been disappointingly friendly {link},Google and Bing page rank panel is ridiculously crowded. Is Al Franken or Justin Timberlake here?  {link},Major iPad design flaw: the SXSW Go iPad app. It  stay open when you switch apps!  ,You should probably put that away... RT  at the Android party and kinda embarrassed by my iPhone ,So Annoyed with  schedule app that has bad  for iPad. My fingers  THAT fat, .,The forbidden apple has been spoiled! Long live ! They are rocking the  &amp;  world.,is a bit disappointed that the two iPad talks had so many overlapping examples ,Fest  be an iPhone douche.Put away your phone and talk to your fellow festgoers in lines, etc.All good people who love film. ,at the Android party and kinda embarrassed by my iPhone ,Can anybody get the  mobile apps to work?  downloaded Android, iPhone and BlackBerry and they all get stuck downloading on startup.,Srsly love   promo  Srsly hate that it excludes  Esp. since my iPad insists  at Disney ,New Iphone autocorrect already tried to change &quot;coworkers&quot; to &quot;visigoths.&quot; Its going to be a long five days of ,The iPhone battery was not made for  ,New circle game? RT    Google (tries again) to launch new social network called Circles: {link} ,What Apple hopes you  notice about iPad 2 {link}     ,the future is about networks, not just data.  why google may not win long term   ,Why are we leaning on web conventions like small buttons on the iPad?  ,LOL 2 true RT   official: people using the iPad 2 to take photos just look weird. , live stream not working on iPad  {link},Hmm... the sxsw.com/interactive/live stream  ipad/mobile compatible. Maybe next year. ,Would like to know which LBS app I downloaded has commandeered my  geolocation setting &amp;  relinquish.  ,Getting ready for &quot;iPad design headaches&quot; ,I never realized how shitty by iphone battery was until   charging every few hours,Several years too late? I think the trend of social apps is over...       ,why the heck  google maps use browser ? am I missing it somehow? using chrome.   ,Why the heck would anyone want Mozilla to switch to CrapKit (WebKit) when Gecko is good? Stupid Apple/Google bandwaggoners. ,Apple cited as the opposite of crowdsourcing  proprietary, Steve Jobs tells you what you want  ,just walked by the line for the iPad 2. Bahahahaha! at least 5 blocks long.  ,Just walked past the supposed   temp store downstairs  and the apple cult was outside taking photos on iphones.,Google prefers to launch hyped new Social features with meh, not bang? via TC {link} ,Like in Vegas during , the iPhone is best used as a hand warmer at ., panel: &quot;Staying Alive: Can indie iPhone game development survive?&quot; Kind of a downer... They should try ! ,Note to self: downside of tweeting on an white iPad 2 expect to be mocked.  ,Google seems to have sabotaged my YouTube account  WTF? Are they trying to OWN the entire online ecosystem?  VERY BAD FORM ,To my friends at  who think I abandoned you, in reality I just  have any means of communication, my iPhone stopped working. ,What happened to the Taxi Magic iPhone app? Now all I can do is call the cabs via it ,Ha! RT  Google guy at  talk is explaining how he made realistic Twitter bots as an experiment. Gee, thanks for doing that.,Oh no utter   fail.  see the letter of the ballroom in the iPhone app as it is too long and  wrap /cc ,Gamechanger like Wave and Buzz no doubt RT  Google to Launch Major New Social Network Called Circles {link} ,It is never more apparent than at  how nice it would be if apple made stuff w/ removable batteries.  ,iPhone users at   any of you have your GPS stuck on? Is one of the new geo app updates doing it? FourSquare/gowalla?,Apple takes bruises from panelists. Not well regarded in sustainability space.  Corporate Sustainability Reporting and Transparency.  ,Apple likes it if you pay them.  what Apple likes.  Barry Diller ,The dailies on iPad are going to ... still impossible to download (20 MB and up downloads) ,google presentation by Mayer is a sales pitch  ,Sooo... Design for iPads session is just an hour long ad for me to buy and iPad and apps? ? ,In iPad Design Headaches: Take Two Tablets, Call Me in the AM panel  excited to hear  live! ,It was awesome to hear Marissa Mayer acknowledge that Google Maps needs to increase support for its product b/c of incorrect routes. ,The SXXpress line is almost as long as the Apple line.  {link}, is exposing my  horrendous battery life.,Novelty of  news  fades fast among digital delegates  Aron Pilhofer of the NYT &amp; Khoi Vinh {link}   ,Outside of 9:30 panels which 75% of people skip I can barely get anything to work on my iPhone.  needs to shrink by 10k people.,Novelty of iPad news apps fades fast among  delegates  by  {link},Pollak, if  having so much trouble the  sell through apple...duh  ,Turning off Twitter until  is over and  is forgotten.,kicking off w/  talking iPad design headaches ,Why is wifi working on my laptop but neither that nor 3g on my iphone? grrr. , for the life of me, I  get my iPad to sync all the sessions, only partial. iPhone is fine, iPad is deciding what it wants. ,Cab ride from hell to get to Apple store at mall. They were sold out. Getting dinner then have to figure how how to get back to hotel. ,Navigating a crowded party sucks. But its way worse when everyone walks around with their face in their iPhone.  , admits that iPhone app for Wordpress is not very good yet. Which is very true. Respect his honesty and awareness ,Best thing  heard this weekend at  &quot;I gave my iPad 2 money to  relief. I  need an  2.&quot;  , glad  not got a faulty iPhone then.  go down well for   battery dies 5 times quicker. Hope  knows.,&quot;google has too many products, and needs to condense them&quot;   (FYI  they have NINE mobile products),&quot;Google looks for technical solutions, which why  not great at community because  a human solution.&quot; ,&quot;Google to Launch Major New Social Network.&quot; really dont need another social network...{link} ,IPad Design Headaches. interface metaphor. Looks like a book, make it behave like a book. Simple stuff, but often forgotten. ,iPad design malady: iPad Elbow  I hate the  back button with the heat of a million suns.  ,I  go to  because  still using an iPhone 3G.   , good job  ! went home &amp; watched season 1 of the guild =D. sucks that your tweet abt the iphone hijack is a top tweet lol,My  Google calendar is getting a little out of control, Google Circles will be Lame.  &lt;3,I think  lost their way by caring too much about their business (instead of their users) Tim   ,Who uses Google TV in this room? Nobody raises a hand in a packed room at the  session at ,Lunch with  at . View from the HTML5 dev trenches: Android is painful, iOS is sleek (for what  is doing) ,Open graph did repair the damage google did to the title tag.  , math: if my flight leaves at 6:45 AM and the clocks go ahead at 2 and my iphone alarm will not go off what time do i miss my flight?, Ha! I feel like the only person at  with a /iPad!,iPad2 in hand thanks to the Popup Apple Store at . First impression: I may have a lemon. Backlight has some bleed thru at the bottom.,Beware, the android  app for schedules is completely innacurate. Just walked to the hyatt for no reason ,Data is the new oil. (Companies like Google and Facebook have monopoly and terms of service to be wary of)   ,Kara Swisher: Apple is the most stylish fascist company in America ,Well, Cashmore just gave the new iPad a crushing .   ,RIP my iPhone 4: June 2010   . You survived a severe drop, but could not evade drowning. {link},95% of iPhone and Droid apps have less than 1,000 downloads total. ,Tomorrow I go back to the Apple store... *sigh* , I hate typing on an iPad. So, yeah, THE Ken Calhoun is this man, the real deal! And I am bringing my laptop to  Sun., I hope this  festival  as crashy as this  iPhone app. , I meant iTunes  work for me (IE:  run Apple software, even if it would run on my Ubuntu desktop).  Not just  dl, I outdid myself this time with the tech . iPhone=broken. Might be the worst thing that can ever happen at ,  glad I  have an iPad 2 ... I think. ,  going to   meet up! I  use  yet because I have an Android phone but show me how it works!,  guessing there will not be an  app in time for ?,Totalitarian thought in action: People worldwide come to Austin, TX for  and decide to spend their time in line at an Apple Store.,It is ridiculous to see someone taking a photo during a session with their iPad. Cannot wait to see concert use.  , If I were you,  stay away from the Apple Store tomorrow , if it makes you feel better  fanbois were acting like giggling schoolgirls in front of Marissa Mayer  ,Related: if u have Verizon, u will do better.  bandwidth issue. My BlackBerry always tweeted when my iPhone dawdled. , design/UI tip: \x89ÛÏbuttons are a hack  them with skepticism \x89ÛÏ  ,Farmers like Blackberry smart phones because  &; last long on the farm&quot;.  ,Tablets like the iPad and Xoom where touch emulates a /keyboard input means  not there yet.   ,Techie Fail RT  One  panel moderator who is from Europe says he spent $3000 in roaming charges on his iPhone at SXSW 2010.,Content will move back to the browser. Why replicate the work on an iPad when u can press one button?   ,Mad mad lines still at the  Apple pop up store here in Austin , thinking that I may actually have to take my laptop to . The iPad alone may not be enough on this trip.,Been playing w/ the Windows7 phone here at .The animation is better than an iphone&amp; it does look very interesting!, Instagram, but  iPhone only at the moment mister &quot;Has to be cool and different and get an EVO&quot; that SUCKS. ,Hey  got invited to a new group at  and your Android app keeps crashing when I try to join! WTF? ,hey  twitter needs a way for us in disaster areas to filter out things like  and  too because right now we really  care,Hey  your app  download from the Android app market. Just a heads up ,Guy with iPad 2 taking photos of  slides. Awkward! , iPad app getting panned for design trumping content, and rightfully so. ,Fuck the iphone! RT  New  for  now in the App Store includes UberGuide to  ... {link}, is about to talk about the mistakes he made building Netflix for the iPhone.  ,Bad news is it costs $1,000? RT  Louis Vuitton has an iPhone app. Called Amble  ,My row this morning two people hand writing notes, one person on a Blackberry and me on a   proof Apple  a monopoly at ,Many  iPad content enhancements are like th worst David Foster Wallace  kinda interesting but mostly irrelevant. ,iPhone crisis at  Phone is stuck on silver logo screen. Has been for last 15 minutes. Help?,I cant wait to give the SAMSUNG people a demo of my horrible, terrible Google Nexus S phone at ,And now, it  pull from website to iPhone. Awesome. Glad I wasted that time.   ,You know  bad with you have to  just to see/use your calendar.    ,  the next step of Google becoming Skynet. ,The Netflix iPhone app was built to a &quot;ridiculous deadline&quot;. Big mistake on the  part to agree to it ,First talk of the day : iPad design headaches. ,Josh Clark:  I hate the  back button with the heat of a million suns.   ,Google takes the mantra of  be  to heart, yet their pride in their products make them guilty of the 7th deadly sin, Superbia ,Why is all the prizes at  apple products? Sigh,Cant handle  traffic? RT  Dear  your own app for iPhone has sucked all day. ,This double buzzing issue with iPhone iOS 4.3 is getting annoying. In other news, an iPhone may fly across the room at  ,&quot;Apple is the most elegant fascist company in America.&quot;  ,Have Google launched their next social media flop yet? ,&quot;Apple likes it if you pay them.  what Apple likes.&quot;  Barry Diller  ( ACC  Ballroom D) [pic]: {link}, apple store run out for the day :( boo apple.,Dear google, your photobooth sucks. ,&quot;Apple: the most elegant fascist corporation in America today.&quot;  Kara Swisher  ,About to learn all about design headaches for iPad ,Barry Diller says  magazines like The Daily  make sense, {link} ,Barry Diller says that  silly if you write content for one form factor only. Apple like it because they get money ,Barry  pragmatic and straight forward which is nice. Looks like he wants us to grab pitchforks vs. Apple , Go  you rate &amp; review sessions (although the experience on the iPhone is a bit torturous).,Google will eat itself      Hilton {link},Google will not launch Facebook competitor Circles   today as thought, creeper  the world over disappointed.,And it will suck. RT  RT  Google will preview major new social service, Circles, at  today {link},Look at all these people at  with iPad 1. No shame.,I feel silly but cannot figure out how to update the  mobile app on android. It  seem to have &quot;options&quot; (bottom left button)?,I was really hoping  would bring an Android version of  Oh well, maybe someday...a long time from now.,Will overload of info delivered by Google kill discovery? Google says ppl  lose curiosity, but I wonder...We need serendipity. , love what  do at . How come  getting login authorization error almost every time I open my app on iPhone?,After failure of Google buzz, Google Latitude, now Google Circle! Seriously Google needs to concentrate more on search!   ,You finally get everyone to buy in to Facebook and then Google introduces Circle. No fair. Stop with all the innovation, people ,You finally get everyone to buy in to Facebook and then Google introduces Circles. No fair. Stop with all the innovation, people , Might need to go to Apple today. I think my &quot;S&quot; &quot;X&quot; and &quot;W&quot; keys are worn out. ,just got mine &amp; i disagree RT   Peter Cashmore on iPad 2  only a minor improvement Not worth it unless  $ 2 burn ,Nope, seems no Google Circles launch today: {link} , my  iPhone app has been down for a few days. Heading to  and want to tweet a lot  can you help?, my iPhone crashed &amp; I had to do a fresh restore &amp; lost my fave Dali/canvas pak? Can I ever get it back? Also are you  ?,Ahhh, darn :( RT    according to  Google has confirmed it is not launching at , if at all.,Please can I hear more people talk about the iPad 2, preferrably with more  hashtags   I really have a deficit of this in my life!,this woman is great  fessing up to issues, admitting google could do better ,Sitting on the floor behind a guy  fondling his new iPad 2 in a very disturbing way. ,Grrr..plancast stuff exported to Google calendar in San Diego/pacific time did not shift to central tme  ,line around the corner for  at , i say wait on it.  already got two cameras on my  sorry ,Bereft wanderer. White cord, limp. Lifeless. There is no outlet for your iPhone here.  ,Man panhandling for an iPad 2 at SXSW. What\x89Ûªs the world coming to?    ,This iPhone  app would b pretty awesome if it  crash every 10mins during extended browsing.  ,SXSW iPhone app is awesome, but iPad app crashes every time. ,Does anyone have an address for the yeasayer concert tonight? Google maps is failing. ,I asked one of the booth people if  paid every single booth here to show their product on iPad2s. She said no.  ,the internet blurs, {link} the iPad fades, {link} \x89ÛÒ  at ,interesting &quot; &quot;Google looks for technical solutions which why  not great  community because its a human solution \x89Û\x9d,My tweeting from  been pretty non existent today thanks to the  iPhone app   Hootsuite is the epic replacement! ,I was going to mock the  for coming to  only to stand in a 4 line for an iPad 2. Then I remembered me and horror movies.,Is starting to think my  is more like the  of phones. Damn u .  (I just wanted  abacus), Nice  song   Don\x89Ûªt hack, Write Rails. Under deliver, over sell. Work for Google. {link}, No, no, no .. I  buy an iPad 2.  wait for the iPad 3 with teleporting capabilities. ,This technology  happen on an   Nokia  ,There is nothing sillier than watching people record video with an ipad. ,Agreed \x89ÛÓ Novelty of  news  fades fast among digital delegates \x89ÛÓ {link}    (via ,In the  &quot;Worst Company in America 2011&quot; tournament bracket  apple is pitted against microsoft in the first round ,I fully anticipate that every  will be toting an iPad 2 at . These people are also why  not worth going.,I am inventing a &quot;dislike&quot; button for  2 lines. {link} ,  das Verpixelungsrecht\x89ÛÓthe right for your house to be pixelated in Google Street View\x89ÛÓ is a theft from the public,Grrrr  not muting  as thought on web or iphone :(,Organic unveils BroadFeed, social news app for ipad. Audience gets up and heads for the door. ,Mayer also admits that Google needs to &quot;step up&quot; its customer support for Google Maps. , to expensive mobile data plans are killing the flavor of contextual discovery abroad  ,Decided to go to LA instead of , because my AT&amp;T iPhone would be about as useful as a brick in Austin.,Heading to iPad Design Headaches in Hilton  Salon J ,Excited to meet the  at  so I can show them my Sprint Galaxy S still running Android 2.1.   ,iPhone too old for  schedule app. Oh noes.  , no way that you an call the iPad count at  reasonable. I think  one for every 2 people.,Apple autocorrect is so weird. Ogilvy autocorrects to idiocy.   :),Google product showcases never feel that cool. No price tag, brand equity, wow factor attached.  marissa mayer,Cashmere of Mashable: Thinks the iPad 2 is not a huge step up. Now waiting for the Apple PR team to parachute into the  session.,Andrew K of PRX equates the homogeneity of the Apple ecosystem w predictability, vs  wild west.    ,Aron Pilhofer of the New York Times and design guru Khoi Vinh express scepticism about iPad news apps at  {link},RT   lost its way by caring too much for the business vs. the     ,RT   technology  happen on an   Nokia  ,MT  (Swisher calls Apple &quot;the classiest fascist company in America&quot;  ,RT     False Alarm: Google Circles Not Coming Now\x89ÛÒand Probably Not Ever?  {link}    ,RT   iPad app getting panned for design trumping content, and rightfully so. ,RT   Peter Cashmore on the iPad 2:  only a minor improvement. Not worth it unless you have money to burn. ,RT  &quot;   is the classiest fascist company in America. Really elegant.&quot;  ,RT  &quot;Apple is the most elegant fascist company in America.&quot;  ,RT  &quot;Apple: the most elegant fascist corporation in America today.&quot;  Kara Swisher  ,RT  &quot;Google looks for technical solutions, which why  not great at community because  a human solution.&quot; ,RT  &quot;I believe in God because there is no other explanation for  continued existence.&quot; Guy Kawasaki  ,RT  &quot;multiple approaches to monetization&quot; re: iPhone game dev &quot;but ads would cheapen our product&quot; ok, good luck with that ,RT  &; a reason why Google  in social  they are too technical.&quot;  ,RT   Trajan: Google has destroyed the &lt;title&gt; tag  websites SEO them. Open Graph Protocol added a clean title tag instead ,RT    reporting: Janecek: Microsoft gives $ to charity. Apple gives nothing. Everyone in room has iPhone. What drives that decision?,RT   :  &quot;No one uses Windows voluntarily.&quot; &quot;Apple is greatest collection of egomaniacs in history.&quot; haha,RT   2011: The Google and Bing smackdown in all its bloody banality (Guardian) {link} via ,RT   ipad store sold out of everything except 64gig wifi only white,RT   rumor mill: iPad 3 will have between 6 and 15 cameras, slightly thinner rare earths case, and &quot;different&quot; but still smudgy screen.,RT   WAZE {link} is duking it out with google re: personalized mapping experience. Friendly &quot;panel crashing&quot;,RT  : iPad Design Headaches (Take Two Tablet and Call Me in the Morning) {link},RT    is glad there are no standard  navigation tools. She might be the only one! ,Diabetes on a plate. Thanks Google, already have that covered. {link} ,RT  Agreed \x89ÛÓ Novelty of  news  fades fast among digital delegates \x89ÛÓ {link}    (via ,RT  And it will suck. RT  RT  Google will preview major new social service, Circles, at  today {link},RT  Anyone who was going to buy a new iPad should donate to   victims instead. ,RT  Apparently, if you Google &quot;ad preferences&quot; and  see what Google thinks  like.   ,RT  Apple autocorrect is so weird. Ogilvy autocorrects to idiocy.   :),RT  Apple is &quot;the classiest, fascist company in America,&quot; says  ,RT  Apple...&quot;the classiest fascist company in America&quot; Kara Swisher ,RT  Best thing  heard this weekend at  &quot;I gave my iPad 2 money to  relief. I  need an iPad 2.&quot; (,RT  Best thing  heard this weekend at  &quot;I gave my iPad 2 money to  relief. I  need an iPad 2.&quot; ( ,RT  Best thing  heard this weekend at  &quot;I gave my iPad 2 money to  relief. I  need an iPad 2.&quot; ,RT  Best thing  heard this weekend at  &quot;I gave my iPad 2 money to  relief. I  need an iPad 2.&quot; &lt;Amen!&gt;,RT  Best thing  heard this wknd   &quot;I gave my iPad 2 money 2  relief. I  need iPad 2&quot;( ),RT  Brought up how Google Maps had rerouted all images of JCPenney to images of Macys or trashy restraunts. They had no comment  ,RT  Decided to go to LA instead of , because my AT&amp;T iPhone would be about as useful as a brick in Austin.,RT  Deleting the  iPhone app!  {link},RT  DELICIOUSLY IRONIC GOOGLE PRIVACY PARTY MADE WHOLE BY &quot;BANKING CARTEL,  DICTATORSHIP TAKEOVER&quot; CAB RANT! ,RT  Diller on Google TV: &quot;The first product  good. It  a consumer product, basically.&quot;  ,RT  Epic.  just one guy waiting in line for the iPad 2 in Austin at SXSW. {link}   ,RT  False Alarm: Google Circles Not Coming Now, And Probably Not Ever {link}  so much for reports that it would unveil at ,RT  Fest  be an iPhone douche.Put away your phone and talk to your fellow festgoers in lines, etc.All good people who love film. ,RT  Finally fed up with , Julian screamed &quot;I got your  zone right here, pigfucker!&quot; and threw his iPad at some kid. ,RT  Fodder for  panel at : &quot;Right to Forget&quot; gains traction in Europe, causing issues for Google {link},RT  forward to delicious  4G here in Austin while iPhone users struggle to do anything. ,RT  Found the app kyping my  geolocation &amp; not releasing when in background. Need a patch,   ,RT  Google Circles will be toast if it  convince facebook users to start an account (or interface with FB)   ,RT  Google  rate restaurants and get personalized recos on where to eat. Um, think foursquare, yelp, etc have this covered already. ,RT  Google Latina and see what you find? Porn...this is the first impression that people get about us?    \x89Û\x9d,RT  google presentation by Mayer is a sales pitch  ,RT  Google to Launch Major New Social Network Called Circles, Possibly Today at  rww.to/f6BCEt |  more overload,RT  Google vs Bing on . Bing has a shot at success w/ structured search.  Potentially higher margin CPA model vs . ,RT  Google was incapable of doing disruptive  and acquired 89 startups over the last few years   ,RT  Hipsters w/ oversized earphones pluged into the new IPAD make me happy that Texas has such loose gun laws ,RT  Hm? Do we need another 1? RT  Google to Launch Major New Social Network Called Circles, Possibly Today {link} ,RT  Hmm... the sxsw.com/interactive/live stream  ipad/mobile compatible. Maybe next year. ,RT  I cant wait to give the SAMSUNG people a demo of my horrible, terrible Google Nexus S phone at ,RT  I feel like my iPhone: Always on, always doing something, running out of battery fast. ,RT  I have yet to walk into a conference room where it  look like an Apple ad.  think there was nothing else. ,RT  I know   because my iPhone has started lying to me about its signal strength.,RT  I think  lost their way by caring too much about their business (instead of their users) Tim   ,RT  I think my effing hubby is in line for an  2. Can someone point him towards the  for wife number .  ,RT  If you have an iPad DO NOT upgrade to the newest iOS yet, TweetDeck is very unstable on it    ,RT  In a room full of geeks talking TV,  just asked, &quot;Who uses Google TV?&quot; No one raised hand.  ,RT  iPad design headaches with Josh Clark    {link},RT  iPad design malady: iPad Elbow  I hate the  back button with the heat of a million suns.  ,RT  Ironic? I googled the directions to  party and ended up walked 6 blocks in the wrong direction. Time for bed I think. ,RT  It is ridiculous to see someone taking a photo during a session with their iPad. Cannot wait to see concert use.  ,RT   official: people using the iPad 2 to take photos just look weird. ,RT  Josh Clark:  I hate the  back button with the heat of a million suns.   ,RT  Just read that  will be at  selling iPad 2 on release day. Good lord, the vortex of smugness that day may be unbearable.,RT  Just saw someone take a picture with an iPad 2 for the first time. Looks as ridiculous as  expect.  ,RT  Line for Source Code is even longer than for iPad 2. Take that, Apple. ,RT  Lonely Planet releases free Austin city guide iPhone app for . Like there  enough apps cluttering up your handset.,RT  New  rule: no more ooing and ahing over your new iPad. We get it. Its not a big deal. Everybody has one now... :),RT  New Iphone autocorrect already tried to change &quot;coworkers&quot; to &quot;visigoths.&quot; Its going to be a long five days of ,RT  Not a fan of a new  trend  audience sharing opinions via holding up an iPad. Not here to listen to you. ,RT  Note to self: downside of tweeting on an white iPad 2 expect to be mocked.  ,RT  Nothing says DOUCHE CAKE like walking around with a IPAD like  an Iphone ,RT  Novelty of iPad news apps fades fast among  delegates  by  {link},RT  Part of Journalsim is the support of democracy, yes? Informed populous, yes? iPad, as a focus, does not support that  ,RT  RT    to Launch Major New Social Network Called , Possibly Today! {link}   &gt;&gt;Really Google? Now?,RT  RT  Apple is &quot;the classiest, fascist company in America,&quot; says  ,RT  RT  forward to delicious  4G here in Austin while iPhone users struggle to do anything. ,RT  RT  Sound of My Voice was shot exploiting Apple &amp; Best  14 return policy on iMacs. Brilliant. ,RT  Slides from  talk, &quot;iPad Design Headaches &quot; {link}  . Really interesting for designers as well.,RT  speaks truth: Watched  staff at the temp  store just high five entire long line and facepalmed. Ugh. ,RT  SXSW 2011: Novelty of iPad news apps fades fast among digital delegates {link} ,RT  Temp Apple Store 6th St  Austin  closed for night &; out of  says Austin cops barring the door!,RT  Temporary  apple store. Apple being sneaky as usual  {link},RT  the future is about networks, not just data.  why google may not win long term   ,RT  The iPad 2 is the also a cartoonishly large digital camera.   {link},RT  This group next to me has 6 ppl  the table. Everyone is using thier phone/ipad instead of taking to each other. ,RT  Who uses Google TV in this room? Nobody raises a hand in a packed room at the  session at ,RT  Will overload of info delivered by Google kill discovery? Google says ppl  lose curiosity, but I wonder...We need serendipity. ,RT  Woman in lobby: &; a website called, like, stupid iPhone speller, and  ppl taking pics of funny autocorrected words.&quot; ,RT  yes, i hate the fanboys. still valid  to those that waited for the iPad2 instead of enjoying : {link}, iPhone app: control mania! Half of the screen used for buttons and filters, other half for content. ,Fuck. iPhone crapped out, will not charge at all. Says it is charging, but at 5% after all night. What are my options in Austin? ,Damn it Google! Your  cup leaked  goo in my camera bag! ,It just looks stupid to take pix with an iPad 2  believe me guys  ,Apple is opening a temporary store in Austin for . Because people obviously want to shell out $1500 for a MacBook while visiting Austin,Apple is opening a temporary store in Austin for . Because people obviously want to shell out $1500 for a MacBook.,Google dropped real estate search b/c to do it right,  need to invest much more than  willing to at the moment. ,NYT app for iPad: not &; an amazing way to serve our readership,&quot; more &; a market opportunity we  ignore.&quot;  ,Looking forward to delicious  4G here in Austin while iPhone users struggle to do anything.  ,Looking forward to the day when  and  release native Android 3.0  clients! Google Latitude sucks! ,Both Microsoft and Google are ripping us off too {link} :(  a Joomla! wannabe logo  {link},Yea!!! Another Google product that  available for Google Apps yet: Google Hotpot. ,&quot;No other reason for  continued survival than the existence of God&quot;      ,Twitter you have failed me for the last time *force chokes iPhone* , a tougher crowd than Colin Quinn RT  Hey Marissa Mayer Please tell us something new Not about products Google launched months ago,But are your phones staying charged? My crappy  would never handle ! RT    no sleep crazyfest!!!!!!,Part of Journalsim is the support of democracy, yes? Informed populous, yes? iPad, as a focus, does not support that  ,Checking out iPad Design Headaches (2 Tablets, Call in the Morning).\n{link}   ,Brought up how Google Maps had rerouted all images of JCPenney to images of Macys or trashy restraunts. They had no comment  ,Put the iPad down idiot!  ,Just stopped at the pop up Apple store.....lame!z!  Come on Apple you can still cut the weight in half. , Peter Cashmore on the iPad 2:  only a minor improvement. Not worth it unless you have money to burn. , Apple Store seems to be out of  iPad2s. ,Clearly Apple has another daylight savings time bug with the iPhone 4 alarm. Also need to remember to fix my  clocks. Whoops. ,Why the Apple  shop at  sucks: {link},Ughhh...Apple Store Austin at the Domain rain out of 3G iPads in first 25 minutes right before I got in  ,All LBS apps on my iPhone think I am in Orlando :)) ! I guess I have 2 check in at Disney vs  :)),Attended preso on living simply  at   and felt a bit guilty taking notes on a new ,Apple...&quot;the classiest fascist company in America&quot; Kara Swisher ,Is twitter broken? Or is it my iPad.  follow the SXSW hash right now. Wow, that might be my nerdiest tweet yet.  .,Okay, fair enough lining up for new tech, but not at the cost of ... It is so much better than a new iPad... {link},Headed for iPad Design Headaches (2 Tablets, Call in the Morning)  {link},Design for iPad is like a design 101 class. Will someone give a talk and assume that we  all ditch our previous experience ,  reporting: Janecek: Microsoft gives $ to charity. Apple gives nothing. Everyone in room has iPhone. What drives that decision?,I can hear those ipad 2 nutters ipading away even now. Such is life in the techno hipster Mecca. ,Apps distract pubs, sez Khoi Vinh. Instead of focusing on reader exp,  delivering same content 3 ways.   {link},How frustrating is it that Zeldman autocorrects to Zelda on the iPhone?  ,Google Circles is (not) a real thing and will (not) be launched today at  {link},&quot;There is no other explanation for  continued survival than the existence of God&quot;  ,&; a reason why Google  in social  they are too technical.&quot;  ,Enough already .... \x89ÛÏ Google to Launch Major New Social Network Called Circles, Possibly Today {link} \x89Û\x9d, really disappointed with the iPad app  lots of error messages have to switch to tweet deck for the rest of ,Farmers prefer Balckberries over iPhones and Andriod devices. More durable?  feelings?   ,Horrible repressed memories of the Apple spinning beach ball coming back at the  talk. , Respectfully disagree about the iphone. Battery life is a problem and it  as ubiquitous as it seems.  ,   Tried 2 days with iPAD, w/o MacBook Pro.  the experiment is over. I heart a real keyboard.,Google (tries again) to launch a new social network called Circles: {link}   ,False Alarm: Google Circles Not Coming Now\x89ÛÒand Probably Not Ever?  {link}    ,iPad 2 turned into giant camera looks plain stupid ,Why is  not available in Canada for iPhone users?   Would love to check it out.,Or Droid RT  Has spontaniety in life been replaced by technology? When your iPhone battery dies you go home. Great ?  ,Will I survive  will only 1.6 gigs of space left on my iPhone? We shall see.,Guess everyone wanted to design ipad apps for their mom ,Steve Jobs  position the iPhone as a device made in China where suicide rates are high He sells dreams   ,&quot;This Google/Bing Q&amp;A panel is like the  most expensive SEO consultation.&quot;   ,Google was incapable of doing disruptive  and acquired 89 startups over the last few years   ,My friends iphone changed BACK an hour instead of forward. Its AT&amp;T. Any hints on how to fix it?!   ,iPhone GPS is messed up. Thinks  in Yonkers. Almost as good as . {link}, does the world really needs Google Circles, Fb will just implement it eventually,Hey Marissa Mayer. Please tell us something new. Not about products Google launched months ago. ,Anyone else having issues with gmail and using google services? .,Anyone else having trouble synching your schedule with the android sxsw go app? ,Anyone else with an iPhone getting the &quot;could not activate cellular data network&quot; error message? Is this a warning of  things to come?, set to fail for being too complicated by June. RT  Google set to launch new social network  today at , Should: if you  get iPad 1. : iPad2 3G 64GB + cover + $100 = MacBook Air. ,\x89ÛÏThe classiest fascist company in existence\x89Û\x9d  Kara Swisher  ,Spending some time this morning resetting my android phone.  First day of  was too much for it.,Just saw someone take a picture with an iPad 2 for the first time. Looks as ridiculous as  expect.  ,Every person I see with a ipad looks crazy ,looks like they have a 80 cell phone in their hand  ,What a dissapointment!!! RT  New  for  now in the App Store includes UberGuide to  sponsored by (cont\x89Û_,Google possibly  launching a social media service  again...  {link},Google lost its way by caring too much for the business vs. the     ,: Mistakes Made Building  for iPhone (Plus, How to See Its Source Code!) {link},: Mistakes Made Building Netflix for iPhone (Plus, How to See Its Source Code!) {link} via ,Opened yesterday. Line too long to wait in just to look.    Apple Store, SXSW {link},Yet the    store was flawless. RT  Apple iPad 2 Launch: Poor Planning On Apple\x89Ûªs Part {link},Massive  fail to run out of   Should have had a semi full parked out front!  {link},I think  has taken it upon itself to make it clear to me that a Gen 1 iPhone  gonna cut it anymore.,My biggest frustration with  so far is no one looks where  going because glued to iPhone.,Is it just me or has the  client for Android gotten really buggy lately?  to blame?,Off to get my badge. Then to find food and drink. Then figure out why my  iPhone is NOT roaming at . Then unpack. Priorities,Proof that the iPad 2 turns you into a douchebag  {link},Looking at the line for the pop up  apple store...I  think of a single object I want that much.,I have yet to walk into a conference room where it  look like an Apple ad.  think there was nothing else. ,  up with the sxsw go app?  faulty. :(,tried installing  on my iphone but it crashes every time i open it. ,I just watched &quot;iPad Design Headaches&quot; at . Buttons are a hack.  ,Not to hate on the iPad, but fleets of nerds armed with iPads navigating through crowds is as far from cool as  EVER seen ,So,  for iPhone: anyway to remove/burn hashtags? Eg ? V v bored already, already,Deleting the  iPhone app!  {link}, the apple   is like a crack house for mac addicts,Apple is the classiest facist company in America.  Kara Swisher ,Oh no another way to talk shit on the net Google to Launch Major New Social Network Called Circles Possibly Today {link} ,  journalists. They just spent all their money on  passes. Who can afford Google TV?,I just noticed DST is coming this weekend. How many iPhone users will be an hour late at SXSW come Sunday morning?  ,Google Hotpot  &quot;Not as good as other services, but we will force it on you anyways.&quot; ,Perfect attention to detail RT  Google recreated the code for  for their doodle, original bugs included.  ,iPhone battery is going quickly. Guy behind me let me borrow his portable charger. I want one! {link} ,wow you suck {link} iPad  Rebecca Black , so ungrateful bc we have too much shit to play with. So turn off some of your shit (iPhone) now and then. ,Bad news update: the  Apple Store is out of iPads! Not sure if they will have more by tomorrow. ,Just because google patented something i.e. (Age of domain in rankings algorithm)  mean they use it  ,If you have an iPad DO NOT upgrade to the newest iOS yet, TweetDeck is very unstable on it    , have to wait 45 weeks for an iPad 2, but not at the  ! Shipments daily (via  ,ProTip: Avoid the  Apple stores on Friday , voxpop of  popular  apps is worth a watch: {link} Not many Android phones on ., know 1 user RT  Who uses Google TV in this room? Nobody raises a hand in a packed room at the  session at ,Dear  goer... Please look up from your fucking iPhone when walking the halls. Thanks Hipsters. Hilarious!,This  I am grateful for: my bicycle, having a  Twitter app. Cursing: losing an hour of zzzs, iPhone battery life.,Dear  iPhone app: you suck again this year! (sitby.us is great but  include film sessions),If there was a popup store in Austin that sold nothing but iPhone battery extenders, it would make so much money. ,   not popular with the . {link}  is a terrible concept anyway ,Hmmm...Taxi Magic on iPhone does not appear to be so magic any more in Austin ,Google guy at  talk is explaining how he made realistic Twitter bots as an experiment. Gee, thanks for doing that.,I think my effing hubby is in line for an  2. Can someone point him towards the  for wife number .  , pretty sure the panelist that thinks &quot;Apple is drowning in their success&quot; is fucking insane. ,Hey is anyone doing  signing up for the group texting app, groupme? got it on my iphone, but no one else is on it, so....kinda useless.,Diller says Google TV &quot;might be run over by the PlayStation and the Xbox, which are essentially ready today.&quot;  '




```python
neg_tokens = nltk.word_tokenize(clean_neg_corpus)
```


```python
neg_tokenized = [word.lower() for word in neg_tokens if word.lower() not in stopwords_list]
```


```python
neg_tokenized
```




    ['3g',
     '3',
     'hrs',
     'tweeting',
     'dead',
     'need',
     'upgrade',
     'plugin',
     'stations',
     'hope',
     'festival',
     'crashy',
     'noticed',
     'dst',
     'coming',
     'weekend',
     'many',
     'users',
     'hour',
     'late',
     'come',
     'sunday',
     'morning',
     'false',
     'alarm',
     'circles',
     'coming',
     'now\x89ûòand',
     'probably',
     'ever',
     'line',
     'store',
     'insane..',
     'attending',
     'design',
     'headaches',
     'boooo',
     'flipboard',
     'developing',
     'version',
     'android',
     'says',
     'provide',
     'chargers',
     'changed',
     'mind',
     'going',
     'next',
     'year',
     'know',
     'dataviz',
     'translates',
     'satanic',
     'seriously',
     'testing',
     'mobile',
     'apps',
     'constant',
     'crashes',
     'causing',
     'lost',
     'schedules',
     'sync',
     'wp7.',
     'ipad2',
     'conflagration',
     'doofusness',
     'spent',
     '1,000+',
     'come',
     'already',
     'used',
     '1',
     'wait',
     'couple',
     'city',
     'blocks',
     '2',
     '2s',
     'seen',
     'wild',
     'people',
     'say',
     'fast',
     'still',
     'pics',
     'terrible',
     'alarms',
     'botch',
     'timechange',
     'many',
     'freak',
     'late',
     'flights',
     'missed',
     'panels',
     'behind',
     'bloody',
     'marys',
     'meant',
     'wish',
     'stupid',
     'found',
     'kyping',
     'geolocation',
     'amp',
     'releasing',
     'background',
     'need',
     'patch',
     'course',
     'built',
     'temp',
     'store',
     'texas',
     'understand',
     'concept',
     'corralling',
     'cattle',
     '\x89ûï',
     'opening',
     'temporary',
     'store',
     'downtown',
     '2',
     'launch',
     'oh',
     'yay',
     'traffic.',
     'store',
     'mall',
     'sunday',
     '10x',
     'crowded',
     'line',
     'fake',
     'need',
     'fucking',
     'dongle.',
     'genius',
     'let',
     'news',
     'apps',
     'last',
     'overheard',
     'interactive',
     'arg',
     'hate',
     'want',
     'blackberry',
     'back',
     '\x89ûï',
     'comes',
     'cool',
     'technology',
     'ever',
     'heard',
     'go',
     'conferences',
     '\x89û\x9d',
     'overheard',
     'mdw',
     'second',
     'halfway',
     'battery',
     'already',
     'even',
     'boarded',
     'plane',
     'took',
     'away',
     'lego',
     'pit',
     'replaced',
     'recharging',
     'station',
     'might',
     'check',
     'prices',
     'crap',
     'samsung',
     'android',
     'bad',
     'shows',
     'late',
     'qs',
     'process',
     'ideas',
     'leaves',
     'early',
     'even',
     'creative',
     'busy',
     'trying',
     'balance',
     'power',
     'power',
     'needs',
     'vs',
     '3g',
     'sucks',
     'quick',
     'might',
     'go',
     'airplane',
     'mode.',
     'battery',
     'keep',
     'tweets',
     'thanks',
     '\x89ûï',
     'best',
     'thing',
     'heard',
     'weekend',
     'gave',
     '2',
     'money',
     'relief',
     'need',
     '2.',
     'vs',
     'bing',
     'bing',
     'shot',
     'success',
     'w/',
     'structured',
     'search',
     'potentially',
     'higher',
     'margin',
     'cpa',
     'model',
     'vs',
     '2',
     'coming',
     'guess',
     'pretty',
     'desperate',
     'give',
     'attention.',
     'shipments',
     'daily',
     'follow',
     '4',
     'updates',
     'store',
     'seems',
     'ipad2s',
     'dead',
     'find',
     'secret',
     'batphone',
     'barry',
     'diller',
     'thinks',
     'content',
     'nuts',
     'japan',
     'docomo',
     'introduced',
     'mobile',
     'apps',
     'six',
     'years',
     'came',
     'store',
     'jeez',
     'guys',
     'dunno',
     'gym',
     'u',
     'realize',
     'need',
     'find',
     'better',
     'stream',
     'follow',
     'inane',
     '2',
     'tweets',
     'surely',
     'southby',
     'getting',
     'full',
     'underway',
     'tell',
     'intermittent',
     'brick',
     'length',
     'penalty',
     'based',
     'severity',
     'breach',
     'webmaster',
     'guidelines',
     'i.e.white',
     'text',
     'white',
     'bgr',
     'might',
     'get',
     '30',
     'day',
     'pen',
     '\x89ûï',
     'launch',
     'major',
     'new',
     'social',
     'network',
     'called',
     'circles',
     'possibly',
     'today',
     '\x89û\x9d',
     'never',
     'beat',
     'myspace.',
     'hey',
     'bout',
     'donating',
     'spending',
     'new',
     'japan',
     'really',
     'need',
     'thing',
     '3g',
     '3',
     'hrs',
     'tweeting',
     'dead',
     'need',
     'upgrade',
     'plugin',
     'stations',
     'feeling',
     'worst',
     'place',
     'try',
     'amp',
     'get',
     '2',
     'everyone',
     'trying',
     'get',
     'one',
     'thought',
     'would',
     'use',
     'lot',
     'even',
     'touched',
     'hmmzies.',
     'interested',
     'location',
     'based',
     'tech',
     'indoor',
     'venues',
     'businesses',
     'convention',
     'centers',
     'etc',
     'tech',
     'needs',
     'improve',
     'first',
     'first',
     'even',
     'exist',
     'last',
     'year',
     'already',
     'feel',
     'like',
     'pulling',
     'antique',
     'everytime',
     'use',
     'shoot',
     'display',
     'search',
     'results',
     'go',
     'questions',
     'later',
     'mocking',
     'like',
     'money',
     'come',
     'on.',
     'one',
     'worst',
     'use',
     'long',
     'time.',
     'disliking',
     'twitter',
     'auto',
     'shortening',
     'links',
     'overheating',
     'many',
     'british',
     'sounding',
     'people',
     'texas',
     'wilting',
     'stress',
     'location',
     'pixieengine',
     'says',
     'future',
     'location',
     'location',
     'location',
     'know',
     'week',
     'tough',
     'battery',
     'help',
     'jeebus',
     'keep',
     'correcting',
     'curse',
     'words.',
     'launch',
     'major',
     'new',
     'social',
     'network',
     'called',
     'circles',
     'updated',
     '*not',
     'launched',
     'soon',
     'care',
     'launch',
     'product',
     'wait',
     'launch',
     'product',
     'exists',
     'wait',
     'product',
     'exist',
     'god',
     'like',
     'imac',
     'macbook',
     'blackberry',
     'staring',
     'enough',
     'time',
     'read',
     'book',
     'remember',
     'ha',
     'seems',
     'like',
     'gt',
     '\x89ûï',
     'news',
     'sxswi',
     'temporary',
     'store',
     '\x89û\x9d',
     'compiling',
     'list',
     'one',
     'doc',
     'taking',
     'lot',
     'longer',
     'thought',
     'many',
     'parties',
     'many',
     'good',
     'musicians.',
     'day',
     '1',
     'charger',
     'kicked',
     'bucket',
     'heck',
     'store',
     'within',
     'walking',
     'distance',
     'feel',
     'unequipped',
     'compared',
     'everyone',
     'else',
     'dang',
     'hm',
     'need',
     'another',
     '1',
     'launch',
     'major',
     'new',
     'social',
     'network',
     'called',
     'circles',
     'possibly',
     'today',
     'launching',
     'anything',
     'partying',
     'nothing',
     'teeming',
     'sea',
     'addicts',
     'busy',
     'twittering',
     'ever',
     'engage',
     'one',
     'anoth\x89û_',
     'cont',
     'fan',
     'new',
     'trend',
     'audience',
     'sharing',
     'opinions',
     'via',
     'holding',
     'listen',
     'tomlinson',
     'tx',
     'observer',
     'says',
     'subscription',
     'data',
     'holding',
     'biggest',
     'impediment',
     'success',
     'open',
     'shop',
     'concept',
     'spend',
     'waiting',
     'packing',
     'point',
     'showing',
     'fragmentation',
     'left',
     'pocket',
     'guide',
     'hotel',
     'know',
     'going',
     'cope',
     'say',
     'usability',
     'ipad/iphone',
     'problem',
     'living',
     'stories',
     'process',
     'creating',
     'content',
     'change',
     'interface',
     'brilliant',
     'read',
     'attention',
     'marketers',
     'media',
     'professionals',
     'save',
     'l.a.m.e',
     'law',
     'averages',
     'better',
     'buzz',
     'circles',
     '______',
     'new',
     'rule',
     'ooing',
     'ahing',
     'new',
     'get',
     'big',
     'deal',
     'everybody',
     'one',
     'says',
     'connect',
     'internet',
     'even',
     'though',
     'wifi',
     'works',
     'great',
     'computer',
     'suggestions',
     'q',
     'social',
     'sites',
     'like',
     'delicious',
     'often',
     'better',
     'results',
     'bing',
     'read',
     'selling',
     '2',
     'release',
     'day',
     'good',
     'lord',
     'vortex',
     'smugness',
     'day',
     'may',
     'unbearable.',
     '\x89ûï',
     '10',
     'dangerous',
     'apps',
     'went',
     'whole',
     'day',
     'w/out',
     'laptop',
     'amp',
     'used',
     '1',
     'things',
     'missed',
     'ftp',
     'cloudapp',
     'fast',
     'typing',
     'amp',
     'skype',
     'think',
     'would',
     'blackberry',
     'gave',
     'finger',
     'guess',
     'carry',
     'around',
     'lame',
     'people',
     'store',
     'smell',
     'great',
     'thank',
     'letting',
     'test',
     'drive',
     'car',
     'store',
     'lets',
     'hope',
     'fix',
     'phone',
     'get',
     'see',
     'fail',
     'social',
     'another',
     'day',
     'okay',
     'circles',
     'debuting',
     'today',
     'trajan',
     'destroyed',
     'lt',
     'title',
     'gt',
     'tag',
     'websites',
     'seo',
     'open',
     'graph',
     'protocol',
     'added',
     'clean',
     'title',
     'tag',
     'instead',
     'know',
     'u',
     'u',
     'selling',
     'weve',
     'never',
     'talked',
     'hate',
     'product',
     'check',
     'heyo',
     '4',
     'pop',
     'store',
     'line',
     'ridic',
     'trying',
     'update',
     'software',
     '4.0',
     'download',
     'far',
     'luck',
     'wonder',
     'phone',
     'mexico.',
     'def',
     'could',
     'use',
     'today',
     'tweeting',
     'via',
     'sorta',
     'pretty',
     'much',
     'sux.',
     'dense',
     'una',
     'vuelta',
     'por',
     'para',
     'ver',
     'la',
     'gran',
     'diferencia..rt',
     'revolution',
     'clumsily',
     'translated',
     'google.',
     '\x89÷¼',
     'better',
     '\x89÷_',
     '\x89ã_',
     'cocky',
     'perhaps',
     'separate',
     'ipad2',
     'store',
     'would',
     'good',
     'idea',
     'guess',
     'need',
     'extra',
     'marketing',
     'disgusted',
     'battery',
     'life',
     'already',
     '11',
     '3:30',
     'pm',
     'blackberry',
     'going',
     'strong',
     'thought',
     'social',
     'get',
     'overblown',
     'may',
     'announcing',
     'circles',
     'today',
     'classiest',
     'fascist',
     'company',
     'america',
     'really',
     'elegant',
     'lost',
     'corrupted',
     'crazy',
     'hotels',
     'good',
     'times',
     'aus',
     'midday',
     'outlet',
     'blocked',
     'immobile',
     'booth',
     'serves',
     'purpose',
     'taunt',
     '2011',
     'novelty',
     'news',
     'apps',
     'fades',
     'fast',
     'among',
     'digital',
     'delegates',
     'via',
     'xipad',
     'true',
     'lost',
     'way',
     'caring',
     'much',
     'business',
     'vs.',
     'shite',
     'new',
     'store',
     'includes',
     'uberguide',
     'sponso\x89û_',
     'cont',
     'know',
     'started',
     'lying',
     'signal',
     'strength.',
     'nexus',
     '10x',
     'useful',
     'iphone4',
     'amp',
     'many',
     'ppl',
     'get',
     'twitter',
     'searches',
     'update',
     'hootsuite',
     'tweetdeck',
     'paper',
     'phones',
     'means',
     'likely',
     'useless',
     'well',
     'worked',
     '11',
     'years',
     'seen',
     'lot',
     'evil.',
     'design',
     'headaches',
     'avoiding',
     'pitfalls',
     'new',
     'design',
     'challenges',
     'disaster',
     'died',
     'middle',
     'function',
     'heading',
     'store.',
     '1',
     'march',
     '11',
     '2011',
     '4:59pm',
     'pst',
     '2011',
     'novelty',
     'news',
     'apps',
     'fades',
     'fast',
     'among',
     'digital',
     'delegates',
     'via',
     '2011',
     'novelty',
     'news',
     'apps',
     'fades',
     'fast',
     'among',
     'digital',
     'delegates',
     '2011',
     'smackdown',
     'bloody',
     'banality',
     '2011',
     'smackdown',
     'bloody',
     'banality',
     'via',
     'even',
     '10am',
     'batt',
     '54',
     'shit',
     'selfish',
     'brand',
     'microsoft',
     'served',
     'brand',
     'well.',
     'talk',
     'mistakes',
     'building',
     'fucking',
     'mac',
     'users',
     '..',
     'cwebb',
     'grant',
     'hill',
     'design',
     'headaches',
     'take',
     'two',
     'tablet',
     'call',
     'morning',
     'brain',
     'apologies',
     'randy',
     'ads',
     'popping',
     'search',
     'results',
     'algorithm',
     'skewed',
     'weekend',
     'randy',
     'queries',
     'attendees.,2+',
     'hour',
     'wait',
     'makeshift',
     'store',
     '2.',
     'make',
     'sense',
     'limit',
     'content',
     'specific',
     'platform',
     'web',
     'interview',
     'w/',
     'guy',
     'kawasaki',
     'believe',
     'god',
     'explanation',
     'continuous',
     'survival',
     'years.',
     'lol',
     'groupme',
     'talks',
     'store',
     'approval',
     'woes',
     'would',
     'big',
     'group',
     'people',
     'would',
     'agree',
     'love',
     'heard',
     'quite',
     'bit',
     'grumbling',
     'holding',
     'back',
     'features',
     '1',
     'people',
     'would',
     'buy',
     'v2.',
     'meetups',
     'work',
     'well',
     'least',
     'ps',
     'meetup',
     'nfc',
     'bc',
     'standardization',
     ...]




```python
neg_freq = FreqDist(neg_tokenized)
neg_freq.most_common(10)
```




    [('2', 61),
     ('store', 44),
     ('new', 43),
     ('like', 39),
     ('design', 28),
     ('social', 28),
     ('people', 27),
     ('circles', 26),
     ('apps', 26),
     ('need', 25)]




```python
list(nltk.bigrams(neg_tokenized[:30]))
```




    [('3g', '3'),
     ('3', 'hrs'),
     ('hrs', 'tweeting'),
     ('tweeting', 'dead'),
     ('dead', 'need'),
     ('need', 'upgrade'),
     ('upgrade', 'plugin'),
     ('plugin', 'stations'),
     ('stations', 'hope'),
     ('hope', 'festival'),
     ('festival', 'crashy'),
     ('crashy', 'noticed'),
     ('noticed', 'dst'),
     ('dst', 'coming'),
     ('coming', 'weekend'),
     ('weekend', 'many'),
     ('many', 'users'),
     ('users', 'hour'),
     ('hour', 'late'),
     ('late', 'come'),
     ('come', 'sunday'),
     ('sunday', 'morning'),
     ('morning', 'false'),
     ('false', 'alarm'),
     ('alarm', 'circles'),
     ('circles', 'coming'),
     ('coming', 'now\x89ûòand'),
     ('now\x89ûòand', 'probably'),
     ('probably', 'ever')]




```python
from wordcloud import WordCloud
wordcloud = WordCloud(stopwords=stopwords_list,collocations=True,max_words=100)
wordcloud.generate(','.join(pos_corpus))
plt.figure(figsize = (10, 12), facecolor = 'green', edgecolor = 'green') 
plt.imshow(wordcloud) 
plt.title('Most Frequent Positive Tweet Words')
plt.axis('off');
```


![png](output_63_0.png)



```python
wordcloud = WordCloud(stopwords=stopwords_list,collocations=True, max_words =100)
wordcloud.generate(','.join(neg_corpus))
plt.figure(figsize = (10, 12), facecolor = 'red', edgecolor = 'red') 
plt.imshow(wordcloud) 
plt.title('Most Frequent Negative Tweet Words')
plt.axis('off');
```


![png](output_64_0.png)


A target column was created that indicated the tweet expressed a Positive emotion. Another column was created for tweets that had no emotion toward brand or product. Lastly, the negative column was created to count for the negative responses in the text.


```python
#add a target column where the emotion toward the product is positive. 
df['target'] = (df['emotion'] == 'Positive emotion').astype(int)
df['negative'] = (df['emotion'] == 'Negative emotion').astype(int)
df['neutral'] = (df['emotion'] == 'No emotion toward brand or product').astype(int)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweets</th>
      <th>product</th>
      <th>emotion</th>
      <th>target</th>
      <th>negative</th>
      <th>neutral</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>.@wesley83 I have a 3G iPhone. After 3 hrs twe...</td>
      <td>iPhone</td>
      <td>Negative emotion</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@jessedee Know about @fludapp ? Awesome iPad/i...</td>
      <td>iPad or iPhone App</td>
      <td>Positive emotion</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@swonderlin Can not wait for #iPad 2 also. The...</td>
      <td>iPad</td>
      <td>Positive emotion</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@sxsw I hope this year's festival isn't as cra...</td>
      <td>iPad or iPhone App</td>
      <td>Negative emotion</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>@sxtxstate great stuff on Fri #SXSW: Marissa M...</td>
      <td>Google</td>
      <td>Positive emotion</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Creating separate columns helps vectorize the text when pulling the text through the model created. 

# Model 1



```python
y = df['emotion'].copy()
X = df['tweets'].copy()
#make copies of the texts so the original file is not altered.
```


```python
X_train,X_test, y_train,y_test = train_test_split(X,y,stratify=y,random_state=45)
```

We can see that the data was evenly distributed between both the train and test sets. 


```python
y_train.value_counts(normalize=True)
#percentage of training data distribution
```




    No emotion toward brand or product    0.602954
    Positive emotion                      0.333184
    Negative emotion                      0.063862
    Name: emotion, dtype: float64




```python
y_test.value_counts(normalize=True)
#percentage of testing data distribution
```




    No emotion toward brand or product    0.603132
    Positive emotion                      0.333333
    Negative emotion                      0.063535
    Name: emotion, dtype: float64




```python
X_train.isna().sum()
#missing value
```




    1




```python
X_train.fillna('',inplace=True)
```


```python
X_train.isna().sum()
```




    0




```python
string = ','.join(str(v) for v in corpus)
patterns = [r"(http?://\w*\.\w*/+\w+)",
            r'\#\w*',
            r'RT [@]?\w*:',
            r'\@\w*',
            r"(?=\S*['-])([a-zA-Z'-]+)"]
            
clean_corpus = re.sub('|'.join(patterns), '', string)
```


```python
tokenizer = nltk.tokenize.word_tokenize(clean_corpus)
```


```python
tokenized = [word.lower() for word in tokenizer if word.lower() not in stopwords_list]
```


```python
vectorizer = TfidfVectorizer(tokenized,stop_words=stopwords_list)
vectorizer
```




    TfidfVectorizer(input=['3g', '3', 'hrs', 'tweeting', 'dead', 'need', 'upgrade',
                           'plugin', 'stations', 'know', 'awesome', 'ipad/iphone',
                           'likely', 'appreciate', 'design', 'giving', 'free', 'ts',
                           'wait', '2', 'sale', 'hope', 'festival', 'crashy',
                           'great', 'stuff', 'fri', 'mayer', 'tim', 'tech', ...],
                    stop_words=['i', 'me', 'my', 'myself', 'we', 'our', 'ours',
                                'ourselves', 'you', "you're", "you've", "you'll",
                                "you'd", 'your', 'yours', 'yourself', 'yourselves',
                                'he', 'him', 'his', 'himself', 'she', "she's",
                                'her', 'hers', 'herself', 'it', "it's", 'its',
                                'itself', ...])




```python
X_train_tfidf = vectorizer.fit_transform(X_train).astype(float)
X_test_tfidf = vectorizer.transform(X_test).astype(float)
```


```python
X_train_tfidf.shape
#(tweets, words)
```




    (6702, 8325)




```python
## Make, fit model
base_tree = RandomForestClassifier(class_weight='balanced')
base_tree.fit(X_train_tfidf,y_train)
```




    RandomForestClassifier(class_weight='balanced')




```python
## Get predictions
y_hat_train = base_tree.predict(X_train_tfidf)
y_hat_test =  base_tree.predict(X_test_tfidf)
```


```python
evaluate_model(y_test, y_hat_test, X_test_tfidf, base_tree) 
```

                                        precision    recall  f1-score   support
    
                      Negative emotion       0.66      0.22      0.33       142
    No emotion toward brand or product       0.72      0.87      0.79      1348
                      Positive emotion       0.67      0.49      0.57       745
    
                              accuracy                           0.70      2235
                             macro avg       0.68      0.53      0.56      2235
                          weighted avg       0.70      0.70      0.68      2235
    



![png](output_85_1.png)



![png](output_85_2.png)



```python
display(base_tree.score(X_train_tfidf, y_train))
display(base_tree.score(X_test_tfidf, y_test))
```


    0.957624589674724



    0.7038031319910515


The overall base model has an accuracy of 70.4%.

# Model 2


```python
y = df['emotion'].copy()
X = df['tweets'].copy()
```


```python
X_train,X_test, y_train,y_test = train_test_split(X,y,stratify=y,random_state=45)
```


```python
y_train.value_counts(normalize=True)
```




    No emotion toward brand or product    0.602954
    Positive emotion                      0.333184
    Negative emotion                      0.063862
    Name: emotion, dtype: float64




```python
y_test.value_counts(normalize=True)
```




    No emotion toward brand or product    0.603132
    Positive emotion                      0.333333
    Negative emotion                      0.063535
    Name: emotion, dtype: float64




```python
X_train.fillna('',inplace=True)
X_train.isna().sum()
```




    0




```python
tokenizer = nltk.tokenize.word_tokenize(clean_corpus)
```


```python
tokenized = [word.lower() for word in tokenizer if word.lower() not in stopwords_list]
```


```python
vectorizer = TfidfVectorizer(tokenized,stop_words=stopwords_list, binary=True)
vectorizer
```




    TfidfVectorizer(binary=True,
                    input=['3g', '3', 'hrs', 'tweeting', 'dead', 'need', 'upgrade',
                           'plugin', 'stations', 'know', 'awesome', 'ipad/iphone',
                           'likely', 'appreciate', 'design', 'giving', 'free', 'ts',
                           'wait', '2', 'sale', 'hope', 'festival', 'crashy',
                           'great', 'stuff', 'fri', 'mayer', 'tim', 'tech', ...],
                    stop_words=['i', 'me', 'my', 'myself', 'we', 'our', 'ours',
                                'ourselves', 'you', "you're", "you've", "you'll",
                                "you'd", 'your', 'yours', 'yourself', 'yourselves',
                                'he', 'him', 'his', 'himself', 'she', "she's",
                                'her', 'hers', 'herself', 'it', "it's", 'its',
                                'itself', ...])




```python
X_train_tfidf = vectorizer.fit_transform(X_train).astype(float)
X_test_tfidf = vectorizer.transform(X_test).astype(float)
```


```python
best_tree = RandomForestClassifier(class_weight='balanced')
best_tree.fit(X_train_tfidf,y_train)
```




    RandomForestClassifier(class_weight='balanced')




```python
rf_random = RandomForestClassifier()
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 300, num = 5)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'criterion': ['gini','entropy']}
```


```python
rf_tree = RandomizedSearchCV(estimator = rf_random, 
                             param_distributions = random_grid, 
                             n_iter = 100,cv = 3, verbose=2, 
                             random_state=45, n_jobs = -1)
```


```python
rf_tree.fit(X_train_tfidf, y_train)
```

    Fitting 3 folds for each of 100 candidates, totalling 300 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  33 tasks      | elapsed:  1.2min
    [Parallel(n_jobs=-1)]: Done 154 tasks      | elapsed:  5.7min
    [Parallel(n_jobs=-1)]: Done 300 out of 300 | elapsed: 12.8min finished





    RandomizedSearchCV(cv=3, estimator=RandomForestClassifier(), n_iter=100,
                       n_jobs=-1,
                       param_distributions={'bootstrap': [True, False],
                                            'criterion': ['gini', 'entropy'],
                                            'max_depth': [10, 20, 30, 40, 50, 60,
                                                          70, 80, 90, 100, 110,
                                                          None],
                                            'max_features': ['auto', 'sqrt'],
                                            'min_samples_leaf': [1, 2, 4],
                                            'min_samples_split': [2, 5, 10],
                                            'n_estimators': [50, 112, 175, 237,
                                                             300]},
                       random_state=45, verbose=2)




```python
display(rf_tree.best_params_)
```


    {'n_estimators': 112,
     'min_samples_split': 10,
     'min_samples_leaf': 2,
     'max_features': 'auto',
     'max_depth': None,
     'criterion': 'entropy',
     'bootstrap': False}



```python
tree_2 = rf_tree.best_estimator_
#tree with best params
y_hat_test = tree_2.predict(X_test_tfidf)
#predictions
```


```python
evaluate_model(y_test, y_hat_test, X_test_tfidf, tree_2)
```

                                        precision    recall  f1-score   support
    
                      Negative emotion       0.58      0.11      0.18       142
    No emotion toward brand or product       0.71      0.87      0.79      1348
                      Positive emotion       0.65      0.50      0.56       745
    
                              accuracy                           0.70      2235
                             macro avg       0.65      0.49      0.51      2235
                          weighted avg       0.69      0.70      0.67      2235
    



![png](output_104_1.png)



![png](output_104_2.png)



```python
display(tree_2.score(X_train_tfidf, y_train))
display(tree_2.score(X_test_tfidf, y_test))
```


    0.8613846612951358



    0.6975391498881431


The second model has an overall accuracy of 69.8%

# Results and Conclusion

Even though Model 2 had a one percent increase in the overall accuracy. The negative responses accuracy decreased in the second model. Which means Model 1 is the better model for predicting positive and negative responses in a given text.

Model 1 performance:
    - Positive 49% accuracy 
    - Negative 22% accuracy
    - Neutral 87% accuracy
    
    
    
 Model 2 performance:
     - Positive 50% accuracy
     - Negative 11% accuracy
     - Neutral 87% accuracy
     
     

Both models are overfit as there is not enough text data for positive and negative responses to train the model. If the end result was to analyze data that have neutral responses. Model 2 would be the best model to use as it is not as overfit as model 1.

# Insight and Recommendations

Due to the time frame at which the text data was extracted. It is recommended to keep in mind the context of the results as there are not many negative responses as positive, and positive as neutral. Also, positive responses uses a bigger variety of words where as negative responses use the same repeated words. 

The top words used in negative responses include:
 - ipad2, 61
 - store, 44
 - new, 43
 - like, 39
 - design, 28
 - social, 28
 - people, 27
 - circles, 26
 - apps, 26
 - need, 25

The top words used in positive responses include:
 - know, 1
 - awesome, 1
 - ipad/iphone, 1
 - likely, 1
 - appreciate, 1
 - design, 1
 - giving, 1
 - free, 1
 - ts, 1
 - wait, 1

### Things to look into:
1. There are a lot of negative responses correlated with the ipad2.


2. In both the negative and positive responses people mention the design. Knowing that a lot of people tweeting about apple and google products are interested in the design. 


3. There are positive responses towards the ipad/iphone.

