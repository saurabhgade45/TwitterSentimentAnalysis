from tkinter import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
from tweepy.streaming import StreamListener
from IPython.display import display
from PIL import ImageTk,Image
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans



from sklearn import datasets, linear_model

class TwitterClient(object):

    def __init__(self):

        consumer_key = 'QZqskJ9NgTQBkKDIFBdOYyLhm'
        consumer_secret = 'qHGke4c0IwnaX1Foqs6dcDCtvCTsn3SA79buxvP7vJRQQdKjM5'
        access_token = '1175279623862538241-BjK4Xw217UMJSSeKi7l0TZN0wLzp6f'
        access_token_secret = 'dQbQOwCMQUBxS8RLQGRkRfdUMqcSr7tHyR0oLEYl3UXLP'

        try:

            self.auth = OAuthHandler(consumer_key, consumer_secret)

            self.auth.set_access_token(access_token, access_token_secret)

            self.api = tweepy.API(self.auth)
        except:
            print("Error: Authentication Failed")

    def clean_tweet(self, tweet):

        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def get_tweet_sentiment(self, tweet):

        analysis = TextBlob(self.clean_tweet(tweet))

        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'

    def get_tweets(self, query, count=10):

        tweets = []

        try:

            fetched_tweets = self.api.search(q=query, count=count)

            for tweet in fetched_tweets:

                parsed_tweet = {}

                parsed_tweet['text'] = tweet.text

                parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text)

                if tweet.retweet_count > 0:

                    if parsed_tweet not in tweets:
                        tweets.append(parsed_tweet)
                else:
                    tweets.append(parsed_tweet)

            return tweets

        except tweepy.TweepError as e:

            print("Error : " + str(e))

def main0():
    # creating object of TwitterClient Class
        api = TwitterClient()



        tweets = api.get_tweets(query='virat kholi', count=100)




        ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
        print('tweets about virat kohli')

        print("Hate tweets percentage: {} %".format(100 * len(ptweets) / len(tweets)))

        ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']

        print("Love tweets percentage: {} %".format(100 * len(ntweets) / len(tweets)))

        print("Neutral tweets percentage: {} %".format(100 * (len(tweets) - len(ntweets) - len(ptweets)) / len(tweets)))

def main2():
    api = TwitterClient()
    tweets = api.get_tweets(query='virat kholi', count=100)
    ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']



    ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
    positive = format(100 * len(ptweets) / len(tweets))
    negative = format(100 * len(ntweets) / len(tweets))
    neutral = format(100 * (len(tweets) - len(ntweets) - len(ptweets)) / len(tweets))
    labels = ['positive [' + str(positive) + '%]', 'neutral[' + str(neutral) + '%]',
              'negative[' + str(negative) + '%]']
    sizes = [positive, neutral, negative]
    colors = ['yellowgreen', 'gold', 'red']
    patches, texts = plt.pie(sizes, colors=colors, startangle=90)
    plt.legend(patches, labels, loc="best")
    plt.title('How people React on Tweets.')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
def main3():
    print("\n\nPositive tweets:")
    api = TwitterClient()
    tweets = api.get_tweets(query='virat kholi', count=100)
    ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
    for tweet in ptweets[:10]:
        print(tweet['text'])
def main4():
    print("\n\nNegative tweets:")
    api = TwitterClient()
    tweets = api.get_tweets(query='virat kholi', count=100)
    ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
    for tweet in ntweets[:10]:
        print(tweet['text'])
def main5():
    api = TwitterClient()
    tweets = api.get_tweets(query='virat kholi', count=100)
    ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']

    ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
    positive = format(100 * len(ptweets) / len(tweets))
    negative = format(100 * len(ntweets) / len(tweets))
    neutral = format(100 * (len(tweets) - len(ntweets) - len(ptweets)) / len(tweets))
    left = [1, 2, 3]

    # heights of bars
    height = [positive, negative, neutral]

    # labels for bars
    tick_label = ['positive response', 'negative response', 'neutral']

    # plotting a bar chart
    plt.bar(left, height, tick_label=tick_label, width=0.8, color=['red', 'green'])

    # naming the x-axis
    plt.xlabel('x - axis')
    # naming the y-axis
    plt.ylabel('y - axis')
    # plot title
    plt.title('analysis')

    # function to show the plot
    plt.show()
def main6():
    api = TwitterClient()
    tweets = api.get_tweets(query='virat kholi', count=100)
    ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']

    ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
    positive = format(100 * len(ptweets) / len(tweets))
    negative = format(100 * len(ntweets) / len(tweets))
    neutral = format(100 * (len(tweets) - len(ntweets) - len(ptweets)) / len(tweets))
    x = int(float(positive))
    y = int(float(negative))
    z = int(float(neutral))

    x = np.linspace(x, y, z)

    # Plot the data
    plt.plot(x, x, label='linear line according to response')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()

consumer_key = 'QZqskJ9NgTQBkKDIFBdOYyLhm'
consumer_secret = 'qHGke4c0IwnaX1Foqs6dcDCtvCTsn3SA79buxvP7vJRQQdKjM5'
access_token = '1175279623862538241-BjK4Xw217UMJSSeKi7l0TZN0wLzp6f'
access_token_secret = 'dQbQOwCMQUBxS8RLQGRkRfdUMqcSr7tHyR0oLEYl3UXLP'



def twitter_setup():

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    # Return API with authentication:
    api = tweepy.API(auth)
    return api


extractor = twitter_setup()

# We create a tweet list as follows:

tweets1 = extractor.user_timeline(screen_name="realDonaldTrump", count=200)
#print("Number of tweets extracted: {}.\n".format(len(tweets1)))
data = pd.DataFrame(data=[tweet.text for tweet in tweets1], columns=['Tweets'])
data['len'] = np.array([len(tweet.text) for tweet in tweets1])
data['ID'] = np.array([tweet.id for tweet in tweets1])
data['Date'] = np.array([tweet.created_at for tweet in tweets1])
data['Source'] = np.array([tweet.source for tweet in tweets1])
data['Likes'] = np.array([tweet.favorite_count for tweet in tweets1])
data['RTs'] = np.array([tweet.retweet_count for tweet in tweets1])

def main7():
    print("5 recent tweets:\n")
    for tweet in tweets1[:5]:
        print(tweet.text)
        print()
def main8():
    print("dataframe of tweets")

    display(data.head(10))
def main9():
    fav_max = np.max(data['Likes'])
    fav = data[data.Likes == fav_max].index[0]
    print("The tweet with more likes is: \n{}".format(data['Tweets'][fav]))
    print("Number of likes: {}".format(fav_max))
    print("{} characters.\n".format(data['len'][fav]))
def main10():
    rt_max = np.max(data['RTs'])
    rt = data[data.RTs == rt_max].index[0]
    print("The tweet with more retweets is: \n{}".format(data['Tweets'][rt]))
    print("Number of retweets: {}".format(rt_max))
    print("{} characters.\n".format(data['len'][rt]))
def main11():
    tlen = pd.Series(data=data['len'].values, index=data['Date'])

    tlen.plot(figsize=(16, 4), color='r')
    plt.show()
def main12():
    tfav = pd.Series(data=data['Likes'].values, index=data['Date'])
    tret = pd.Series(data=data['RTs'].values, index=data['Date'])
    tfav.plot(figsize=(16, 4), label="Likes", legend=True)
    tret.plot(figsize=(16, 4), label="Retweets", legend=True);
    plt.show()
my_dict = { 'name' : ["Remesh pawar", "Rajan singh", "monohar desai", "josn datt", "saurabh gaikwad","monohar patil", "Harish Jadhav","Rakesh sharama","Kunal Monohar","R.V.Kesav"],
                   'age' : [20,27, 35, 55, 18, 21, 35,22,45,34],
                'date':["21/8/2000","22/9/2001","31/3/2004","2/6/1999","22/9/1998","31/4/2004","5/7/1997","8/8/1998","3/4/2002","2/4/2004"],

                 'com_name': ["Indian Oil Corporation Ltd", "Reliance Industries Ltd.",
                         "Oil And Natural Gas Corporation Ltd.",
                         "State Bank of India", "Tata Motors Ltd.", "Bharat Petroleum Corporation Ltd.",
                         "Hindustan Petroleum Corporation Ltd.", "Rajesh Exports Ltd.", "Tata Steel Ltd.",
                         "Coal India Ltd."],
                 'com_no_networks':["21000","22000","32000","45000","94000","33000","84000","75000","87000","91000"],
                'follwers': [420, 527, 1335, 255, 818, 421, 735, 922, 245, 334]
                   }
df = pd.DataFrame(my_dict)

df.com_no_networks=pd.to_numeric(df.com_no_networks)

def disp():
    print(df)
def disp1():
    m = np.max(df['com_no_networks'])
    fav = df[df.com_no_networks == m].index[0]
    print("maximum number of networks of company: \n{}".format(df['com_name'][fav]))
def disp2():
    n = np.max(df['follwers'])
    print("maximum number of follwers")
    print(n)
def disp3():
    x = np.mean(df['age'])
    print("avg age of people")
    print(x)
def disp4():
    tlen = pd.Series(data=df['follwers'].values, index=df['date'])
    tfav = pd.Series(data=df['com_no_networks'].values, index=df['date'])
    tlen.plot(figsize=(16, 4), color='r')
    plt.show()
def disp5():
    tlen = pd.Series(data=df['follwers'].values, index=df['date'])
    tfav = pd.Series(data=df['com_no_networks'].values, index=df['date'])
    tfav.plot(figsize=(16, 4), label="follwers", legend=True)
    tlen.plot(figsize=(16, 4), label="com_no_networks", legend=True);
    plt.show()

my_dict1 = { 'user_id' : ["1192601", "2083884", "1203168	", "1733186	", "1524765","1136133", "1680361","1365174","1712567","1612453"],
                   'age' : [20,27, 35, 55, 18, 21, 35,22,45,34],
                'date':["21/8/2000","22/9/2001","31/3/2004","2/6/1999","22/9/1998","31/4/2004","5/7/1997","8/8/1998","3/4/2002","2/4/2004"],

                 'tenure': [212,34,22,67,38,415,33,789,1001,203],
                 'friends_count':["210","2200","320","450","940","330","8400","7500","8700","9100"],
                'likes': [420, 527, 1335, 255, 818, 421, 735, 922, 245, 334],
            'comments':[220,234,546,110,45,67,89,22,9,36]
                   }
df1 = pd.DataFrame(my_dict1)
def disp6():
    print(df1)
def disp40():
    m = np.max(df1['friends_count'])
    fav = df1[df1.friends_count == m].index[0]
    print("maximum number of friend user id: \n{}".format(df1['user_id'][fav]))

def disp9():
    n = np.max(df1['likes'])
    print("maximum number of likes")
    print(n)
def disp10():
    x = np.mean(df1['age'])
    print("avg age of people")
    print(x)
def disp11():
    tlen = pd.Series(data=df1['likes'].values, index=df1['friends_count'])


    tlen.plot(figsize=(16, 4), color='r')
    plt.show()


def disp12():
    tlen = pd.Series(data=df1['likes'].values, index=df1['friends_count'])
    tfav = pd.Series(data=df1['comments'].values, index=df1['friends_count'])

    tfav.plot(figsize=(16, 4), label="comments", legend=True)
    tlen.plot(figsize=(16, 4), label="likes", legend=True);
    plt.show()






def open_window3():
    top=Toplevel()
    top.title("analysis Using Lexicon")
    mylabel = Label(top, text="Lexicon Analysis", fg="red")
    mylabel.pack(fill=X, pady=20)
    top.geometry("500x500+120+120")
    button1=Button(top,text="Give percentage",command=main0,fg="blue")
    button1.pack(fill=X, pady=20)
    button2 = Button(top, text="Display some tweets", command=open_window5,fg="black")
    button2.pack(fill=X, pady=20)
    button3 = Button(top, text="Pie chart", command=main2,fg="green")
    button3.pack(fill=X, pady=20)
    button4 = Button(top, text="Bar Chart", command=main5,fg="red")
    button4.pack(fill=X, pady=20)
    button5 = Button(top, text="Line chart", command=main6,fg="blue")
    button5.pack(fill=X, pady=20)
    button6 = Button(top, text="Back", command=top.destroy,fg="black")
    button6.pack(fill=X)
def open_window5():
    top=Toplevel()
    top.title("Display tweets")
    mylabel = Label(top, text="Display Tweets", fg="red")
    mylabel.pack(fill=X, pady=20)
    top.geometry("500x500+120+120")
    button1=Button(top,text="positive",command=main3,fg="blue")
    button1.pack(fill=X, pady=20)
    button2 = Button(top, text="negative", command=main4,fg="green")
    button2.pack(fill=X, pady=20)
    button2 = Button(top, text="back", command=top.destroy,fg="black")
    button2.pack(fill=X)
def open_window4():
    top=Toplevel()
    top.title("Analysis Using Dataframe")
    mylabel = Label(top, text="Use of DataFrame", fg="red")
    mylabel.pack(fill=X, pady=15)
    top.geometry("500x500+120+120")
    button1=Button(top,text="display some tweets",command=main7,fg="black")
    button1.pack(fill=X, pady=15)
    button2 = Button(top, text="Show in dataframe", command=main8,fg="blue")
    button2.pack(fill=X, pady=15)
    button3 = Button(top, text="most likes", command=main9,fg="red")
    button3.pack(fill=X, pady=15)
    button4 = Button(top, text="most retweets", command=main10,fg="green")
    button4.pack(fill=X, pady=15)
    button5 = Button(top, text="analysis like per date", command=main11,fg="blue")
    button5.pack(fill=X, pady=15)
    button6 = Button(top, text="likes Vs retwwets", command=main12,fg="black")
    button6.pack(fill=X, pady=15)
    button7 = Button(top, text="Back", command=top.destroy,fg="black")
    button7.pack(fill=X)
def open_window1():
    top=Toplevel()
    top.title("Fackbook")
    mylabel = Label(top, text="Fackbook Analysis", fg="red")
    mylabel.pack(fill=X, pady=15)
    top.geometry("500x500+120+120")
    button1=Button(top,text="Display In dataframe",command=disp6,fg="black")
    button1.pack(fill=X,pady=15)
    button2 = Button(top, text="maximum friend count", command=disp40,fg="blue")
    button2.pack(fill=X,pady=15)
    button3 = Button(top, text="most likes", command=disp9,fg="red")
    button3.pack(fill=X,pady=15)
    button4 = Button(top, text="avg age", command=disp10,fg="blue")
    button4.pack(fill=X,pady=15)
    button5 = Button(top, text="likes per friend_count", command=disp11,fg="green")
    button5.pack(fill=X,pady=15)
    button5 = Button(top, text="likes vs comments", command=disp12,fg="blue")
    button5.pack(fill=X,pady=15)
    button6 = Button(top, text="Back", command=top.destroy,fg="black")
    button6.pack(fill=X)

def open_window2():
    top=Toplevel()
    top.title("LikedIN")
    mylabel = Label(top, text="Linked In Analysis", fg="pink")
    mylabel.pack(fill=X, pady=20)
    top.geometry("500x500+120+120")
    button1=Button(top,text="show In dataframe",command=disp,fg="black")
    button1.pack(fill=X,pady=20)
    button2 = Button(top, text="maximum network to company", command=disp1,fg="blue")
    button2.pack(fill=X,pady=20)
    button3 = Button(top, text="most follwers", command=disp2,fg="red")
    button3.pack(fill=X,pady=20)
    button4 = Button(top, text="Avg age", command=disp3,fg="blue")
    button4.pack(fill=X,pady=20)
    button5 = Button(top, text="follwers per date", command=disp4,fg="green")
    button5.pack(fill=X,pady=20)
    button6 = Button(top, text="networks per date", command=disp5,fg="blue")
    button6.pack(fill=X,pady=20)
    button7 = Button(top, text="back", command=top.destroy,fg="black")
    button7.pack(fill=X)
def kmean3():
    Data = {
        'twitter': [25, 34, 22, 27, 33, 33, 31, 22, 35, 34, 67, 54, 57, 43, 50, 57, 59, 52, 65, 47, 49, 48, 35, 33, 44, 45,
              38, 43, 51, 46],
        'fackbook': [79, 51, 53, 78, 59, 74, 73, 57, 69, 75, 51, 32, 40, 47, 53, 36, 35, 58, 59, 50, 25, 20, 14, 12, 20, 5, 29,
              27, 8, 7]
        }

    df = pd.DataFrame(Data, columns=['twitter', 'fackbook'])
    n=int(input("how many clusters u want to form"))
    kmeans = KMeans(n_clusters=n).fit(df)
    centroids = kmeans.cluster_centers_
    print(centroids)

    plt.scatter(df['twitter'], df['fackbook'], c=kmeans.labels_.astype(float), s=50, alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
    plt.show()
def disp7():
    my_dict6 = {'year': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2018, 2020],
                'users': [70, 75, 85, 98, 110, 115, 135, 165, 248, 313, 346]}
    df3 = pd.DataFrame(my_dict6)

    X = df3.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
    Y = df3.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions
    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')
    plt.show()
def tw1():
    my_dict6 = {'year': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2018, 2020],
                'users': [30, 40, 50, 54, 101, 117, 138, 151, 167, 185, 200]}
    df3 = pd.DataFrame(my_dict6)

    X = df3.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
    Y = df3.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions
    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')
    plt.show()
def tt():
    my_dict7 = {'year': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2018, 2020],
                'users': [27, 30, 35, 38, 40, 43, 46, 43,45, 48, 48]}
    df4 = pd.DataFrame(my_dict7)

    X = df4.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
    Y = df4.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions
    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')
    plt.show()

def open_window20():
    top = Toplevel()
    top.title("Linear Regression")
    mylabel = Label(top, text="Regrassion model", fg="red")
    mylabel.pack(fill=X, pady=20)
    top.geometry("500x500+120+120")
    button1 = Button(top, text="twitter", command=tw1, fg="black")
    button1.pack(fill=X, pady=20)
    button2 = Button(top, text="Fackbook", command=disp7, fg="blue")
    button2.pack(fill=X, pady=20)
    button3 = Button(top, text="Linked In", command=tt, fg="red")
    button3.pack(fill=X, pady=20)
    button7 = Button(top, text="back", command=top.destroy, fg="black")
    button7.pack(fill=X)

def open_window30():
    top = Toplevel()
    top.title("Models")
    mylabel = Label(top, text="Models In python", fg="red")
    mylabel.pack(fill=X, pady=20)
    top.geometry("500x500+120+120")
    button1 = Button(top, text="Linear regression", command=open_window20, fg="black")
    button1.pack(fill=X, pady=20)
    button2 = Button(top, text="Kmeans", command=kmean3, fg="blue")
    button2.pack(fill=X, pady=20)
    button3 = Button(top, text="Lexicon", command=open_window3, fg="red")
    button3.pack(fill=X, pady=20)
    button7 = Button(top, text="back", command=top.destroy, fg="black")
    button7.pack(fill=X)




root=Tk()
root.title("Medias")
root.iconbitmap('C:\\Users\\chand\\icon.ico')

myimg = ImageTk.PhotoImage(Image.open('C:\\Users\\chand\\saurabh.jpg'))
mylabel1 = Label(image=myimg)

mylabel1.pack()

mylabel=Label(root,text="Social Media Analysis",fg="red")
mylabel.pack(fill=X,pady=20)
button1=Button(root,text="Twitter analysis",command=open_window4,fg="black")
button1.pack(fill=X,pady=20)
button2=Button(root,text="Facebook",command=open_window1,fg="blue")
button2.pack(fill=X,pady=20)
button3=Button(root,text="LinkedIN",command=open_window2,fg="red")
button3.pack(fill=X,pady=20)
button8=Button(root,text="models",command=open_window30,fg="black")
button8.pack(fill=X,pady=20)
button4=Button(root,text="Back",command=root.destroy,fg="green")
button4.pack(fill=X,pady=20)

root.geometry("500x600+120+120")
root.mainloop()