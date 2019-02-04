import twitter
import nltk
import string

Twitter = twitter.Api(consumer_key='INSERT HERE',
		  consumer_secret='INSERT HERE',
		  access_token_key='INSERT HERE',
		  access_token_secret='INSERT HERE')
tweet_tokenizer = nltk.TweetTokenizer()
punctuation = string.punctuation.replace('#', '').replace('@', '') #Want to keep these in the tweet

target = input("Target twitter handle: ")
f = open('{}.txt'.format(target), 'w+')


def sanitize_Tweet(tweet_text):
    sep_Tweet = tweet_tokenizer.tokenize(tweet_text)
    for i in range(len(sep_Tweet)):
        if sep_Tweet[i][0] == '@':
            try:
                user = Twitter.GetUser(screen_name = sep_Tweet[i][1:]).name
                if i < (len(sep_Tweet) - 1):
                    if sep_Tweet[i+1][0] == '@':
                        user = Twitter.GetUser(screen_name = sep_Tweet[i][1:]).name + ','
                tweet_text = tweet_text.replace(sep_Tweet[i], user) #Replace twitter handle with username. @iWillSmith -> Will Smith
            except:
                return #If there's an error just return and go to the next. No need to dilly dally when we're just trying to get as many as we can.
        elif sep_Tweet[i] == '&amp;':
            tweet_text = tweet_text.replace('&amp;', '&') #Small quirk in the API. Just sanitize it
        elif 'https' in sep_Tweet[i]:
            tweet_text = tweet_text.replace(sep_Tweet[i], '') #Get rid of any links, they're no good for chat bots.

    tweet_text = tweet_text.replace('\n', ' ') #Replace any line breaks with a space. Trusting users to have periods before line breaks
    return tweet_text.strip().lower()


lis = [Twitter.GetUserTimeline(screen_name='{}'.format(target), include_rts = False, count = 1)[0].id]
for i in range(0, 20):
    
    statuses = Twitter.GetUserTimeline(screen_name='{}'.format(target), include_rts = False, count = 200, max_id = lis[-1])
    for s in statuses:
        #Check if we've seen this tweet before. len(lis)//2: tells the program to look starting from the middle, saves some computation time.
        if s.id not in lis[len(lis)//2:]:
            san_Tweet = sanitize_Tweet(s.text)
            try:
                if len(san_Tweet) > 0:
                    f.write('{}\n'.format(san_Tweet))
            except:
                pass
        lis.append(s.id)
        if len(lis) % 20 == 0:
            print(len(lis)) #Update on how many tweets have been gathered.
f.close()
