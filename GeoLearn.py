#-*- coding: utf-8 -*-
#==============================================================================
#       AUTHOR: Rex Liu
# ORGANIZATION: University of Ottawa

#  DESCRIPTION: A module containing classes that faciliate learning users' locs
#==============================================================================


class Tweet:
    def __init__(self, id_tweet, id_user, time, profile_loc, geotag,
                 text, topic, rev):
        self.id_tweet = id_tweet
        self.id_user = id_user
        self.time = time
        self.profile_loc = profile_loc
        self.geotag = geotag
        self.text = text
        self.topic = topic
        self.rev = rev


class User:
    def __init__(self, id_user, geotag, rev, text):
        self.id_user = id_user
        self.geotag = geotag
        self.rev = rev
        self.text = text

    def addTweets(self, new_text):
        self.text += new_text


class TestUser:
    def __init__(self, id_user):
        self.id_user = id_user
        self.tweets = []

    def addTweets(self, new_tweet):
        self.tweets.append(new_tweet)
