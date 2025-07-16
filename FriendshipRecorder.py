import tweepy
import json
import time
from multiprocessing import Queue, Process
import pyhdb
import datetime

""" Streaming Component """
def monitor_sample(q):
    while True:
        try:
            tc = TwitterStreamClient(q)
            tc.stream.sample()
            tc.stream.disconnect()
        except:
            pass

class TwitterStreamClient:
    def __init__(self, q):
        self.queue = q
        self.auth = tweepy.OAuthHandler(None,None)
        self.auth.set_access_token(None,None)
        self.api = tweepy.API(self.auth)
        self.stream_listener = StreamListener(self.queue)
        self.stream = tweepy.Stream(auth=self.api.auth, listener=self.stream_listener)

class StreamListener(tweepy.StreamListener):
    def __init__(self, q):
        self.q = q
        self.tp60 = 0  # throughput per 60, to monitor the current tweet per minute rate on twitters 1%
        self.pp60 = 0  # processed per 60, the number of elements inserted into the queue
        self.check60 = time.time()  # last checked timestamp for the tp60 count
        self.dist = [0, 0, 0]

    def on_data(self, data):
        try:
            data = json.loads(data)

            if not 'delete' in data:
                """ monitor current stream throughput """
                ts = time.time()

                if ts - self.check60 > 60:
                    print(str(time.strftime("%Y-%m-%d %H:%M")) + " Tp60: " + str(self.tp60))
                    print(self.dist, self.pp60)
                    self.tp60 = 1
                    self.pp60 = 0
                    self.check60 = ts
                else:
                    self.tp60 += 1

                if self.pp60 > 100:  # this is the request limit and corresponds to the number of credentials
                    return

                """ extract data """
                uid = data.get('user', {}).get('id', None)  # (following) user id

                if uid is None:
                    return

                rtid = data.get('retweeted_status', {}).get('user', {}).get('id', None)
                rid = data.get('in_reply_to_user_id', None)
                qid = data.get('quoted_status', {}).get('user', {}).get('id', None)
                ufriends = data.get('user', {}).get('friends_count', -1)
                ufollowers = data.get('user', {}).get('followers_count', -1)
                ulisted = data.get('user', {}).get('listed_count', -1)

                if any((rtid, rid, qid)) and not (rtid and qid):
                    type = "rt" if rtid else "r" if rid else "q"
                    oid = rtid if rtid else rid if rid else qid  # original user id
                else:
                    return

                """ ensure balanced data sampling """
                if rtid and min(self.dist) + 10  < self.dist[0]:
                    return
                if rid and min(self.dist) + 10  < self.dist[1]:
                    return
                if qid and min(self.dist) + 10  < self.dist[2]:
                    return

                if self.q.qsize() > 200:  # queue is nearly full
                    return

                if rtid:
                    self.dist[0] += 1
                    ofriends = data.get('retweeted_status', {}).get('user', {}).get('friends_count', -1)
                    ofollowers = data.get('retweeted_status', {}).get('user', {}).get('followers_count', -1)
                    olisted = data.get('retweeted_status', {}).get('user', {}).get('listed_count', -1)
                elif rid:
                    self.dist[1] += 1
                    ofriends = None
                    ofollowers = None
                    olisted = None
                else:
                    self.dist[2] += 1
                    ofriends = data.get('quoted_status', {}).get('user', {}).get('friends_count', -1)
                    ofollowers = data.get('quoted_status', {}).get('user', {}).get('followers_count', -1)
                    olisted = data.get('quoted_status', {}).get('user', {}).get('listed_count', -1)

                if ufriends == -1 or ofriends == -1:  # protected user
                    return

                data_dict = {
                    "uid": uid, "oid": oid,
                    "ufriends": ufriends, "ofriends": ofriends,
                    "ufollowers": ufollowers, "ofollowers": ofollowers,
                    "ulisted": ulisted, "olisted": olisted,
                    "uactivity": None, "oactivity": None,
                    "following": None, "followed": None,
                    "type": "{0}".format(type), "delay": 0
                }

                """ push monitored elements into queue """
                self.q.put(data_dict)
                self.pp60 += 1
        except:
            return

""" Crawling Component """
def run(c, q, insert_q):
    tc = TwitterClient(c, q, insert_q)
    tc.execute()

class TwitterClient:
    def __init__(self, c, q, insert_q):
        self.queue = q
        self.auth = tweepy.OAuthHandler(c['consumer_key'], c['consumer_secret'])
        self.auth.set_access_token(c['access_key'], c['access_secret'])
        self.api = tweepy.API(self.auth, wait_on_rate_limit=True, parser=tweepy.parsers.JSONParser())
        self.TSTAMP_OFFSET = 1288834974657  # snowflake
        self.insert_q = insert_q

    def execute(self):
        while True:
            next = q.get()
            try:
                # new entries need annotations
                if next['delay'] == 0:
                    if next['type'] == 'r':  # origin popularity is missing
                        uactivity, oactivity, ofriends, ofollowers, olisted = self.request_activity(next['uid'],
                                                                                        next['oid'], extract_extended_origin=True)
                        if uactivity == -1:  # error handling (=skipping) for protected user profiles
                            continue

                        next['ofriends'] = ofriends
                        next['ofollowers'] = ofollowers
                        next['olisted'] = olisted
                    else:
                        uactivity, oactivity = self.request_activity(next['uid'], next['oid'])

                        if uactivity == -1:  # error handling (=skipping) for protected user profiles
                            continue

                    next['uactivity'] = uactivity
                    next['oactivity'] = oactivity

                # extract friendship status regardless of delay
                next['following'], next['followed'] = self.request_friendship_status(next['uid'], next['oid'])

                # determine time when entry has to be recrawled
                if next['following']:
                    next_ts = None
                else:
                    if next['delay'] == 0:
                        next_ts = datetime.datetime.now() + datetime.timedelta(hours=1)
                    elif next['delay'] == 1:
                        next_ts = datetime.datetime.now() + datetime.timedelta(hours=23)
                    elif next['delay'] == 2:
                        next_ts = datetime.datetime.now() + datetime.timedelta(hours=24)
                    else:  # delay == 3
                        next_ts = None

                # write down results
                self.insert_q.put((next['uid'], next['oid'], next['ufriends'], next['ofriends'],
                                            next['ufollowers'], next['ofollowers'], next['ulisted'], next['olisted'],
                                            next['uactivity'], next['oactivity'], next['following'], next['followed'],
                                            next['type'], next['delay'], next_ts))
            except:
                continue

    def request_activity(self, uid, oid, extract_extended_origin=False):
        # timestamp from 2 weeks ago
        since = datetime.datetime.timestamp(datetime.datetime.now() - datetime.timedelta(weeks=2))
        since = int(since * 1000 - self.TSTAMP_OFFSET)  # timestamp conversion to Twitter's Snowflake scheme
        since = bin(since) + "0000000000000000000000"  # conversion to binary tweet id with 0 valued dc, wt, sn
        since = int(since[2:], base=2)  # conversion to decimal representation of tweet id from two weeks ago

        # extract user activity in the last two weeks
        try:
            tweet_timeline = self.api.user_timeline(uid, count=200, since_id=since)
        # for those retarded users that protect their tweets making tweepy throw an not authorized error
        except tweepy.TweepError:
            if extract_extended_origin:
                return -1, -1, -1, -1, -1
            else:
                return -1, -1


        if len(tweet_timeline) < 200:
            uactivity = len(tweet_timeline) / 14.0  # tweets per day in the last two weeks
        else:
            ts = tweet_timeline[-1].get('created_at')  # marks the time from now where 200 tweets have been posted
            ts = datetime.datetime.strptime(ts[4:-10] + ts[-4:], "%b %d %H:%M:%S %Y")  # convert to datetime object
            tdiff = datetime.datetime.now() - ts  # timedelta, but only stores days, seconds and microseconds
            days = tdiff.days + tdiff.seconds / 3600.0  # obtain a float marking the time difference in days
            uactivity = 200 / days

        # extract origin activity in the last two weeks (plus popularity counts if param is set)
        try:
            tweet_timeline = self.api.user_timeline(oid, count=200, since_id=since)
        # for those retarded users that protect their tweets making tweepy throw an not authorized error
        except tweepy.TweepError:
            if extract_extended_origin:
                return -1, -1, -1, -1, -1
            else:
                return -1, -1

        if len(tweet_timeline) < 200:
            oactivity = len(tweet_timeline) / 14.0  # tweets per day in the last two weeks
        else:
            ts = tweet_timeline[-1].get('created_at')  # marks the time from now where 200 tweets have been posted
            ts = datetime.datetime.strptime(ts[4:-10] + ts[-4:], "%b %d %H:%M:%S %Y")  # convert to datetime object
            tdiff = datetime.datetime.now() - ts  # timedelta, but only stores days, seconds and microseconds
            days = tdiff.days + tdiff.seconds / 3600.0  # obtain a float marking the time difference in days
            oactivity = 200 / days

        if extract_extended_origin:
            if len(tweet_timeline) != 0:  # extract popularity values from timeline
                ofriends = tweet_timeline[0].get('user', {}).get('friends_count', 0)
                ofollowers = tweet_timeline[0].get('user', {}).get('followers_count', 0)
                olisted = tweet_timeline[0].get('user', {}).get('listed_count', 0)
            else:  # there are no tweets to extract popularity values from
                user_obj = self.api.get_user(user_id=oid)
                ofriends = user_obj.get('friends_count', 0)
                ofollowers = user_obj.get('followers_count', 0)
                olisted = user_obj.get('listed_count', 0)

            return uactivity, oactivity, ofriends, ofollowers, olisted

        return uactivity, oactivity

    def request_friendship_status(self, uid, oid):
        status = self.api.show_friendship(source_id=uid, target_id=oid)

        is_following = status.get('relationship', {}).get('source', {}).get('following', None)
        is_being_followed = status.get('relationship', {}).get('source', {}).get('followed_by', None)

        return is_following, is_being_followed

""" Database Connection """
class DatabaseConnector:
    def __init__(self, q, insert_q):
        self.conn = pyhdb.connect(None)
        self.conn.autocommit = True
        self.cursor = self.conn.cursor()
        self.q = q
        self.insert_q = insert_q

    def write_data(self):
        stmt = "INSERT INTO follow_behavior VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"

        while True:
            time.sleep(0.01)
            data = insert_q.get()
            self.cursor.execute(stmt, data)

    def check_recrawl_db(self):
        time_low = datetime.datetime.now() - datetime.timedelta(minutes=5)
        time_high = datetime.datetime.now() + datetime.timedelta(minutes=5)

        stmt = "SELECT * FROM follow_behavior WHERE nextts >= '" + time_low.strftime('%Y-%m-%d %H:%M:%S') + \
               "' AND nextts < '" + time_high.strftime('%Y-%m-%d %H:%M:%S') + "'"
        self.cursor.execute(stmt)
        res = self.cursor.fetchall()

        for r in res:
            data_dict = {
                "uid": r[0], "oid": r[1],
                "ufriends": r[2], "ofriends": r[3],
                "ufollowers": r[4], "ofollowers": r[5],
                "ulisted": r[6], "olisted": r[7],
                "uactivity": r[8], "oactivity": r[9],
                "following": r[10], "followed": r[11],
                "type": "{0}".format(r[12]), "delay": r[13] + 1
            }

            self.q.put(data_dict)

if __name__ == '__main__':

    q = Queue(500)
    insert_q = Queue(2000)

    dbc = DatabaseConnector(q, insert_q)

    dbw = Process(target=dbc.write_data, args=())  # worker for inserting datasets into db
    dbw.start()

    stream_process = Process(target=monitor_sample, args=(q,))  # worker for adding fresh tweets to the queue
    stream_process.start()

    with open('remaining_accounts.json') as f:
        cred = json.load(f)

    creds = []

    for v in cred.values():
        creds.append(v)

    workers = [Process(target=run, args=(c, q, insert_q)) for c in creds]  # workers for requesting Twitter

    for w in workers:
        w.start()

    last_checked = None

    while True:  # check for recrawl
        last_checked = time.time()
        dbc.check_recrawl_db()
        time.sleep(600 - (time.time() - last_checked))  # check once per 10 min

