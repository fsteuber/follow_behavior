import numpy as np
#from matplotlib import pyplot as plt
#import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kde
import pandas as pd
import pickle

def write_pickle_file():
    df = pd.read_csv('follow_data.csv', sep=", ",
                     names=["uid", "oid", "ufriends", "ofriends", "ufollowers", "ofollowers", "ulisted", "olisted",
                            "uactivity", "oactivity", "following", "followed", "type", "delay", "geotagged"])
    pickle.dump(df, open('follow_data.pickle', 'wb'))


def is_following_per_delay(df):
    labels = []

    for category in (0, 1, 2, 3):
        tmp = df[df['delay'] == category]
        ncount = len(tmp)

        nf = len(tmp[tmp['following'] == True])
        nnf = len(tmp[tmp['following'] == False])

        labels.append('{:.1f}%'.format(100. * nnf / ncount))
        labels.append('{:.1f}%'.format(100. * nf / ncount))

    labels = [labels[0],labels[2],labels[4],labels[6],labels[1],labels[3],labels[5],labels[7]]

    ax = sns.countplot(data=df, x="delay", hue="following")

    plt.xlabel('Verzögerung')
    plt.xticks((0,1,2,3), ('beim Antworten', 'nach 1h', 'nach 1d', 'nach 2d'))
    plt.ylabel('Häufigkeit [log10]')
    plt.yscale("log")
    plt.title('Nutzer folgt dem Author des Originaltweets')

    for i, p in enumerate(ax.patches):
        x = p.get_bbox().get_points()[:, 0]
        y = p.get_bbox().get_points()[1, 1]
        ax.annotate(labels[i], (x.mean(), y), ha='center', va='bottom')  # set the alignment of the text

    plt.show()

def is_following_per_type(df):
    labels = []
    ddf = pd.DataFrame(columns=('type', 'following', 'delay'))

    for category in (0, 1, 2, 3):
        tmp = df[df['delay'] == category]

        for t in ('rt', 'r', 'q'):  # ('\'rt\'', '\'r\'', '\'q\''):
            ttmp = tmp[tmp['type']==t]
            ncount = len(ttmp)

            nf = len(ttmp[ttmp['following'] == True])
            la = '{:.1f}'.format(100. * nf / ncount)

            ddf = ddf.append({'type': "Retweet" if t == "rt" else "Reply" if t == "r" else "Quote",
                              'following': float(la),
                              'delay': category},
                             ignore_index=True)
            nnf = len(ttmp[ttmp['following'] == False])
            #labels.append('{:.1f}%'.format(100. * nnf / ncount))
            labels.append(la+"%")

    print(labels)
    labels = [labels[0],labels[3],labels[6],labels[9],labels[1],labels[4],labels[7],labels[10],labels[2],labels[5],labels[8],labels[11]]
    print(ddf)
    # ax = sns.countplot(data=ddf, x="delay", hue="type")
    ax = sns.barplot(data=ddf, hue='type', x='delay', y='following')

    plt.xlabel('Verzögerung')
    plt.xticks((0,1,2,3), ('beim Antworten', 'nach 1h', 'nach 1d', 'nach 2d'))
    plt.ylabel('Häufigkeit [%]')
    plt.yscale('log')
    plt.title('Nutzer folgt dem Author des Originaltweets')


    for i, p in enumerate(ax.patches):
        x = p.get_bbox().get_points()[:, 0]
        y = p.get_bbox().get_points()[1, 1]
        ax.annotate(labels[i], (x.mean(), y), ha='center', va='bottom')  # set the alignment of the text

    plt.show()

def popularity_density_follow(data_frame):
    df = data_frame
    df['ufollowers'] = df['ufollowers'].apply(lambda n: max(0, np.log10(n)))
    df['ofollowers'] = df['ofollowers'].apply(lambda n: max(0, np.log10(n)))

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(5, 8))

    nbins = 200

    tmp = df[df['following'] == True]
    tmp = tmp[tmp['followed'] == True]

    x = tmp['ufollowers'].to_numpy()
    y = tmp['ofollowers'].to_numpy()

    s = np.vstack((x, y))

    k = kde.gaussian_kde(s)
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    axes[0][0].set_title('both follow')
    axes[0][0].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
    axes[0][0].contour(xi, yi, zi.reshape(xi.shape))
    axes[0][0].set_ylabel("oFollowers [log10]")

    tmp = df[df['following'] == False]
    tmp = tmp[tmp['followed'] == True]

    x = tmp['ufollowers'].to_numpy()
    y = tmp['ofollowers'].to_numpy()

    s = np.vstack((x, y))

    k = kde.gaussian_kde(s)
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    axes[0][1].set_title('only is followed')
    axes[0][1].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
    axes[0][1].contour(xi, yi, zi.reshape(xi.shape))

    tmp = df[df['following'] == True]
    tmp = tmp[tmp['followed'] == False]

    x = tmp['ufollowers'].to_numpy()
    y = tmp['ofollowers'].to_numpy()

    s = np.vstack((x, y))

    k = kde.gaussian_kde(s)
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    axes[1][0].set_title('only follows')
    axes[1][0].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
    axes[1][0].contour(xi, yi, zi.reshape(xi.shape))
    axes[1][0].set_ylabel("oFollowers [log10]")
    axes[1][0].set_xlabel("uFollowers [log10]")

    tmp = df[df['following'] == False]
    tmp = tmp[tmp['followed'] == False]

    x = tmp['ufollowers'].to_numpy()
    y = tmp['ofollowers'].to_numpy()

    s = np.vstack((x, y))

    k = kde.gaussian_kde(s)
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    axes[1][1].set_title('neither follow')
    axes[1][1].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
    axes[1][1].contour(xi, yi, zi.reshape(xi.shape))
    axes[1][1].set_xlabel("uFollowers [log10]")

    #plt.show()
    plt.savefig("follow_behavior.png", bbox_inch="tight")
    #plt.savefig("follow_behavior.pdf", bbox_inch="tight")


def popularity_density_follow_hist2d(data_frame):
    df = data_frame
    df['ufollowers'] = df['ufollowers'].apply(lambda n: max(0, np.log10(n)))
    df['ofollowers'] = df['ofollowers'].apply(lambda n: max(0, np.log10(n)))

    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8, 5))

    nbins = 250

    tmp = df[df['following'] == True]
    tmp = tmp[tmp['followed'] == True]

    x = tmp['ufollowers'].to_numpy()
    y = tmp['ofollowers'].to_numpy()

    axes[0].set_title('both follow')
    axes[0].hist2d(x, y, bins=nbins, cmap=plt.cm.BuGn_r)
    axes[0].set_ylabel("oFollowers [log10]")
    axes[0].set_xlabel("uFollowers [log10]")

    tmp = df[df['following'] == False]
    tmp = tmp[tmp['followed'] == False]

    x = tmp['ufollowers'].to_numpy()
    y = tmp['ofollowers'].to_numpy()

    axes[1].set_title('neither follow')
    axes[1].hist2d(x, y, bins=nbins, cmap=plt.cm.BuGn_r)
    axes[1].set_xlabel("uFollowers [log10]")

    plt.savefig("follow_behavior2.pdf", bbox_inch="tight")
    plt.savefig("follow_behavior2.png", bbox_inch="tight")
    # plt.show()



def is_following_per_geotag(df):
    labels = []
    df['geolokalisiert'] = df['geotagged'].apply(lambda n: bool(n))
    ddf = pd.DataFrame(columns=('Geotagged', 'following', 'delay'))
    print(len(df[df["geotagged"] == 0]), len(df[df["geotagged"] == 1]))

    for category in (0, 1, 2, 3):
        tmp = df[df['delay'] == category]
        print(len(tmp[tmp["geotagged"] == 0]), len(tmp[tmp["geotagged"] == 1]))

        for g in (True, False):
            ttmp = tmp[tmp['geotagged']==g]
            ncount = len(ttmp)

            nf = len(ttmp[ttmp['following'] == True])
            la = '{:.1f}'.format(100. * nf / ncount)

            ddf = ddf.append({'Geotagged': g,
                              'following': float(la),
                              'delay': category},
                             ignore_index=True)
            nnf = len(ttmp[ttmp['following'] == False])
            #labels.append('{:.1f}%'.format(100. * nnf / ncount))
            labels.append(la+"%")

    print(labels)
    labels = [labels[0],labels[2],labels[4],labels[6],labels[1],labels[3],labels[5],labels[7]]
    print(ddf)
    ax = sns.countplot(data=df, x="delay", hue="geolokalisiert")
    #ax = sns.barplot(data=ddf, hue='Geotagged', x='delay', y='following')
    #ax = sns.barplot(data=df, hue='geotagged', x='delay', y='following')

    plt.xlabel('Verzögerung')
    plt.xticks((0,1,2,3), ('beim Antworten', 'nach 1h', 'nach 1d', 'nach 2d'))
    plt.ylabel('Häufigkeit von geolokalisierten Tweets [log10]')
    plt.yscale('log')
    plt.title('Follow-Rate bei Differenzierung bezüglich Geolokalisation')


    for i, p in enumerate(ax.patches):
        x = p.get_bbox().get_points()[:, 0]
        y = p.get_bbox().get_points()[1, 1]
        ax.annotate(labels[i], (x.mean(), y), ha='center', va='bottom')  # set the alignment of the text

    plt.show()

def is_following_per_listed(data_frame):
    df = data_frame
    df['olisted'] = df['olisted'].apply(lambda n: max(0, np.log10(n)))

    fig, ax = plt.subplots()
    plt.title("Anzahl an öffentlichen Listen, in denen der Quellnutzer auftaucht")
    sns.distplot(df['olisted'], ax=ax, label="im gesamten Datensatz")
    ax.set_ylabel('Häufigkeit [%]')

    tmp = df[df['following'] == True]
    tmp = tmp[tmp['followed'] == False]

    sns.distplot(tmp['olisted'], ax=ax, label="eingeschränkt auf Nutzer, denen gefolgt wird")
    ax.legend()
    ax.set_xlabel("Anzahl an Listen [log10]")
    plt.show()

def is_following_per_activity(data_frame):
    df = data_frame
    df['uactivity'] = df['uactivity'].apply(lambda n: max(0, np.log10(n)))

    fig, ax = plt.subplots()
    plt.title("Aktivität des Quellnutzers")
    sns.distplot(df['uactivity'], ax=ax, label="im gesamten Datensatz")
    ax.set_ylabel('Häufigkeit [%]')

    tmp = df[df['followed'] == True]
    #tmp = tmp[tmp['followed'] == False]

    sns.distplot(tmp['uactivity'], ax=ax, label="eingeschränkt auf Nutzer, denen gefolgt wird")
    ax.legend()
    ax.set_xlabel("Tweets pro Tag [log10]")
    plt.show()


if __name__ == '__main__':
    sns.set()

    df = pickle.load(open('follow_data.pickle', 'rb'))
    df['type'] = df['type'].apply(lambda n: 'r' if n == '\'r\'' else 'rt' if n== '\'rt\'' else 'q')
    df['following'] = df['following'].apply(lambda n: bool(n))
    #is_following_per_delay(df)
    #is_following_per_type(df)
    #is_following_per_geotag(df)
    #is_following_per_listed(df)
    #is_following_per_activity(df)
    popularity_density_follow(df)
