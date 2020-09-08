import asyncio
import base64
import discord
import json
import nltk
import numpy
import os
import pickle
import praw
import random
import re
import signal
import sys
import tensorflow
import tflearn
import time
import traceback
import zlib

from discord.ext import commands
from nltk.stem.lancaster import LancasterStemmer
from termcolor import colored, cprint
from time import sleep

CONST_REG = r'(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))'

sub = None
reddit = None
watchlist = []

def main():
    with open("settings.json") as jsonFile1:
        cfg = json.load(jsonFile1)

    with open('training/intents.json') as jsonFile2:
        data = json.load(jsonFile2)

    with open("model/data.pickle", "rb") as p:
        words, labels, training, output = pickle.load(p)

    def bag_of_words(s, words):
        bag = [0 for _ in range(len(words))]

        s_words = nltk.word_tokenize(s)
        s_words = [stemmer.stem(word.lower()) for word in s_words]

        for se in s_words:
            for i, w in enumerate(words):
                if w == se:
                    bag[i] = 1

        return numpy.array(bag)

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    stemmer = LancasterStemmer()
    tensorflow.reset_default_graph()
    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)
    model = tflearn.DNN(net)
    model.load("model/model.tflearn")

    client = commands.Bot(command_prefix=cfg['discord']['cmdPrefix'])

    color = {
    "ACCEPTABLE": "green",
    "NEUTRAL": "white",
    "POSSIBLE WARNING": "red"
    }

    os.system('cls' if os.name == 'nt' else 'clear')

    async def read_comments():
        global reddit
        global sub
        global watchlist
        await client.wait_until_ready()

        commentStream = sub.stream.comments(skip_existing=cfg['praw']['skipExisting'], pause_after=-1)

        if (cfg['praw']['toolbox']['monitorUsers']):
            usernotesJson = json.loads(sub.wiki[cfg['praw']['toolbox']['usernotePage']].content_md)
            decompressed = zlib.decompress(base64.b64decode(usernotesJson['blob']))
            for user in json.loads(decompressed).keys():
                 watchlist.append(user)

            print(f'{len(watchlist)} users in toolbox usernotes')

        elevated_ch = client.get_channel(cfg['discord']['channels']['elevated'])
        realtime_ch = client.get_channel(cfg['discord']['channels']['realtime'])
        unsure_ch = client.get_channel(cfg['discord']['channels']['unsure'])
        userWatch_ch = client.get_channel(cfg['discord']['channels']['userWatch'])

        cprint("\n    Comment Stream Ready\n", 'green')

        while not client.is_closed():
            try:
                for comment in commentStream:
                    if comment is None:
                        break

                    raw = " ".join(comment.body.lower().splitlines())
                    raw = re.sub(CONST_REG, ' ', raw, flags=re.MULTILINE)
                    raw = re.sub(r'([\'’])', '', raw)
                    raw = re.sub(r'[^a-z\s]', ' ', raw)
                    raw = re.sub(r'[ ]+', ' ', raw.strip())
                    inp = re.sub(r'( x b )|( nbsp )', ' ', raw)
                    user = comment.author.name
                    link = comment.permalink.replace(re.search(r'/r/[\w]+/comments/[\w\d]+/([\w\d_]+)/[\w\d]+/', comment.permalink).group(1), '-', 1)

                    if (len(inp) <= 0):
                        continue

                    results = model.predict([bag_of_words(inp, words)])[0]
                    results_index = numpy.argmax(results)
                    tag = labels[results_index]
                    confidence = results[results_index] * 100

                    if (results[results_index] > cfg['model']['confidence']):
                        for tg in data["intents"]:
                            if tg['tag'] == tag:
                                classification = tg['classification']

                        if (classification == 'POSSIBLE WARNING'):
                            await elevated_ch.send(f'**[{confidence:0.3f}% {classification}]** By: {user}\n```{comment.body}```\n<http://reddit.com{link}>')
                        else:
                            await realtime_ch.send(f'**[{confidence:0.3f}% {classification}]** By: {user}\n```{comment.body}```\n<http://reddit.com{link}>')

                        if (comment.author.name in watchlist):
                            await userWatch_ch.send(f'**[{confidence:0.3f}% {classification}]** By: {user}\n```{comment.body}```\n<http://reddit.com{link}>')

                        if (cfg['debug']['outputResults']):
                            print(f'\n{inp}')
                            cprint(f'\n    [{confidence:0.3f}% {classification}]', color[classification])
                            print(f'    By: {user}\n    http://reddit.com{link}\n')

                    else:
                        await unsure_ch.send(f"**[UNSURE {confidence:0.3f}% {tg['classification']}]** By: {user}\n```{comment.body}```\n<http://reddit.com{link}>")

                        if (comment.author.name in watchlist):
                            await userWatch_ch.send(f"**[UNSURE {confidence:0.3f}% {tg['classification']}]** By: {user}\n```{comment.body}```\n<http://reddit.com{link}>")

                        if (cfg['debug']['outputResults']):
                            print(f'\n{inp}')
                            cprint(f'\n    [UNSURE {confidence:0.3f}% {tag}]', 'cyan')
                            print(f'    By: {user}\n    http://reddit.com{link}\n')

            except KeyboardInterrupt:
                sys.exit(1)
            except Exception as e:
                print(f'EXCEPTION:\n{e}')

            await asyncio.sleep(1)

    @client.event
    async def on_ready():
        global reddit
        global sub
        reddit = praw.Reddit(cfg['praw']['cred'])
        sub = reddit.subreddit(cfg['praw']['sub'])

        cprint(f'    Discord connection established, logged in as {client.user}', 'green')
        client.loop.create_task(read_comments())

    @client.command()
    async def ping(ctx):
        await ctx.message.channel.send("Pong!")

    @client.command()
    async def reload(ctx, *args):
        global reddit
        global sub
        global watchlist

        if not args:
            await ctx.send('No argument found')
        elif ((args[0] == 'watchlist') and (cfg['praw']['toolbox']['monitorUsers'])):
            watchlist = []
            usernotesJson = json.loads(sub.wiki[cfg['praw']['toolbox']['usernotePage']].content_md)
            decompressed = zlib.decompress(base64.b64decode(usernotesJson['blob']))
            for user in json.loads(decompressed).keys():
                 watchlist.append(user)

            print(f'{len(watchlist)} users in toolbox usernotes')
            await ctx.send(f'Watchlist reloaded with {len(watchlist)} users in toolbox usernotes')
        else:
            await ctx.send(f'Invalid argument {args}')


    @client.event
    async def on_reaction_add(reaction, user):
        src = reaction.message.channel.id

        if re.search(r'```([\w\d\s\W\D]+)```', reaction.message.content) is None:
            return

        if reaction.emoji == cfg['discord']['reactions']['acceptable']:
            cat = 0
        elif reaction.emoji == cfg['discord']['reactions']['neutral']:
            cat = 1
        elif reaction.emoji == cfg['discord']['reactions']['warning']:
            cat = 2
        else:
            await reaction.message.channel.send(f'Unknown signifer. Remove reaction to reclassify.')
            return

        if len(reaction.message.reactions) > 1:
            await reaction.message.channel.send(f'Comment has already been reclassaified.')
            return

        raw = re.search(r'```([\w\d\s\W\D]+)```', reaction.message.content.lower())
        raw = re.sub(CONST_REG, ' ', raw.group(1))
        raw = re.sub(r'([\'’])', '', raw)
        raw = re.sub(r'[^a-z ]', ' ', raw)
        raw = re.sub(r'[ ]+', ' ', raw.strip())
        inp = re.sub(r'( x b )|( nbsp )', ' ', raw)

        with open("training/intents.json", "r+") as jsonFile2:
            tmp = json.load(jsonFile2)
            tmp['intents'][cat]['patterns'].append(inp)
            jsonFile2.seek(0)
            json.dump(tmp, jsonFile2)
            jsonFile2.truncate()

        await reaction.message.channel.send(f'Comment added to training data: `{inp[:25]}`')
        print(f'{reaction.emoji}: {inp}')

    client.run(cfg['discord']['clientID'])

def exit_gracefully(signum, frame):
    signal.signal(signal.SIGINT, original_sigint)

    try:
        if input("\nDo you really want to quit? (y/n)> ").lower().startswith('y'):
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nQuitting")
        sys.exit(1)

    signal.signal(signal.SIGINT, exit_gracefully)

if __name__ == "__main__":
    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, exit_gracefully)
    main()
