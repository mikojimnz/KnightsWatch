import asyncio
import base64
import discord
import json
import nltk
import numpy
import os
import pickle
import praw
import prawcore
import random
import re
import signal
import sys
import tensorflow
import tflearn
import time
import traceback
import zlib

from discord.ext import commands, tasks
from nltk.stem.lancaster import LancasterStemmer
from termcolor import colored, cprint
from time import sleep

sub = None
reddit = None
watchlist = []
ignored = []
exceptCnt = 0

def main():
    with open("settings.json") as jsonFile1:
        cfg = json.load(jsonFile1)

    with open('training/intents.json') as jsonFile2:
        data = json.load(jsonFile2)

    with open("model/data.pickle", "rb") as p:
        words, labels, training, output = pickle.load(p)

    def sanatize_text(input):
        raw = " ".join(input.lower().splitlines())
        raw = re.sub(r'(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))', ' ', raw, flags=re.MULTILINE)
        raw = re.sub(r'([\'’])', '', raw)
        raw = re.sub(r'[^a-z\s]', ' ', raw)
        raw = re.sub(r'[ ]+', ' ', raw.strip())
        return re.sub(r'( x b )|( nbsp )', ' ', raw)

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
    "WARNING": "red"
    }

    os.system('cls' if os.name == 'nt' else 'clear')

    @tasks.loop(seconds=0)
    async def read_comments():
        global reddit
        global sub
        global watchlist
        global ignored
        global exceptCnt

        await client.wait_until_ready()
        commentStream = sub.stream.comments(skip_existing=cfg['praw']['skipExisting'], pause_after=-1)
        submissionStream = sub.stream.submissions(skip_existing=cfg['praw']['skipExisting'], pause_after=-1)
        elevated_ch = client.get_channel(cfg['discord']['channels']['elevated'])
        realtime_ch = client.get_channel(cfg['discord']['channels']['realtime'])
        unsure_ch = client.get_channel(cfg['discord']['channels']['unsure'])
        userWatch_ch = client.get_channel(cfg['discord']['channels']['userWatch'])
        submission_ch = client.get_channel(cfg['discord']['channels']['submissions'])

        print(f'    {len(watchlist)} users in toolbox usernotes\n    {len(ignored)} users being ignored.')
        cprint("\n    Comment Stream Ready\n", 'green')
        await client.change_presence(status=discord.Status.online, activity=discord.Game(name='with Reddit'))

        while not client.is_closed():
            try:
                for comment in commentStream:
                    if comment is None:
                        break

                    user = f'({comment.author.name})' if (comment.author.name in watchlist) else comment.author.name
                    link = f'http://reddit.com{comment.permalink}'
                    inp = sanatize_text(comment.body)

                    if (len(inp) <= 0) or (user in ignored):
                        continue

                    results = model.predict([bag_of_words(inp, words)])[0]
                    results_index = numpy.argmax(results)
                    tag = labels[results_index].upper()
                    confidence = results[results_index] * 100

                    if (results[results_index] < cfg['model']['confidence']):
                        color = discord.Colour.purple()
                    elif tag == 'WARNING':
                        color = discord.Colour.red()
                    elif tag == 'NEUTRAL':
                        color = discord.Colour.lighter_gray()
                    else:
                        color = discord.Colour.green()

                    embed = discord.Embed(
                        title = comment.submission.title[:255],
                        description = comment.body[:2048],
                        color = color,
                        url = link
                    )

                    try:
                        embed.set_author(name=f'{user}', icon_url=comment.author.icon_img)
                    except NotFound:
                        embed.set_author(name=f'*{user}*')

                    embed.insert_field_at(index=0, name=f"{time.strftime('%b %d, %Y - %H:%M:%S UTC',  time.gmtime(comment.created_utc))}. [{confidence:0.2f}%]", value=comment.id)
                    await realtime_ch.send(embed=embed)

                    if tag == 'WARNING':
                        await elevated_ch.send(embed=embed)

                    if comment.author.name in watchlist:
                        await userWatch_ch.send(embed=embed)

                    if (results[results_index] < cfg['model']['confidence']):
                        await unsure_ch.send(embed=embed)

                    if (cfg['debug']['outputResults']):
                        print(f'\n{inp}')
                        cprint(f'\n    [{confidence:0.3f}% {tag}]', color[classification])
                        print(f'    By: {user}\n    {link}\n')

                for submission in submissionStream:
                    if submission is None:
                        break

                    embed = discord.Embed(
                        title = submission.title[:255],
                        description = submission.link_flair_text,
                        url = f'http://reddit.com{submission.permalink}',
                        color = discord.Colour.greyple()
                    )
                    user = f'({submission.author.name})' if (submission.author.name in watchlist) else submission.author.name

                    try:
                        embed.set_author(name=f'{user}', icon_url=submission.author.icon_img)
                    except NotFound:
                        embed.set_author(name=f'*{user}*')

                    embed.insert_field_at(index=0, name=f"{time.strftime('%b %d, %Y - %H:%M:%S UTC',  time.gmtime(submission.created_utc))}", value=submission.id)
                    await submission_ch.send(embed=embed)

                    if submission.author.name in watchlist:
                        await userWatch_ch.send(embed=embed)

            except KeyboardInterrupt:
                sys.exit(1)
            except prawcore.exceptions.NotFound:
                pass
            except Exception as e:
                await client.change_presence(status=discord.Status.idle, activity=discord.Game(name='an exception. Check logs.'))
                traceback.print_exc()
                exceptCnt += 1
                print(f'Exception #{exceptCnt}\nSleeping for {60 * exceptCnt} seconds')
                sleep(60 * exceptCnt)

            await asyncio.sleep(1)

    @read_comments.before_loop
    async def load_lists():
        global reddit
        global sub
        global watchlist
        global ignored

        if (cfg['praw']['toolbox']['monitorUsers']):
            watchlist.clear()
            usernotesJson = json.loads(sub.wiki[cfg['praw']['toolbox']['usernotePage']].content_md)
            decompressed = zlib.decompress(base64.b64decode(usernotesJson['blob']))
            for user in json.loads(decompressed).keys():
                 watchlist.append(user)

        ignored.clear()
        wikiConfig = json.loads(sub.wiki[cfg['praw']['wikiConfig']].content_md)
        ignored = wikiConfig['ignored']

    @client.event
    async def on_ready():
        global reddit
        global sub
        reddit = praw.Reddit(cfg['praw']['cred'])
        sub = reddit.subreddit(cfg['praw']['sub'])

        cprint(f'    Discord connection established, logged in as {client.user}', 'green')
        read_comments.start()

    @client.command()
    async def clearexcpt(ctx):
        global exceptCnt
        exceptCnt = 0
        read_comments.restart()
        await client.change_presence(status=discord.Status.online, activity=discord.Game(name='with Reddit'))
        await ctx.message.channel.send("Clearing Exception")

    @client.command()
    async def ping(ctx):
        await ctx.message.channel.send("Pong!")

    @client.command()
    async def reload(ctx, *args):
        global reddit
        global sub
        global watchlist
        global ignored

        if not args:
            await ctx.send('No argument found')
            return
        elif ((args[0] == 'watchlist') and (cfg['praw']['toolbox']['monitorUsers'])):
            watchlist.clear()
            usernotesJson = json.loads(sub.wiki[cfg['praw']['toolbox']['usernotePage']].content_md)
            decompressed = zlib.decompress(base64.b64decode(usernotesJson['blob']))
            for user in json.loads(decompressed).keys():
                 watchlist.append(user)

            print(f'    {len(watchlist)} users in toolbox usernotes')
            await ctx.send(f'Watchlist reloaded with {len(watchlist)} users in toolbox usernotes')
        elif (args[0] == 'ignored'):
            ignored.clear()
            wikiConfig = json.loads(sub.wiki[cfg['praw']['wikiConfig']].content_md)
            ignored = wikiConfig['ignored']

            print(f'    {len(ignored)} users being ignored.')
            await ctx.send(f'Ignored reloaded with {len(ignored)} users in config')
        else:
            await ctx.send(f'Invalid argument {args}')
            return

    @client.command()
    async def remove(ctx, *args):
        global reddit
        global sub

        if not args or (len(locals()) != 2):
            await ctx.send('Invlaid arguments found. !remove [Comment URL | Comment ID ] (Rule Number)')
            return

        pattern = re.search(r'http[s]?:\/\/reddit.com\/r\/[\w]+\/comments\/[\w\d]+\/-\/([\w\d]+)/|(^[\w\d]+$)', args[0])

        if pattern:
            if pattern.group(1) == None:
                id = pattern.group(2)
            else:
                id = pattern.group(1)

            try:
                rule = int(args[1]) - 1
                comment = reddit.comment(id)
                comment.mod.remove(spam=False, mod_note=f'KnightsWatch Removal - Rule {args[1]}')
                reply = comment.mod.send_removal_message(title='ignored', type='public', message=f'Removed. Reason:\n> {sub.rules[rule]}')
                reply.mod.lock()

                await ctx.send(f'Removed {id} for `{sub.rules[rule]}`')
            except Exception as e:
                await ctx.send(f'Error: `{e}`')

        else:
            await ctx.send(f'Unknown args1 {args[0]}')
            return

    @client.event
    async def on_reaction_add(reaction, user):

        if len(reaction.message.reactions) > 1:
            await reaction.message.channel.send(f'Comment has already been moderated.')
            return

        if reaction.message.embeds is None:
            await reaction.message.channel.send(f'Message does not contain data.')
            return

        async def addData(cat):
            inp = sanatize_text(reaction.message.embeds[0].description)

            with open("training/intents.json", "r+") as jsonFile2:
                tmp = json.load(jsonFile2)
                tmp['intents'][cat]['patterns'].append(inp)
                jsonFile2.seek(0)
                json.dump(tmp, jsonFile2)
                jsonFile2.truncate()

            await reaction.message.channel.send(f'Comment added to training data: `{inp[:25]}`')
            print(f'{reaction.emoji}: {inp}')

        async def reactionRemove(rule):
                id = reaction.message.embeds[0].fields[0].value

                try:
                    item = reddit.submission(id) if (reaction.message.embeds[0].color == discord.Colour.greyple()) else reddit.comment(id)
                    item.mod.remove(spam=False, mod_note=f'KnightsWatch Removal - Rule {rule + 1}')
                    reply = item.mod.send_removal_message(title='ignored', type='public', message=f'Removed. Reason:\n> {sub.rules[rule]}')
                    reply.mod.lock()

                    await reaction.message.channel.send(f'Removed {id} for `{sub.rules[rule]}`')
                except Exception as e:
                    await reaction.message.channel.send(f'Error: `{e}`')

        if reaction.emoji == cfg['discord']['reactions']['acceptable']:
            await addData(0)
        elif reaction.emoji == cfg['discord']['reactions']['neutral']:
            await addData(1)
        elif reaction.emoji == cfg['discord']['reactions']['warning']:
            await addData(2)
        elif reaction.emoji == "1️⃣":
            await reactionRemove(0)
        elif reaction.emoji == "2️⃣":
            await reactionRemove(1)
        elif reaction.emoji == "3️⃣":
            await reactionRemove(2)
        elif reaction.emoji == "4️⃣":
            await reactionRemove(3)
        elif reaction.emoji == "5️⃣":
            await reactionRemove(4)
        elif reaction.emoji == "6️⃣":
            await reactionRemove(5)
        elif reaction.emoji == "7️⃣":
            await reactionRemove(6)
        elif reaction.emoji == "8️⃣":
            await reactionRemove(7)
        elif reaction.emoji == "9️⃣":
            await reactionRemove(8)
        elif reaction.emoji == "0️⃣":
            await reactionRemove(9)
        else:
            await reaction.message.channel.send(f'Unknown reaction. Remove reaction to moderate.')
            return

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
