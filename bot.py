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

def restart_program():
    python = sys.executable
    os.execl(python, python, * sys.argv)

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
    queueStream = sub.mod.modqueue(limit=None)
    reportsStream = sub.mod.reports(limit=None)
    elevated_ch = client.get_channel(cfg['discord']['channels']['elevated'])
    modQueue_ch = client.get_channel(cfg['discord']['channels']['modQueue'])
    realtime_ch = client.get_channel(cfg['discord']['channels']['realtime'])
    unsure_ch = client.get_channel(cfg['discord']['channels']['unsure'])
    userWatch_ch = client.get_channel(cfg['discord']['channels']['userWatch'])
    submission_ch = client.get_channel(cfg['discord']['channels']['submissions'])
    modQueueIDs = []

    async def createEmbed(item=None):
        user = f'({item.author.name})' if (item.author.name in watchlist) else item.author.name

        if user in ignored: return None

        if type(item) == praw.models.reddit.comment.Comment:
            inp = sanatize_text(item.body)
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
                title = item.submission.title[:255],
                description = item.body[:2048],
                color = color,
                url = f'http://reddit.com{item.permalink}'
            )

            try:
                embed.set_author(name=f'{user}', icon_url=item.author.icon_img)
            except prawcore.exceptions.NotFound:
                print("prawcore.exceptions.NotFound Line 138")
                embed.set_author(name=f'*{user}*')
            except prawcore.exceptions.ServerError:
                exceptCnt += 1
                traceback.print_exc()
                await asyncio.sleep(60 * exceptCnt)

            embed.insert_field_at(index=0, name=f"{time.strftime('%b %d, %Y - %H:%M:%S UTC',  time.gmtime(item.created_utc))}. [{confidence:0.2f}%]", value=item.id)

            if tag == 'WARNING':
                await elevated_ch.send(embed=embed)

            if (results[results_index] < cfg['model']['confidence']):
                await unsure_ch.send(embed=embed)

            if (cfg['debug']['outputResults']):
                print(f'\n{inp}')
                cprint(f'\n    [{confidence:0.3f}% {tag}]', color[classification])
                print(f'    By: {user}\n    {link}\n')

        else:
            embed = discord.Embed(
                title = item.title[:255],
                description = item.link_flair_text,
                url = f'http://reddit.com{item.permalink}',
                color = discord.Colour.greyple()
            )

            try:
                embed.set_author(name=f'{user}', icon_url=item.author.icon_img)
            except prawcore.exceptions.NotFound:
                print("prawcore.exceptions.NotFound Line 172")
                embed.set_author(name=f'*{user}*')
            except prawcore.exceptions.ServerError:
                exceptCnt += 1
                traceback.print_exc()
                await asyncio.sleep(60 * exceptCnt)

            embed.insert_field_at(index=0, name=f"{time.strftime('%b %d, %Y - %H:%M:%S UTC',  time.gmtime(item.created_utc))}", value=item.id)

        if item.author.name in watchlist:
            await userWatch_ch.send(embed=embed)

        return embed;

    print(f'    {len(watchlist)} users in toolbox usernotes\n    {len(ignored)} users being ignored.')
    cprint("\n    Comment Stream Ready\n", 'green')
    await client.change_presence(status=discord.Status.online, activity=discord.Game(name='with Reddit'))

    while not client.is_closed():
        # print("looping")
        try:
            for comment in commentStream:
                if comment is None: break

                embed = await createEmbed(comment)
                if embed == None: continue
                await realtime_ch.send(embed=embed)

            for submission in submissionStream:
                if submission is None: break

                embed = await createEmbed(submission)
                if embed == None: continue
                await submission_ch.send(embed=embed)

            for item in queueStream:
                if item is None: break

                if item.id in modQueueIDs:
                    continue
                else:
                    if len(modQueueIDs) > 1000: modQueueIDs.clear()
                    modQueueIDs.append(item.id)
                    embed = await createEmbed(item)
                    if embed == None: continue
                    await modQueue_ch.send(embed=embed)

            for report in reportsStream:
                if report is None: break

                if report.id in modQueueIDs:
                    continue
                else:
                    if len(modQueueIDs) > 1000: modQueueIDs.clear()
                    modQueueIDs.append(report.id)
                    embed = await createEmbed(report)
                    if embed == None: continue
                    await modQueue_ch.send(embed=embed)

        except KeyboardInterrupt:
            sys.exit(1)
        except prawcore.exceptions.NotFound:
            print("prawcore.exceptions.NotFound Line 233")
            pass
        except prawcore.exceptions.ServerError as e:
            exceptCnt += 1
            traceback.print_exc()
            print(f'Reddit Server Error #{exceptCnt}\nSleeping for {60 * exceptCnt} seconds Line 237')
            await asyncio.sleep(60 * exceptCnt)
        except Exception as e:
            exceptCnt += 1
            traceback.print_exc()
            print(f'Exception #{exceptCnt}\nSleeping for {60 * exceptCnt} seconds Line 248')
            await client.change_presence(status=discord.Status.idle, activity=discord.Game(name='an exception. Check logs.'))
            await asyncio.sleep(60 * exceptCnt)

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
async def restart(ctx):
    await ctx.message.channel.send("Restarting Bot")
    restart_program()

@client.command()
async def crowdcontrol(ctx, *args):
    global sub

    if not args:
        await ctx.send('No argument found')
        return
    elif (isinstance(int(args[0]), int) and (0 <= int(args[0]) <= 3)):
        try:
            sub.mod.update(crowd_control_mode=True, crowd_control_level=args[0], crowd_control_chat_level=args[0])

            await ctx.send(f'Crowd Control set to {args[0]}')
        except Exception as e:
            await ctx.send(f'Error: `{e}`')
    else:
        await ctx.send(f'Invalid argument {args[0]}')
        return

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
async def on_raw_reaction_add(payload):
    channel = client.get_channel(payload.channel_id)
    msg = await channel.fetch_message(payload.message_id)

    if msg.embeds is None:
        await channel.send(f'Message does not contain data.')
        return

    embed = msg.embeds[0]

    if len(msg.reactions) > 1:
        await channel.send(f'Comment has already been moderated.')
        return

    async def addData(cat):
        inp = sanatize_text(embed.description)

        with open("training/intents.json", "r+") as jsonFile2:
            tmp = json.load(jsonFile2)
            tmp['intents'][cat]['patterns'].append(inp)
            jsonFile2.seek(0)
            json.dump(tmp, jsonFile2)
            jsonFile2.truncate()

        await channel.send(f'Comment added to training data: `{inp[:25]}`')
        print(f'{payload.emoji}: {inp}')

    async def reaction_remove(rule):
        id = embed.fields[0].value

        try:
            item = reddit.submission(id) if (embed.color == discord.Colour.greyple()) else reddit.comment(id)
            item.mod.remove(spam=False, mod_note=f'KnightsWatch Removal - Rule {rule + 1}')
            reply = item.mod.send_removal_message(title='ignored', type='public', message=f'Removed. Reason:\n> {sub.rules[rule]}')
            reply.mod.lock()

            await channel.send(f'Removed {id} for `{sub.rules[rule]}`')
        except Exception as e:
            await channel.send(f'Error: `{e}`')

    async def reaction_lock():
        id = embed.fields[0].value

        try:
            item = reddit.submission(id) if (embed.color == discord.Colour.greyple()) else reddit.comment(id)
            item.mod.lock()

            await channel.send(f'Locked `{id}`')
        except Exception as e:
            await channel.send(f'Error: `{e}`')

    async def reaction_unlock():
        id = embed.fields[0].value

        try:
            item = reddit.submission(id) if (embed.color == discord.Colour.greyple()) else reddit.comment(id)
            item.mod.unlock()

            await channel.send(f'Unlocked `{id}`')
        except Exception as e:
            await channel.send(f'Error: `{e}`')

    async def reaction_nuke():
        if (embed.color == discord.Colour.greyple()):
            return

        try:
            item = reddit.comment(embed.fields[0].value)
            item.refresh()
            cnt = len(item.replies.list())

            for reply in item.replies.list():

                reply.mod.remove()

            item.mod.remove()

            await channel.send(f'Nuked `{item.id}` with {cnt} replies')
        except Exception as e:
            await channel.send(f'Error: `{e}`')

    if payload.emoji.name == u'✅':
        await addData(0)
    elif payload.emoji.name == u'🆗':
        await addData(1)
    elif payload.emoji.name == u'❌':
        await addData(2)
    elif payload.emoji.name == u'0️⃣':
        await reaction_remove(9)
    elif payload.emoji.name == u'1️⃣':
        await reaction_remove(0)
    elif payload.emoji.name == u'2️⃣':
        await reaction_remove(1)
    elif payload.emoji.name == u'3️⃣':
        await reaction_remove(2)
    elif payload.emoji.name == u'4️⃣':
        await reaction_remove(3)
    elif payload.emoji.name == u'5️⃣':
        await reaction_remove(4)
    elif payload.emoji.name == u'6️⃣':
        await reaction_remove(5)
    elif payload.emoji.name == u'7️⃣':
        await reaction_remove(6)
    elif payload.emoji.name == u'8️⃣':
        await reaction_remove(7)
    elif payload.emoji.name == u'9️⃣':
        await reaction_remove(8)
    elif payload.emoji.name == u'🔒':
        await reaction_lock()
    elif payload.emoji.name == u'🔓':
        await reaction_unlock()
    elif payload.emoji.name == u'☢️':
        await reaction_nuke()
    else:
        await channel.send(f'Unknown reaction. Remove reaction to moderate.')
        return

client.run(cfg['discord']['clientID'])
