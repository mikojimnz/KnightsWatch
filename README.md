# The KnightsWatch

ML moderation bot for Reddit.

Uses machine learning to flag down comments that are possibly toxic.

Based from [python chatbot](https://techwithtim.net/tutorials/ai-chatbot/part-1/) by [Tech with Tim]([https://github.com/techwithtim](https://github.com/techwithtim)).

## Installation
1. Requires Python 3.6. Virtual environment recommended.
2. Use `pip3 install -r requirements.txt` to install dependencies.
3. Configure the `praw.ini` file, located at `env/lib/python3.6/site-packages/praw`, with your bot tokens. [PRAW Documentation](https://praw.readthedocs.io/en/latest/getting_started/configuration/prawini.html)
4. Edit the `settings.json` to match your PRAW configuration. Copy the sample configuration from the `sample-config` folder.
5. Create the `intents.json` file in the `training` folder. You can copy the sample configuration from the `sample-config` folder or modify the categories and labels.

## Training

### Method 1

**Use this method if there is no model trained yet.** This method will gather the latest comments and output them. Sort the comments and it will automatically populate the `intents.json` file for you.


1. Run the `collecter.py` script.
2. Comments will be collected automatically and will await user input.
3. Type:
	- a: If the comment is acceptable.
	- n: If the comment is neutral.
	- w: If the comment is considered to be  a warning.
	- Type any other character to skip the entry.
4. Press Enter to submit.
5. Rebuild the model as needed.

### Method 2

This method will gather the latest comments and output them. It will display what it currently thinks a comment is categorized as. Sort the comments and it will automatically populate the `intents.json` file for you.

1. Run the `collecter_trainer.py` script.
2. Comments will be collected automatically and will await user input.
3. Type:
	- a: If the comment is acceptable.
	- n: If the comment is neutral.
	- w: If the comment is considered to be  a warning.
	- Type any other character to skip the entry.
4. Press Enter to submit.
5. Rebuild the model as needed.

### Method 3

Manual entry method.

1. Run the `self_assign.py` script.
2. Enter a comment, it will be sanitized and re-outputted.
3. Type:
	- a: If the comment is acceptable.
	- n: If the comment is neutral.
	- w: If the comment is considered to be  a warning.
	- Type any other character to skip the entry.
4. Press Enter to submit.

## Notes

- Comments are sanitized of all punctuation and potential offending characters.
- Training data and models have been removed for public release.
