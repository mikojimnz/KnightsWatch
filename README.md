# The KnightsWatch

ML moderation bot for Reddit.

Uses machine learning to flag down comments that are possibly toxic.

Based from [python chatbot](https://techwithtim.net/tutorials/ai-chatbot/part-1/) by [Tech with Tim]([https://github.com/techwithtim](https://github.com/techwithtim)).

## Installation
1. Requires Python 3.6. Virtual environment recommended.
2. Use `pip3 install -r requirements.txt` to install dependencies.
3. Run `python3 generate_model.py` to build model.
4. Configure the `praw.ini` file, located at `env/lib/python3.6/site-packages/praw`, with your bot tokens.
5. Edit the `settings.json` to match your PRAW configuration.

## Training

### Method 1 (Preferred)

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

### Method 2

Manually edit the `training/intents.json` file and paste in comments in the appropriate categories: Acceptable, Neutral, or Warning.


## Notes

- Comments are sanitized of all punctuation and potential offending characters.

## Future Projects
 - Integrate Reddit ModTools
 - Discord integration
