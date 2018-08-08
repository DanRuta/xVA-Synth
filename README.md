# xVA Synth

Tacotron based Electron app for voice synthesis in the style of specific voices and characters from Bethesda games.

The app serves as a framework, which loads and uses whichever models are given to it. As such, the app does nothing by itself, and models need to be installed. Models which have a corresponding asset file will be loaded in its respective game/category. Anything else gets loaded in the "Other" category.

<img width="100%" src="readme images/github-README.png">


## Instructions

Once the app is up and running, select the voice set category from the top left dropdown, then click a specific voice set.

A list of already synthesized audio files, if any, is displayed. For synthesis, click the `Load model` button. This may take a minute, on a slow machine.

Once finished, type your text in the text area and click the `Generate Voice` button. Once generated, you will be shown a preview of the output. Click the `Keep sample` button to save to file, or click the `Generate Voice` after making ammends to the text input, to discard it and re-generate.

In the below list of audio files, you can preview, click to open the containing folder, or delete each one.

**Note about synthesis quality**

Given the very small amount of data used in training, and the somewhat outdated synthesis code, the output is mediocre at best, and outright terrible at other times. Proper sentences can be still be created with trial and error.

The best approach I have found is to generate samples of at least 2 seconds in length, and not much more than 5. If you need a lot of text to be synthesized, the current best approach is to synthesize smaller clauses, and splicing them together in Audacity. If you need something really short, and it can't synthesize it, you can add a small sentence (EG `Some stuff.`) before and/or after your text, and cutting it out in Audacity.

All models have also been trained on my personal machine, with a GTX 1080, meaning that `batch_size` had to be limited, to stay within memory constraints. With any luck, I may get access to some beefier machines in the future to train models with better attention.

## Pro tips:

- Right click a voice set in the left bar to hear a sample of the voice.

- To change pronounciation, you can change spelling (cmudict is supported for models that were trained with it)

- You can use full stops and commas to change timing

- Try doing multiple takes, using different spellings, punctuation and input lengths.

- A model's first synthesis takes the longest. Subsequent ones are faster due to caching.

- Acronyms should be spelled out phonetically. EG: xVA -> Ex vee ay

- Numbers automatically get converted to text. However, if you need to pronounce years, such as 1990 -> nineteen ninety, instead of one thousand nine hundred and ninety, you should split the two numbers, like 19 90.

## Development

`npm install` dependencies, and run with `npm start`. Use virtualenv, and `pip install -r requirements.txt` using Python 3.6.

The app uses both JavaScript (Electron, UI) and Python code (Tacotron Model). As the python script needs to remain running alongside the app, and receive input, communication is done via an HTTP server, with the JavaScript code sending localhost requests, at port 8008. During development, the python source is used. In production, the compiled python is used.

## Packaging

Use pyinstaller to compile the python, and run the scripts in `package.json` to create the electron distributables.

## Models

The existing models have been trained on roughly 500k steps, each, at roughly 10 outputs_per_step, with batch_size of 16 (in order to be able to fit everything on 8GB of VRAM).

Currently, the following models have been trained:

### Skyrim:

<ul>
    <li>Female Even Toned</li>
    <li>Male Dunmer</li>
    <li>Male Soldier</li>
</ul>

### Oblivion:
<ul>
    <li>Male Breton</li>
</ul>

### Fallout 4:
<ul>
    <li>Nora (needs re-training)</li>
</ul>

## Future Plans

### App
This is an just early experiment. The quality of the voice files currently leaves to be desired, due to the low amount of data available. As technology improves, time permitting, the core synthesis algorithms will get updates, and if necessary, models retrained.

The app is capable of using CMUDict for the models that have been trained with it. So far, however, the models that I have trained with support for it have been of lower quality. However, if trained with with support for it, CMUDict syntax can be used in the input textarea.

### Models

Models are being trained for the following games:

- The Elder Scrolls V: Skyrim
- The Elder Scrolls IV: Oblivion
- Fallout 4

Time/interest/data permitting, other games/categories may be explored.

## Credits

Models are trained, and evaluation is done using code from keithito's implementation of Tacotron: https://github.com/keithito/tacotron