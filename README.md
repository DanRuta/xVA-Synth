# xVA Synth

xVASynth is a machine learning based speech synthesis app, using voices from characters/voice sets from Bethesda games.

## See it in action (YouTube link)

<a href="http://www.youtube.com/watch?feature=player_embedded&v=ofN0yMI9PNI
" target="_blank"><img src="http://img.youtube.com/vi/ofN0yMI9PNI/0.jpg"
alt="xVASynth YouTube demo" width="240" height="180" border="10" /></a>


<img width="100%" src="readme images/github-README.png">

This is an Electron UI wrapped around inference of FastPitch models trained on voice data from video games. The app serves as a framework, which loads and uses whichever models are given to it. As such, the app does nothing by itself, and models need to be installed. Models which have a corresponding asset file will be loaded in their respective game/category. Anything else gets loaded in the "Other" category.

The main benefit of this tool is allowing mod creators to generate new voice lines for third party game modifications (mods). There are other uses, such as creating machinima, and just generally having fun with familiar voices.

Join chat on Discord here: https://discord.gg/nv7c6E2TzV


## Installation
The base application can be downloaded and placed anywhere. Aim to install it onto an SSD, if you have the space, to reduce voice set loading time. To install voice sets, you can drop the files into the respective game directory: `xVASynth/resources/app/models/<game>/`


## Instructions

Watch the above video for a summary of this section.

To start, download the latest release (from here: https://github.com/DanRuta/xVA-Synth/releases), double click the xVASynth.exe file, and make sure to click Allow, if Windows asks for permission to run the python server script (this is used internally). Alternatively, check out the `Development` section, to see how to run the non-packaged dev code.


Once the app is up and running, select the voice set category (the game) from the top left dropdown, then click a specific voice set.

A list of already synthesized audio files for that voice set, if any, is displayed. For synthesis, click the `Load model` button. This may take a minute, on a slow machine.

Once finished, type your text in the text area and click the `Generate Voice` button. Once generated, you will be shown a preview of the output. Click the `Keep sample` button to save to file, or click the `Generate Voice` after making ammends to the text input, to discard it and re-generate.

You can adjust the pitch and durations of individual letters by moving the letter sliders up and down, or by using the tools in the toolbars below.

In the list of audio files, you can preview, re-name, click to open the containing folder, or delete each one.

If the required CUDA dependencies are installed on your system, you can enable GPU inference by switching the toggle in the settings menu (click the setting cog at the top right).

Finally, a low-quality but high-speed vocoder model is available through the "Quick-and-dirty" checkbox, to speed up pitch editing. Use the high quality model when generating the final audio.

**Note about synthesis quality**

The best approach I have found is to generate samples of at least 2 seconds in length, and not much more than 5 (short is good for the less good voices). If you need a lot of text to be synthesized, the current best approach is to synthesize smaller clauses, and splicing them together in Audacity. You can vary the punctuation and spelling of words to get different output.

All models have been trained on my personal machine, with a GTX 1080, meaning that `batch_size` had to be limited, to stay within memory constraints. With any luck, I may get access to some beefier machines in the future to train higher quality models.

## Pro tips:

- Right click a voice set in the left bar to hear a sample of the voice.

- You can change the spelling to change pronounciation for difficult words

- You can use full stops and commas to change timing

- Acronyms should be spelled out phonetically. EG: xVA -> Ex vee ay

- Numbers automatically get converted to text. However, if you need to pronounce years, such as 1990 -> nineteen ninety, instead of one thousand nine hundred and ninety, you should split the two numbers, like 19 90.

## Development

`npm install` dependencies, and run with `npm start`. Use virtualenv, and `pip install -r requirements.txt` using Python 3.6.

The app uses both JavaScript (Electron, UI) and Python code (FastPitch Model). As the python script needs to remain running alongside the app, and receive input, communication is done via an HTTP server, with the JavaScript code sending localhost requests, at port 8008. During development, the python source is used. In production, the compiled python is used.

## Packaging

First, run the scripts in `package.json` to create the electron distributables.
Second, use pyinstaller to compile the python. `pip install pyinstaller` and run `pyinstaller -F server.spec`. Discard the `build` folder, and move the `server` folder (in `dist`) into `release-builds/xVASynth-win32-x64/resources/app`, and rename it to `cpython`. Distribute the contents in `xVASynth-win32-x64` and run app through `xVASynth.exe`.

Run the distributed app once and check the `server.log` file for any problems. You may need to copy over asset files from dependencies manually, if they fail to get copied over by pyinstaller. For example, librosa has two files in `<your env>Lib/site-packages/librosa/util/example_data` that need to get copied over to `cpython/librosa/util/example_data/` (you may need to create this second directory structure).

Make sure you remove your environment folder, if it is copied over to the distributed output.

Though, if you're just tweaking small things in JS/HTML/CSS, it may be easier to just edit/copy over the files into an existing packaged distributable. There's no code obfuscation or anything like that in place.

## Models

A large number of models have been trained for this app. They are publicly hosted on the nexus.com website, on the game page respective to the game the voice belongs to.

Some of these voices share the same model, due to having the same voice actor, across games. Some voices were trained together for the majority of iterations, and only fine-tuned at the end, independently.

## Future Plans

Future plans are currently to continue training models for more voices, with plenty more remaining.

### Games

Models are being trained for the following games:

- The Elder Scrolls V: Skyrim
- The Elder Scrolls IV: Oblivion
- The Elder Scrolls III: Morrowind
- Fallout 4
- Fallout New Vegas
- Fallout 3
- Starfield

Time/interest/data permitting, other games/categories may be explored.