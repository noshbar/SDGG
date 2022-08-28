# SDGG - Stable Diffusion Gradio GUI

This is a web-based user interface for generating images using the Stable Diffusion repository.

**NOTE**:
* This was done quickly to solve a problem, not to win code quality awards. You can file issues if you feel like it, though :shrug:
* This produces uncensored results, be aware.

(for my own sake: I've barely spent any time trying to understand the whole ML framework and Gradio, and I hacked this together as quickly as I could so that I could put it up before I start my new job)

![GUI Preview](preview.jpg)

### Quickstart

Setup:
1. Clone [the official Stable Diffusion repo](https://github.com/CompVis/stable-diffusion) and create and activate the environment as instructed in their docs
    * developed against commit 69ae4b35e0a0f6ee1af8bb9a5d0016ccb27e36dc, in case things break in the future
2. Clone this repo and copy the files to the same folder you just cloned the above to
3. Run `pip install gradio`
    * developed against version 3.1.7
4. Run `pip install -e git+https://github.com/lstein/k-diffusion.git@master#egg=k-diffusion`

Running:
1. Run `python scripts\sdgg.py`
2. Wait for it to load and show a message like `Running on local URL:  http://127.0.0.1:7860/`
3. Browse to that address in your browser and start generating!

Main user options:
* '--help' to show all options
* `-of <category>` lets you choose the subfolder within `outputs`, grouping them by category, e.g., one for "animals", one for "fantasy", one for "taxes", whatever, to prevent things getting too unwieldly
* '-ip true' will show images as they're ready, not wait until all 3 are done. *CAUTION*: this is a complete hack, abusing Gradio, so use with discretion

### Longstart

This is how I got up and running:
1. Get the weights from https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
    * requires creating a HuggingFace account
    * requires checking their agreement checkbox
1. Install Anaconda
1. Clone [the official Stable Diffusion repo](https://github.com/CompVis/stable-diffusion)
1. `cd stable-diffusion`
1. Copy the weights to `models/ldm/stable-diffusion-v1/model.ckpt` (creating the folder if necessary)
1. `conda env create -f environment.yaml`
1. `conda activate ldm`
1. Test if it's all working with `python scripts/txt2img.py --prompt "a photograph of an astronaut riding a horse" --plms` (it will download additional stuff)
1. Clone this repo
1. Copy all the files to the `stable-diffusion` folder
1. `pip install gradio`
1. `pip install -e git+https://github.com/lstein/k-diffusion.git@master#egg=k-diffusion`
1. Run `python scripts\sdgg.py`
1. Wait for the URL to browse to, and start using it!

### Features

* text to image
* image to image
* simple GUI to adjust most common generation settings
* saves all the prompts you use to a database so you can reuse them in future (searching by keyword coming soon, maybe)
* saves your session details so if the server restarts and you refresh your page, you'll get where you last left off
* tries to prevent generating the same images again, if they're found, they'll be returned immediately from the outputs folder
* provides rudimentary browsing of generated images, allowing you to copy settings used for generating
* weighted prompts
    * the example from the [lstein fork](https://github.com/lstein/stable-diffusion) is: `tabby cat:0.25 white duck:0.75 hybrid`, "This will tell the sampler to invest 25% of its effort on the tabby cat aspect of the image and 75% on the white duck aspect"

### GOTCHAs

* clicking "use" on the prompt history tab or "use settings" on the image history tab WILL copy the settings to the "text to image" tab, but you'll have to swap tabs manually
* the prompt history will not contain new prompts until the server is restarted (you can however use the image history)
* when using parallel generation, the seeds reported on the second 2 images is wrong, I can't figure out how to get the internal seeds used

### Side note

I initially cloned [this fork](https://github.com/lstein/stable-diffusion) repo because it had `dream.py` in it, an interactive console application that I thought was neat.  
However, it has progressed quickly itself, too, meaning updating that fork breaks this repo.  

It now has a bunch of nice features, like being able to post-process faces to add more detail, and even has a web interface of its own.

So only `sdgg.py` was done by me, I got the other scripts from that fork. It was only a stopgap anyway though, and I'd like to do away with any dependencies other than the official Stable Diffusion repo... eventually.

### TODO

Oh so SO much
* add proper searching through prompts (matches, contains, is like, etc.)
* wait for the dropdown bug to be fixed so new prompts can be added to the prompt history without restarting the server first
* refactor out into different classes, get rid of globals, etc.
* refactor to use offical repo without changes
* file/fix Gradio bugs so that things like using a dropdown box work to cause actions to happen, to get rid of some buttons
* figure out how to make buttons occur before the elements they act on
* figure out how to make the images in the gallery clickable (not possible in Gradio at the moment, and _js fails to work)
