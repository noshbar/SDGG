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
2. Clone this repo and copy the files to the same folder you just cloned the above to
3. Run `pip install gradio`
4. Run `pip install -e git+https://github.com/lstein/k-diffusion.git@master#egg=k-diffusion`

Running:
1. Run `python scripts\sdgg.py`
    * optionally pass `-p true` to generate images in parallel (slight time saving)
        * AND optionally pass `-bs 1` to only generate 1/2/3 image(s) at the same time, if you're having memory issues with parallel
2. Wait for it to load and show a message like `Running on local URL:  http://127.0.0.1:7860/`
3. Browse to that address in your browser and start generating!


### Side note

I initially cloned [this fork](https://github.com/lstein/stable-diffusion) repo because it had `dream.py` in it, an interactive console application that I thought was neat.  
However, it has progressed quickly itself, too, meaning updating that fork breaks this repo.  

It now has a bunch of nice features, like being able to post-process faces to add more detail, and even has a web interface of its own.

So only `sdgg.py` was done by me, I got the other scripts from that fork. It was only a stopgap anyway though, and I'd like to do away with any dependencies other than the official Stable Diffusion repo... eventually.

### Features

* text to image
* image to image
* simple GUI to adjust most common generation settings
* saves all the prompts you use to a database so you can reuse them in future (searching by keyword coming soon, maybe)
* saves your session details so if the server restarts and you refresh your page, you'll get where you last left off
* tries to prevent generating the same images again, if they're found, they'll be returned immediately from the outputs folder
* provides rudimentary browsing of generated images, allowing you to copy settings used for generating

### GOTCHAs

* clicking "use" on the prompt history tab or "use settings" on the image history tab WILL copy the settings to the "text to image" tab, but you'll have to swap tabs manually
* the prompt history will not contain new prompts until the server is restarted (you can however use the image history)
* when using parallel generation, the seeds reported on the second 2 images is wrong, I can't figure out how to get the internal seeds used

### TODO

Oh so SO much
* add proper searching through prompts (matches, contains, is like, etc.)
* wait for the dropdown bug to be fixed so new prompts can be added to the prompt history without restarting the server first
* refactor out into different classes, get rid of globals, etc.
* refactor to use offical repo without changes
* file/fix Gradio bugs so that things like using a dropdown box work to cause actions to happen, to get rid of some buttons
* figure out how to make buttons occur before the elements they act on
* figure out how to make the images in the gallery clickable (not possible in Gradio at the moment, and _js fails to work)
