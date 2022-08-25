# `pip install gradio` first!

import gradio as gr
import sqlite3 as lite

import argparse
import shlex
import atexit
import os
import sys
import random
from PIL import Image,PngImagePlugin
from numpy import asarray

DEBUG = False

# hack hack la la hack
arg_parser = argparse.ArgumentParser(description='Sable Diffusion Gradio GUI')
arg_parser.add_argument('-bs', '--batch_size', dest='IMAGE_COUNT', type=int, help='how many images to generate, try 1 (or -s) if you''re having VRAM issues', default=3)
arg_parser.add_argument('-s', '--serial', dest='SERIAL', type=bool, help='generate 1-by-1, use for lower VRAM', default=False)
arg_parser.add_argument('-df', '--downsampling_factor', dest='downsampling_factor', type=int, help='BUGGY! for less VRAM usage, lower quality, faster generation, try 9 as a value', default=8)
args = arg_parser.parse_args()
IMAGE_COUNT = args.IMAGE_COUNT # MAX 3!
if IMAGE_COUNT>3:
    IMAGE_COUNT = 3
DOWNSAMPLING = args.downsampling_factor
SERIAL = args.SERIAL
ITERATIONS = 1
SHOW_COUNT = IMAGE_COUNT

if SERIAL and IMAGE_COUNT>1:
    ITERATIONS = IMAGE_COUNT
    IMAGE_COUNT = 1

def change_database(db_filename):
    global database
    if (database):
        database.close()

    if (os.path.isfile(db_filename)):
        try:
            database = lite.connect(db_filename, check_same_thread = False)
            if (database):
                return [True, 'Database changed to: ' + db_filename]
            else:
                return [False, 'Database could not be changed to: ' + db_filename]
        except lite.Error as e:
            return [False, 'Exception throw trying to change database to: ' + db_filename, e.args[0]]

    try:
        database = lite.connect(db_filename, check_same_thread = False)
        if (database):
            tables = [
            "CREATE TABLE image(id integer primary key, filename varchar(64), prompt_id integer, settings_id integer, seed integer);",
            "CREATE VIRTUAL TABLE prompt USING fts5(description);",
            "CREATE TABLE settings(id integer primary key, width integer, height integer, steps integer, cfg_scale real);",
            "CREATE TABLE session(id integer primary key, prompt varchar(2048), seed integer, width integer, height integer, steps integer, cfg_scale real);",
            ]
            cursor = database.cursor()
            for table in tables:
                cursor.execute(table)
            database.commit()
            return [True, 'Database created at: ' + db_filename]
        else:
            return [False, 'Could not create database at: ' + db_filename]
    except lite.Error as e:    
        return [False, "Database error %s: " % e.args[0]]
            
    return [False, 'Unknown error changing database: ' + db_filename]
                
        
        
        
def generate(prompt, seed, steps, width, height, cfg_scale):
    global database
    global outdir
    global t2i

    images = [None, None, None]
    seeds = [0, 0, 0]
   
    if prompt.strip() == "":
        return [None, None, None, 0, 0, 0, "Please enter a valid prompt first"]

    message = "Successfully generated"

    save_session_settings(prompt, seed, steps, cfg_scale, width, height)

    # try find if the image was already generated
    prompt = prompt.lower()
    try:
        # store the hash
        cursor = database.cursor()
        cursor.execute("SELECT rowid FROM prompt WHERE description=?", [prompt])
        result = cursor.fetchone()
        if result is None:
            cursor = database.cursor()
            cursor.execute("INSERT INTO prompt(description) VALUES(?)", [prompt])
            database.commit()
            prompt_id = cursor.lastrowid
            prompt_existed = False
        else:
            prompt_id = result[0]
            prompt_existed = True

        # store the settings here
        cursor = database.cursor()
        cursor.execute("SELECT id FROM settings WHERE width=? AND height=? AND steps=? AND cfg_scale=?", [width, height, steps, cfg_scale])
        result = cursor.fetchone()
        if result is None:
            cursor = database.cursor()
            cursor.execute("INSERT INTO settings(width, height, steps, cfg_scale) VALUES(?, ?, ?, ?)", [width, height, steps, cfg_scale])
            database.commit()
            settings_id = cursor.lastrowid
            settings_existed = False
        else:
            settings_id = result[0]
            settings_existed = True

        # retrieve existing images, or generate new ones here
        image_existed = False
        if settings_existed and prompt_existed:
            cursor = database.cursor()
            cursor.execute("SELECT filename FROM image WHERE prompt_id=? AND settings_id=? AND seed=?", [prompt_id, settings_id, seed])
            rows = cursor.fetchmany(SHOW_COUNT)
            if rows:
                print(f"Prompt already run, returning existing images if possible\n")
                image_existed = True
                for index, row in enumerate(rows):
                    images[index] = asarray(Image.open(row[0]))
                    seeds[index] = seed
        # if we couldn't get 3 images, just regenerate
        if not image_existed:
            results = t2i.txt2img(prompt=prompt, outdir=outdir, seed=seed, steps=steps, width=width, height=height, cfg_scale=cfg_scale)
            for index, result in enumerate(results):
                images[index] = asarray(Image.open(result[0]))
                seeds[index] = result[1]
                cursor = database.cursor()
                cursor.execute("INSERT INTO image(filename, prompt_id, settings_id, seed) values(?, ?, ?, ?)", [result[0], prompt_id, settings_id, result[1]])
                database.commit()
    except lite.Error as e:
        message = 'Database exception: ' + e.args[0]
                
    return [images[0], images[1], images[2], seeds[0], seeds[1], seeds[2], message]

def get_prompts():
    global database
    result = []
    
    try:
        cursor = database.cursor()
        cursor.execute("SELECT description FROM prompt")
        prompts = cursor.fetchall()
        for prompt in prompts:
            result.append(prompt)
    except lite.Error as e:
        result = ['Database exception: ' + e.args[0]]
            
    return result

   
def save_session_settings(prompt, seed, steps, cfg_scale, width, height):
    global database
    try:
        cursor = database.cursor()
        cursor.execute("REPLACE INTO session(id, prompt, seed, steps, cfg_scale, width, height) VALUES(?, ?, ?, ?, ?, ?, ?)", [1, prompt, seed, steps, cfg_scale, width, height])
        database.commit()
    except lite.Error as e:
        print(f"Database exception while saving session\n: {e.args[0]}")
    return session

def get_session_settings():
    global database
    result = { \
        "prompt": "", \
        "seed": 31337, \
        "steps": 50, \
        "cfg_scale": 7.5, \
        "width": 512, \
        "height": 512, \
    }
    try:
        cursor = database.cursor()
        cursor.execute("SELECT prompt, seed, steps, cfg_scale, width, height FROM session WHERE rowid=1")
        session = cursor.fetchone()
        if session and len(session[0].strip()) > 0:
            result["prompt"] = session[0]
            result["seed"] = session[1]
            result["steps"] = session[2]
            result["cfg_scale"] = session[3]
            result["width"] = session[4]
            result["height"] = session[5]
    except lite.Error as e:
        print(f"Database exception while loading session\n: {e.args[0]}")
    return result
    
def get_images_next(p, start):
    global database
    prompt = p[0]
    image1 = None
    image2 = None
    image3 = None
    seed1 = 0
    seed2 = 0
    seed3 = 0
    more = False
    try:
        cursor = database.cursor()
        cursor.execute("SELECT rowid FROM prompt WHERE description=?", [prompt])
        prompts_id = cursor.fetchone()[0]
        cursor = database.cursor()
        cursor.execute("SELECT filename, seed, id FROM image WHERE prompt_id=? AND id>? ORDER BY id LIMIT 4", [prompts_id, start])
        images = cursor.fetchall()
        count = len(images)
        more = count > 3
        if count > 0:
            image1 = asarray(Image.open(images[0][0]))
            seed1 = images[0][1]
            start = images[0][2]                        
        if count > 1:
            image2 = asarray(Image.open(images[1][0]))
            seed2 = images[1][1]
            start = images[1][2]                        
        if count > 2:
            image3 = asarray(Image.open(images[2][0]))
            seed3 = images[2][1]
            start = images[2][2]                        
        if not more:
            start = 0
    except lite.Error as e:
        print(f"EXCEPTION: {e}")
    if more:
        button_text = "View more images"
    else:
        button_text = "Show first images"
    return start, \
        gr.Button.update(value=button_text), \
        gr.update(value=image1, visible=True), gr.update(value=image2, visible=True), gr.update(value=image3, visible=True), \
        gr.update(value=seed1, visible=True), gr.update(value=seed2, visible=True), gr.update(value=seed3, visible=True)

last_filter_mode='none'
last_filter_text=''        
def get_images_history(start, amount, filter_mode, filter_text):
    global database
    global last_filter_mode
    global last_filter_text
    result = []    
   
    # I give up being smart at this point, globals it is. Reset to the start of search results if parameters have changed
    if (not last_filter_mode == filter_mode) or (not last_filter_text == filter_text):
        last_filter_mode = filter_mode
        last_filter_text = filter_text
        start = 0
   
    more = False
    try:
        cursor = database.cursor()
        if (not filter_mode) or (filter_mode == 'none') or (filter_text.strip() == ""):
            cursor.execute("SELECT filename, seed, id FROM image WHERE id>? ORDER BY id LIMIT ?", [start, amount+1])
        else:   
            cursor.execute('SELECT filename, seed, id FROM image WHERE id>? AND prompt_id IN (SELECT rowid FROM prompt WHERE description LIKE ''?'') ORDER BY id LIMIT ?', [start, '%'+filter_text+'%', amount+1])
        images = cursor.fetchall()
        count = len(images)
        more = count > amount
        if more:
            images.pop() # don't use the last one, it was for knowing if there's more
        for image in images:
            result.append({ 'filename': image[0], 'seed': image[1], 'id': image[2] })
            start = image[2]
        if not more:
            start = 0
    except lite.Error as e:
        print(f"Database exception trying to get image history: {e}")
    return result, start, more
    
    
# globals here we go!    
width   = 512
height  = 512
config  = "configs/stable-diffusion/v1-inference.yaml"
weights = "models/ldm/stable-diffusion-v1/model.ckpt"
outdir = "outputs/sdgg"

# make sure the output directory exists
if not os.path.exists(outdir):
    os.makedirs(outdir)

database = None
result, message = change_database(outdir + "/sdgg.db")
if not result:
    print(f"Could not create database! Error: {message}\n")
    quit()

# debug for trying to make the dropdown from refreshing infinitely, didn't work
last_prompt = ""    
    
    
with gr.Blocks(title="Stable Diffusion GUI") as demo:
    # try load a previous session, if any
    session = get_session_settings()

    with gr.Tabs():
        # Text to image tab
        with gr.TabItem("Text to Image"):
            tti_prompt = gr.Textbox(label="Prompt        (the sentence to generate the image from)", max_lines=1, value=session["prompt"])
            with gr.Row():
                tti_seed = gr.Number(label="Seed        (use this number to regenerate the same image/style in future)", value=session["seed"], precision=0)
                tti_random = gr.Button(value="Randomize seed")
                tti_random.click(fn=lambda: gr.update(value=random.randint(0, 4294967295)), inputs=None, outputs=tti_seed)
            tti_steps = gr.Slider(label="Steps        (how much it tries to refine the output)", minimum=0, maximum=200, value=session["steps"], step=1)
            tti_cfg_scale = gr.Slider(label="Config scale        (how hard it tries to fit the image to the description, it can try TOO hard)", minimum=0, maximum=30, value=session["cfg_scale"], step=0.1)
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            tti_width = gr.Number(label="Width (multiple of 64)", value=session["width"], precision=0)
                        with gr.Column():
                            tti_height = gr.Number(label="Height (multiple of 64)", value=session["height"], precision=0)
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            gr.Button(value="Portrait").click(fn=lambda: [448, 640], inputs=None, outputs=[tti_width, tti_height])
                        with gr.Column():
                            gr.Button(value="Landscape").click(fn=lambda: [640, 448], inputs=None, outputs=[tti_width, tti_height])
                        with gr.Column():
                            gr.Button(value="Square").click(fn=lambda: [512, 512], inputs=None, outputs=[tti_width, tti_height])
            generate_btn = gr.Button("Generate", variant='primary')
            with gr.Row():
                with gr.Column():
                    tti_output1 = gr.Image(label="Generated image")
                    tti_seed1 = gr.Textbox(label="Seed", max_lines=1, interactive=False)
                with gr.Column():
                    tti_output2 = gr.Image(label="Generated image", visible=SHOW_COUNT>1)
                    tti_seed2 = gr.Textbox(label="Seed", max_lines=1, interactive=False, visible=SHOW_COUNT>1)
                with gr.Column():
                    tti_output3 = gr.Image(label="Generated image", visible=SHOW_COUNT>2)
                    tti_seed3 = gr.Textbox(label="Seed", max_lines=1, interactive=False, visible=SHOW_COUNT>2)
            message = gr.Textbox(label="Messages", max_lines=1, interactive=False)
            generate_btn.click(fn=generate, inputs=[tti_prompt, tti_seed, tti_steps, tti_width, tti_height, tti_cfg_scale], outputs=[tti_output1, tti_output2, tti_output3, tti_seed1, tti_seed2, tti_seed3, message])
            
        # Image history tab
        with gr.TabItem("Image history"):
            # callback functions
            def update_image_history(start_, filter_mode_, filter_text_):
                result_ = []
                history_, next_start_, more_ = get_images_history(start_, 25, filter_mode_, filter_text_)
                result_.append(next_start_)
                if more_:
                    result_.append(gr.update(value='More')) # 2 of them 'cos we gots 2 buttons
                else:
                    result_.append(gr.update(value='Restart from beginning'))
                count_ = len(history_)
                for index_ in range(25):
                    if index_ < count_:
                        result_.append(gr.update(visible=True, value=history_[index_]['filename'], label=history_[index_]['seed']))
                        # y'know, I tried everything to try get the filename from here into use_image_history() below, but making a variable
                        # during the initial image grid creation caused it to not be updated, and passing images as an input gets you a numpy array, sooo...
                        result_.append(gr.update(visible=True, value="Reuse settings ☝️ ["+str(history_[index_]['id'])+"]")) # for the reuse settings button
                    else:
                        result_.append(gr.update(value=None, visible=False))
                        result_.append(gr.update(visible=False)) # for the reuse settings button
                if more_:
                    result_.append(gr.update(value='More')) # 2 of them 'cos we gots 2 buttons
                else:
                    result_.append(gr.update(value='Restart from beginning'))
                return result_

            def use_image_history(filename_):
                global database
                result_ = [None, None, None, None, None, None] # seed, steps, width, height, cfg_scale, prompt
                id_ = filename_[filename_.find('[')+1:-1]
                try:
                    cursor_ = database.cursor()
                    cursor_.execute("SELECT image.seed, settings.steps, settings.width, settings.height, settings.cfg_scale, prompt.description \
                    FROM image \
                    INNER JOIN settings ON settings.id=image.settings_id \
                    INNER JOIN prompt ON prompt.ROWID=image.prompt_id \
                    WHERE image.id=?" , [id_])
                    row_ = cursor_.fetchone()
                    if row_:
                        result_ = [row_[0], row_[1], row_[2], row_[3], row_[4], row_[5]]
                except lite.Error as e:
                    print(f"Database exception trying to get image history: {e}")
                return result_
                
            # UI controls    
            ima_start = gr.Variable(value=0)
            with gr.Row():
                with gr.Column():
                    ima_filter_mode = gr.Dropdown(label="Filter mode", choices=["none", "contains"], value="none")
                with gr.Column():
                    ima_filter_text = gr.Textbox(label="Filter text", visible=False)
                ima_filter_mode.change(fn=lambda mode: gr.update(visible=not mode=='none'), inputs=ima_filter_mode, outputs=ima_filter_text)
            ima_fetch1 = gr.Button(value='Fetch', variant='primary')
            ima_outputs = [ima_start, ima_fetch1]
            for y in range(5):
                with gr.Row():
                    for x in range(5):
                        with gr.Column():
                            ima_image = gr.Image(interactive=False)
                            ima_outputs.append(ima_image)
                            ima_butt = gr.Button(value='Reuse settings ☝️', variant='secondary', visible=False)
                            ima_butt.click(fn=use_image_history, inputs=ima_butt, outputs=[tti_seed, tti_steps, tti_width, tti_height, tti_cfg_scale, tti_prompt])
                            ima_outputs.append(ima_butt)
            ima_fetch2 = gr.Button(value='Fetch', variant='primary')
            ima_outputs.append(ima_fetch2)
            ima_fetch1.click(fn=update_image_history, inputs=[ima_start, ima_filter_mode, ima_filter_text], outputs=ima_outputs)
            ima_fetch2.click(fn=update_image_history, inputs=[ima_start, ima_filter_mode, ima_filter_text], outputs=ima_outputs)
   
        # Prompt History tab    
        with gr.TabItem("Prompt history"):
            with gr.Row():
                prompt_choice = gr.Dropdown(label="Prompts", choices=get_prompts())
                # doesn't work for some reason, I really think dropdowns are broken at the moment
                #refresh_btn = gr.Button(value="Refresh")
                #refresh_btn.click(fn=get_prompts, inputs=None, outputs=prompt_choice)
            with gr.Row():
                clipboard_btn = gr.Button(value="Copy to clipboard")
                clipboard_btn.click(fn=None, inputs=[prompt_choice], outputs=None, _js="(p) => navigator.clipboard.writeText(p)")
                use_btn = gr.Button(value="Use")
                use_btn.click(fn=lambda p: gr.update(value=p[0]), inputs=prompt_choice, outputs=tti_prompt)
            with gr.Row():
                with gr.Column():
                    saved_output1 = gr.Image(label="Generated image", visible=False)
                    saved_seed1 = gr.Textbox(label="Seed", max_lines=1, interactive=False, visible=False)
                with gr.Column():
                    saved_output2 = gr.Image(label="Generated image", visible=False)
                    saved_seed2 = gr.Textbox(label="Seed", max_lines=1, interactive=False, visible=False)
                with gr.Column():
                    saved_output3 = gr.Image(label="Generated image", visible=False)
                    saved_seed3 = gr.Textbox(label="Seed", max_lines=1, interactive=False, visible=False)

            window_start = gr.Variable(value=0)    
            next_btn = gr.Button(value="View images", visible=True)
            next_btn.click(fn=get_images_next, inputs=[prompt_choice, window_start], outputs=[window_start, next_btn, saved_output1, saved_output2, saved_output3, saved_seed1, saved_seed2, saved_seed3])

            # I _think_ there's a Gradio bug with dropdowns being infinitely refreshed upon change. There's a similar bug already filed, so wait and see.
            # For just rely on a button to make things happen, ugh
            #def prompt_change(p):
            #    global last_prompt
            #    if last_prompt == p[0]:
            #        print(f"{last_prompt} == {p[0]}\n")
            #    last_prompt = p[0]
            #    print("updating\n")
            #    return [gr.update(value=0), gr.update(visible=True, value=p[0])]
            #prompt_choice.change(fn=prompt_change, inputs=[prompt_choice], outputs=[window_start, prompt_text])
            #prompt_text.change(fn=get_images_next, inputs=[prompt_choice, window_start], outputs=[window_start, next_btn, saved_output1, saved_output2, saved_output3, saved_seed1, saved_seed2, saved_seed3])
   
        # Settings tab
        with gr.TabItem("Settings"):
            # TODO: refactor this _hard_
            def apply_settings(count_, downsampling_, serial_):
                global IMAGE_COUNT
                global outdir
                global width
                global height
                global weights
                global config
                global t2i
                global SERIAL
                global ITERATIONS
                SERIAL = (serial_ == "on")
                IMAGE_COUNT = count_
                DOWNSAMPLING = downsampling_
                if SERIAL and IMAGE_COUNT>1:
                    ITERATIONS = IMAGE_COUNT
                    IMAGE_COUNT = 1
                
                t2i = T2I(width=width,
                          height=height,
                          batch_size=IMAGE_COUNT,
                          iterations=ITERATIONS,
                          outdir=outdir,
                          sampler_name="klms", # or plms?
                          weights=weights,
                          full_precision=True,
                          config=config,
                          downsampling_factor=downsampling_,
                )
                # reload the model
                t2i.load_model()
                return [gr.update(visible=count_>1), gr.update(visible=count_>2), gr.update(visible=count_>1), gr.update(visible=count_>2), gr.update(value='Done')]
                
            gr.Label(value="NOTE: these are not saved right now, check out the parameters of this Python script using --help for a more permanent solution")
            set_count = gr.Dropdown(label="Images to generate  (try generating only 1 if you're struggling with memory issues and want to retain quality)", choices=[1,2,3], value=3)
            set_serial = gr.Dropdown(label="Serial generation  (try turning this on if you're having memory issues)", choices=["off", "on"], value="off")
            set_downsampling = gr.Slider(label="Downsampling factor        (BUGGY! increasing this reduces quality, but lowers VRAM usage. if you're struggling, try setting this to 9)", minimum=1, maximum=20, value=8, step=1)
            with gr.Row():
                with gr.Column():
                    set_apply = gr.Button(value='Apply', variant='primary')
                with gr.Column():
                    set_status = gr.Label(label='Status')
            set_apply.click(fn=apply_settings, inputs=[set_count, set_downsampling, set_serial], outputs=[tti_output2, tti_output3, tti_seed2, tti_seed3, set_status])
   
   
sys.path.append('.')
from ldm.simplet2i import T2I

# TODO:
# * make this unload after e.g., 5 minutes of inactivity, then reload when used again
# * make a settings tab that recreates this, so you can change the batch size if memory is low
if not DEBUG:
    t2i = T2I(width=width,
              height=height,
              batch_size=IMAGE_COUNT,
              iterations=ITERATIONS,
              outdir=outdir,
              sampler_name="klms", # or plms?
              weights=weights,
              full_precision=True,
              config=config,
              downsampling_factor=DOWNSAMPLING,
    )
    # preload the model
    t2i.load_model()

# go!    
demo.launch()
