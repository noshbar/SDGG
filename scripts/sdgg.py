# `pip install gradio` first!
# tested against version 3.1.7

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
sys.path.append('.')
from ldm_old.simplet2i import T2I

DEBUG = False

# hack hack la la hack
arg_parser = argparse.ArgumentParser(description='Stable Diffusion Gradio GUI')
arg_parser.add_argument('-df', '--downsampling_factor', dest='downsampling_factor', type=int, help='BUGGY! for less VRAM usage, lower quality, faster generation, try 9 as a value', default=8)
arg_parser.add_argument('-s', '--sampler', dest='sampler', type=str, help='which sampler to use (klms/plms)', default='klms')
arg_parser.add_argument('-p', '--parallel', dest='PARALLEL', type=bool, help='generate the entire batch at once, slightly quicker, uses more VRAM', default=False)
arg_parser.add_argument('-bs', '--batch_size', dest='IMAGE_COUNT', type=int, help='with -p, how many images to generate if you''re having VRAM issues (1..3)', default=3)
arg_parser.add_argument('-of', '--output_folder', dest='OUTPUT_FOLDER', type=str, help='the sub-folder within ./outputs to store generated images and the prompt database', default='sdgg')
args = arg_parser.parse_args()

# globals here we go!    
OUTDIR   = "outputs/" + args.OUTPUT_FOLDER
CONFIG   = "configs/stable-diffusion/v1-inference.yaml"
WEIGHTS  = "models/ldm/stable-diffusion-v1/model.ckpt"

SAMPLER = args.sampler
IMAGE_COUNT = args.IMAGE_COUNT # MAX 3!
if IMAGE_COUNT>3:
    IMAGE_COUNT = 3
DOWNSAMPLING = args.downsampling_factor
SHOW_COUNT = IMAGE_COUNT

if args.PARALLEL:
    ITERATIONS = 1
else:
    ITERATIONS = IMAGE_COUNT
    IMAGE_COUNT = 1
    
DATABASE = None

   
    
def change_database(db_filename):
    global DATABASE
    if (DATABASE):
        DATABASE.close()

    if (os.path.isfile(db_filename)):
        try:
            DATABASE = lite.connect(db_filename, check_same_thread = False)
            if (DATABASE):
                return [True, 'Database changed to: ' + db_filename]
            else:
                return [False, 'Database could not be changed to: ' + db_filename]
        except lite.Error as e:
            return [False, 'Exception throw trying to change database to: ' + db_filename, e.args[0]]

    try:
        DATABASE = lite.connect(db_filename, check_same_thread = False)
        if (DATABASE):
            tables = [
            "CREATE TABLE image(id integer primary key, filename varchar(64), prompt_id integer, settings_id integer, seed integer);",
            "CREATE VIRTUAL TABLE prompt USING fts5(description);",
            "CREATE TABLE settings(id integer primary key, width integer, height integer, steps integer, cfg_scale real);",
            "CREATE TABLE session(id integer primary key, prompt varchar(2048), seed integer, width integer, height integer, steps integer, cfg_scale real);",
            ]
            cursor = DATABASE.cursor()
            for table in tables:
                cursor.execute(table)
            DATABASE.commit()
            return [True, 'Database created at: ' + db_filename]
        else:
            return [False, 'Could not create database at: ' + db_filename]
    except lite.Error as e:    
        return [False, "Database error %s: " % e.args[0]]
            
    return [False, 'Unknown error changing database: ' + db_filename]
                
        
        
        
def generate(init_image_filename, prompt, seed, steps, width, height, cfg_scale):
    global DATABASE
    global OUTDIR
    global GENERATOR

    images = [None, None, None]
    seeds = [0, 0, 0]
    ids = [0, 0, 0]
   
    if prompt.strip() == "":
        return [None, None, None, 0, 0, 0, "Please enter a valid prompt first"]

    message = "Successfully generated"

    if not init_image_filename:
        save_t2i_session_settings(prompt, seed, steps, cfg_scale, width, height)

    # try find if the image was already generated
    prompt = prompt.lower()
    try:
        # store the hash
        cursor = DATABASE.cursor()
        cursor.execute("SELECT rowid FROM prompt WHERE description=?", [prompt])
        result = cursor.fetchone()
        if result is None:
            cursor = DATABASE.cursor()
            cursor.execute("INSERT INTO prompt(description) VALUES(?)", [prompt])
            DATABASE.commit()
            prompt_id = cursor.lastrowid
            prompt_existed = False
        else:
            prompt_id = result[0]
            prompt_existed = True

        # store the settings here
        cursor = DATABASE.cursor()
        cursor.execute("SELECT id FROM settings WHERE width=? AND height=? AND steps=? AND cfg_scale=?", [width, height, steps, cfg_scale])
        result = cursor.fetchone()
        if result is None:
            cursor = DATABASE.cursor()
            cursor.execute("INSERT INTO settings(width, height, steps, cfg_scale) VALUES(?, ?, ?, ?)", [width, height, steps, cfg_scale])
            cursor.execute("SELECT last_insert_rowid();")
            row = cursor.fetchone();
            DATABASE.commit()
            settings_id = row[0]
            settings_existed = False
        else:
            settings_id = result[0]
            settings_existed = True

        # retrieve existing images, or generate new ones here. if doing image to image, no way to know if the source changed right now, so always regenerate
        image_existed = False
        if (init_image_filename is None) and settings_existed and prompt_existed:
            cursor = DATABASE.cursor()
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
            if init_image_filename:
                results = GENERATOR.img2img(prompt=prompt, outdir=OUTDIR, seed=seed, steps=steps, width=width, height=height, cfg_scale=cfg_scale, init_img=init_image_filename)
            else:
                results = GENERATOR.txt2img(prompt=prompt, outdir=OUTDIR, seed=seed, steps=steps, width=width, height=height, cfg_scale=cfg_scale)
            for index, result in enumerate(results):
                images[index] = asarray(Image.open(result[0]))
                seeds[index] = result[1]
                cursor = DATABASE.cursor()
                cursor.execute("INSERT INTO image(filename, prompt_id, settings_id, seed) values(?, ?, ?, ?)", [result[0], prompt_id, settings_id, result[1]])
                cursor.execute("SELECT last_insert_rowid();")
                row = cursor.fetchone();
                ids[index] = row[0]
                DATABASE.commit()
    except lite.Error as e:
        message = 'Database exception: ' + e.args[0]
                
    results = [images[0], images[1], images[2], seeds[0], seeds[1], seeds[2], message, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)]
    for index in range(3):
        visible = not images[index] is None
        value = "Delete forever ["+str(ids[index])+"]"
        results.append(gr.update(visible=visible, value=value))
    return results

    
def get_prompts():
    global DATABASE
    result = []
    
    try:
        cursor = DATABASE.cursor()
        cursor.execute("SELECT description FROM prompt")
        prompts = cursor.fetchall()
        for prompt in prompts:
            result.append(prompt)
    except lite.Error as e:
        result = ['Database exception: ' + e.args[0]]
            
    return result

   
def save_t2i_session_settings(prompt, seed, steps, cfg_scale, width, height):
    global DATABASE
    try:
        cursor = DATABASE.cursor()
        cursor.execute("REPLACE INTO session(id, prompt, seed, steps, cfg_scale, width, height) VALUES(?, ?, ?, ?, ?, ?, ?)", [1, prompt, seed, steps, cfg_scale, width, height])
        DATABASE.commit()
    except lite.Error as e:
        print(f"Database exception while saving session\n: {e.args[0]}")
    result = { \
        "prompt": prompt, \
        "seed": seed, \
        "steps": steps, \
        "cfg_scale": cfg_scale, \
        "width": width, \
        "height": height, \
    }
    return result

    
def get_t2i_session_settings():
    global DATABASE
    result = { \
        "prompt": "", \
        "seed": 31337, \
        "steps": 50, \
        "cfg_scale": 7.5, \
        "width": 512, \
        "height": 512, \
    }
    try:
        cursor = DATABASE.cursor()
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

    
def get_i2i_session_settings():
    result = { \
        "prompt": "", \
        "seed": 31337, \
        "steps": 50, \
        "cfg_scale": 7.5, \
        "width": 512, \
        "height": 512, \
    }
    # no DB settings for it yet
    return result
    
    
def get_images_next(p, start):
    global DATABASE
    prompt = p[0]
    image1 = None
    image2 = None
    image3 = None
    seed1 = 0
    seed2 = 0
    seed3 = 0
    more = False
    try:
        cursor = DATABASE.cursor()
        cursor.execute("SELECT rowid FROM prompt WHERE description=?", [prompt])
        prompts_id = cursor.fetchone()[0]
        cursor = DATABASE.cursor()
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
    global DATABASE
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
        cursor = DATABASE.cursor()
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


def make_red_cross_image():
    not_exist = Image.new('RGB', (64, 64))
    for y in range(32):
        for x in range(32):
            if x == y:
                colour = (255, 0, 0)
            else:
                colour = (255, 255, 255)
            not_exist.putpixel((x, y), colour)
            not_exist.putpixel((63-x, y), colour)
            not_exist.putpixel((x, 63-y), colour)
            not_exist.putpixel((63-x, 63-y), colour)
    return asarray(not_exist)

def remove_image_forever(text_id):
    global DATABASE
    id = text_id[text_id.find('[')+1:text_id.rfind(']')]
    try:
        cursor = DATABASE.cursor()
        cursor.execute("SELECT filename FROM image WHERE id=?", [id])
        row = cursor.fetchone()
        if row:
            try:
                if os.path.exists(row[0]):
                    os.remove(row[0])
                cursor = DATABASE.cursor()
                cursor.execute("DELETE FROM image WHERE image.id=?", [id])
                DATABASE.commit()
            except OSError as error:
                print(f"Could not delete image {error}\n")
    except lite.Error as e:
        print(f"Database exception trying get image to delete: {e}")
    return [gr.update(value=make_red_cross_image()), gr.update(visible=False), gr.update(visible=False)]            
      
    
def generate_t2i(prompt, seed, steps, width, height, cfg_scale):
    return generate(None, prompt, seed, steps, width, height, cfg_scale)

def generate_i2i(image_numpy, prompt, seed, steps, cfg_scale):
    global OUTDIR
    image = Image.fromarray(image_numpy)    
    filename = OUTDIR + "/temp.png"
    image.save(filename)
    width, height = image.size
    steps = int(steps * 1.34) # no idea why, but the image to image script seems to use 74% of what you provide
    
    return generate(filename, prompt, seed, steps, width, height, cfg_scale)
    
def create_generation_tab(title, with_image_input, session):
    with gr.TabItem(title):
        if with_image_input:
            image = gr.Image(label="Guide/initial image (keep small with dimensions that are multiples of 64)", interactive=True)
        else:
            image = None
            
        local_prompt = gr.Textbox(label="Prompt        (the sentence to generate the image from)", max_lines=1, value=session["prompt"])
        with gr.Row():
            local_seed = gr.Number(label="Seed        (use this number to regenerate the same image/style in future)", value=session["seed"], precision=0)
            local_random = gr.Button(value="Randomize seed")
            local_random.click(fn=lambda: gr.update(value=random.randint(0, 4294967295)), inputs=None, outputs=local_seed)
        local_steps = gr.Slider(label="Steps        (how much it tries to refine the output)", minimum=0, maximum=200, value=session["steps"], step=1)
        local_cfg_scale = gr.Slider(label="Config scale        (how hard it tries to fit the image to the description, it can try TOO hard)", minimum=0, maximum=30, value=session["cfg_scale"], step=0.1)
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        local_width = gr.Number(label="Width (must be multiple of 64)", value=session["width"], precision=0, interactive=not with_image_input)
                    with gr.Column():
                        local_height = gr.Number(label="Height (must be multiple of 64)", value=session["height"], precision=0, interactive=not with_image_input)
            if not with_image_input:
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            gr.Button(value="Portrait").click(fn=lambda: [448, 640], inputs=None, outputs=[local_width, local_height])
                        with gr.Column():
                            gr.Button(value="Landscape").click(fn=lambda: [640, 448], inputs=None, outputs=[local_width, local_height])
                        with gr.Column():
                            gr.Button(value="Square").click(fn=lambda: [512, 512], inputs=None, outputs=[local_width, local_height])
        generate_btn = gr.Button("Generate", variant='primary')
        with gr.Row():
            with gr.Column():
                local_output1 = gr.Image(label="Generated image", interactive=False)
                local_seed1 = gr.Textbox(label="Seed", max_lines=1, interactive=False)
                use_btn1 = gr.Button(value="Use for image-to-image generation", visible=False)
                remove_btn1 = gr.Button(value="Remove forever", visible=False)
                remove_btn1.click(fn=remove_image_forever, inputs=remove_btn1, outputs=[local_output1, use_btn1, remove_btn1])
            with gr.Column():
                local_output2 = gr.Image(label="Generated image", visible=SHOW_COUNT>1, interactive=False)
                local_seed2 = gr.Textbox(label="Seed", max_lines=1, interactive=False, visible=True)
                use_btn2 = gr.Button(value="Use for image-to-image generation", visible=False)
                remove_btn2 = gr.Button(value="Remove forever", visible=False)
                remove_btn2.click(fn=remove_image_forever, inputs=remove_btn2, outputs=[local_output2, use_btn2, remove_btn2])
            with gr.Column():
                local_output3 = gr.Image(label="Generated image", visible=SHOW_COUNT>2, interactive=False)
                local_seed3 = gr.Textbox(label="Seed", max_lines=1, interactive=False, visible=True)
                use_btn3 = gr.Button(value="Use for image-to-image generation", visible=False)
                remove_btn3 = gr.Button(value="Remove forever", visible=False)
                remove_btn3.click(fn=remove_image_forever, inputs=remove_btn3, outputs=[local_output3, use_btn3, remove_btn3])
        message = gr.Textbox(label="Messages", max_lines=1, interactive=False)
        outputs_ = [local_output1, local_output2, local_output3, local_seed1, local_seed2, local_seed3, message]
        outputs_ += [use_btn1, use_btn2, use_btn3, remove_btn1, remove_btn2, remove_btn3]
        if with_image_input:
            generate_btn.click(fn=generate_i2i, inputs=[image, local_prompt, local_seed, local_steps, local_cfg_scale], outputs=outputs_)
        else:
            generate_btn.click(fn=generate_t2i, inputs=[local_prompt, local_seed, local_steps, local_width, local_height, local_cfg_scale], outputs=outputs_)
        return local_prompt, local_seed, local_steps, local_width, local_height, local_cfg_scale, local_output1, local_seed1, local_output2, local_seed2, local_output3, local_seed3, image, [use_btn1, use_btn2, use_btn3]

        
def create_prompt_history_tab(tti_prompt, iti_prompt):    
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
            use_btn.click(fn=lambda p: [gr.update(value=p[0]), gr.update(value=p[0])], inputs=prompt_choice, outputs=[tti_prompt, iti_prompt])
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
   
    
def create_image_history_tab(outputs):
    with gr.TabItem("Image history"):
        # callback functions
        def update_image_history(start_, filter_mode_, filter_text_, offset_):
            global DATABASE
            if 'oldest' in offset_:
                start_ = 0
            if 'most recent' in offset_:
                cursor_ = DATABASE.cursor()
                cursor_.execute("SELECT id FROM image ORDER BY id DESC LIMIT 25")
                row_ = cursor_.fetchall()
                start_ = row_[-1][0]
                
                
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
                    exists_ = os.path.exists(history_[index_]['filename'])
                    if exists_:
                        path_ = history_[index_]['filename']
                    else:
                        path_ = make_red_cross_image()
                    result_.append(gr.update(visible=True, value=path_, label=history_[index_]['seed']))
                    # y'know, I tried everything to try get the filename from here into use_image_history() below, but making a variable
                    # during the initial image grid creation caused it to not be updated, and passing images as an input gets you a numpy array, sooo...
                    result_.append(gr.update(visible=exists_, value="Reuse settings ☝️ ["+str(history_[index_]['id'])+"]")) # for the reuse settings button
                    result_.append(gr.update(visible=True, value="Delete forever ☝️ ["+str(history_[index_]['id'])+"]")) # for the reuse settings button
                else:
                    result_.append(gr.update(value=None, visible=False))
                    result_.append(gr.update(visible=False)) # for the reuse settings button
                    result_.append(gr.update(visible=False)) # for the reuse settings button
            if more_:
                result_.append(gr.update(value='More')) # 2 of them 'cos we gots 2 buttons
            else:
                result_.append(gr.update(value='Restart from beginning'))
            return result_

        def use_image_history(filename_):
            global DATABASE
            result_ = [None, None, None, None, None, None] # seed, steps, width, height, cfg_scale, prompt            
            id_ = filename_[filename_.find('[')+1:filename_.rfind(']')]
            try:
                cursor_ = DATABASE.cursor()
                cursor_.execute("SELECT image.seed, settings.steps, settings.width, settings.height, settings.cfg_scale, prompt.description, image.filename \
                FROM image \
                INNER JOIN settings ON settings.id=image.settings_id \
                INNER JOIN prompt ON prompt.ROWID=image.prompt_id \
                WHERE image.id=?" , [id_])
                row_ = cursor_.fetchone()
                if row_:
                    result_ = [row_[0], row_[1], row_[2], row_[3], row_[4], row_[5]]
            except lite.Error as e:
                print(f"Database exception trying to get image history: {e}")
            return result_ + result_ + [row_[6]] # twice because text to image AND image to image are updated
            
        # UI controls    
        ima_start = gr.Variable(value=0)
        with gr.Row():
            with gr.Column():
                ima_filter_mode = gr.Dropdown(label="Filter mode", choices=["none", "contains word"], value="none")
            with gr.Column():
                ima_filter_text = gr.Textbox(label="Filter text", visible=False)
            ima_filter_mode.change(fn=lambda mode: gr.update(visible=not mode=='none'), inputs=ima_filter_mode, outputs=ima_filter_text)
        with gr.Row():
            with gr.Column():
                ima_fetch1 = gr.Button(value='Fetch', variant='primary')
            with gr.Column():
                ima_begin = gr.Button(value='Go to oldest images')
            with gr.Column():
                ima_end = gr.Button(value='Go to most recent images')
        ima_outputs = [ima_start, ima_fetch1]
        for y in range(5):
            with gr.Row():
                for x in range(5):
                    with gr.Column():
                        ima_image = gr.Image(interactive=False)
                        ima_outputs.append(ima_image)
                        ima_butt = gr.Button(value='Reuse settings ☝️', variant='secondary', visible=False)
                        ima_butt.click(fn=use_image_history, inputs=ima_butt, outputs=outputs)
                        ima_outputs.append(ima_butt)
                        ima_remove = gr.Button(value='Delete ☝️ forever', variant='secondary', visible=False)
                        ima_remove.click(fn=remove_image_forever, inputs=ima_remove, outputs=[ima_image, ima_butt, ima_remove])
                        ima_outputs.append(ima_remove)
        ima_fetch2 = gr.Button(value='Fetch', variant='primary')
        ima_outputs.append(ima_fetch2)
        ima_fetch1.click(fn=update_image_history, inputs=[ima_start, ima_filter_mode, ima_filter_text, ima_fetch1], outputs=ima_outputs)
        ima_fetch2.click(fn=update_image_history, inputs=[ima_start, ima_filter_mode, ima_filter_text, ima_fetch2], outputs=ima_outputs)    
        ima_begin.click(fn=update_image_history, inputs=[ima_start, ima_filter_mode, ima_filter_text, ima_begin], outputs=ima_outputs)
        ima_end.click(fn=update_image_history, inputs=[ima_start, ima_filter_mode, ima_filter_text, ima_end], outputs=ima_outputs)
    

def main():    
    global OUTDIR
    global DEBUG
    global GENERATOR
    
    # make sure the output directory exists
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)
    
    result, message = change_database(OUTDIR + "/sdgg.db")
    if not result:
        print(f"Could not create database! Error: {message}\n")
        quit()
        
    # the main Gradio GUI layout generation
    with gr.Blocks(title="Stable Diffusion GUI") as demo:
        with gr.Tabs():
            use_buttons = []
            # Text to image tab
            tti_prompt, tti_seed, tti_steps, tti_width, tti_height, tti_cfg_scale, tti_output1, tti_seed1, tti_output2, tti_seed2, tti_output3, tti_seed3, _, tti_use_buttons,  = \
                create_generation_tab("Text to Image", False, get_t2i_session_settings())
            use_buttons = use_buttons + tti_use_buttons
            # Image to image tab
            iti_prompt, iti_seed, iti_steps, iti_width, iti_height, iti_cfg_scale, iti_output1, iti_seed1, iti_output2, iti_seed2, iti_output3, iti_seed3, iti_image, iti_use_buttons = \
                create_generation_tab("Image to Image", True, get_i2i_session_settings())
            use_buttons = use_buttons + iti_use_buttons

            use_buttons[0].click(fn=lambda prompt, seed, source: [gr.update(value=prompt), gr.update(value=seed), gr.update(value=source)], inputs=[tti_prompt, tti_seed, tti_output1], outputs=[iti_prompt, iti_seed, iti_image])
            use_buttons[1].click(fn=lambda prompt, seed, source: [gr.update(value=prompt), gr.update(value=seed), gr.update(value=source)], inputs=[tti_prompt, tti_seed, tti_output2], outputs=[iti_prompt, iti_seed, iti_image])
            use_buttons[2].click(fn=lambda prompt, seed, source: [gr.update(value=prompt), gr.update(value=seed), gr.update(value=source)], inputs=[tti_prompt, tti_seed, tti_output3], outputs=[iti_prompt, iti_seed, iti_image])
            
            use_buttons[3].click(fn=lambda source: gr.update(value=source), inputs=iti_output1, outputs=iti_image)
            use_buttons[4].click(fn=lambda source: gr.update(value=source), inputs=iti_output2, outputs=iti_image)
            use_buttons[5].click(fn=lambda source: gr.update(value=source), inputs=iti_output3, outputs=iti_image)
            
            # Image history tab
            create_image_history_tab([tti_seed, tti_steps, tti_width, tti_height, tti_cfg_scale, tti_prompt, iti_seed, iti_steps, iti_width, iti_height, iti_cfg_scale, iti_prompt, iti_image])
            # Prompt History tab    
            create_prompt_history_tab(tti_prompt, iti_prompt)
       
    # TODO:
    # * make this unload after e.g., 5 minutes of inactivity, then reload when used again
    if not DEBUG:
        GENERATOR = T2I(width=512,
                    height=512,
                    batch_size=IMAGE_COUNT,
                    iterations=ITERATIONS,
                    outdir=OUTDIR,
                    sampler_name=SAMPLER,
                    weights=WEIGHTS,
                    full_precision=True,
                    config=CONFIG,
                    downsampling_factor=DOWNSAMPLING,
        )
        # preload the model
        GENERATOR.load_model()

    # go!    
    demo.launch()
    
if __name__ == "__main__":
    main()
    