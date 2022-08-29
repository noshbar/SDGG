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
import uuid
import threading
from PIL import Image,PngImagePlugin
from numpy import asarray, arange
sys.path.append('.')
from ldm_old.simplet2i import T2I

# hack hack la la hack
arg_parser = argparse.ArgumentParser(description='Stable Diffusion Gradio GUI')
arg_parser.add_argument('-df', '--downsampling_factor', dest='downsampling_factor', type=int, help='BUGGY! for less VRAM usage, lower quality, faster generation, try 9 as a value', default=8)
arg_parser.add_argument('-s', '--sampler', dest='sampler', type=str, help='which sampler to use (klms/plms)', default='plms')
arg_parser.add_argument('-p', '--parallel', dest='PARALLEL', type=bool, help='generate the entire batch at once, slightly quicker, uses more VRAM', default=False)
arg_parser.add_argument('-bs', '--batch_size', dest='IMAGE_COUNT', type=int, help='with -p, how many images to generate if you''re having VRAM issues (1..3)', default=3)
arg_parser.add_argument('-of', '--output_folder', dest='OUTPUT_FOLDER', type=str, help='the sub-folder within ./outputs to store generated images and the prompt database', default='sdgg')
arg_parser.add_argument('-ip', '--instant_preview', dest='INSTANT_PREVIEW', type=str, help='enable a hack for instant previewing of generated pictures, 1-by-1', default=False)
arg_parser.add_argument('-gs', '--gfpgan-source', dest='GFP_FOLDER', type=str, help='where to find the "gfpgan" folder', default='.') #  https://github.com/TencentARC/GFPGAN/
arg_parser.add_argument('-gm', '--gfpgan-model', dest='GFP_MODEL_PATH', type=str, help='where to find the "GFPGANv1.3.pth" model', default='./models/gfpgan') # https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth
arg_parser.add_argument('-gd', '--gfpgan-disable', dest='GFP_DISABLED', type=str, help='disable GFPGAN', default=False)
arg_parser.add_argument('-rs', '--real-esrgan-source', dest='REAL_ESRGAN_FOLDER', type=str, help='where to find the "gfpgan" folder', default=None) #
arg_parser.add_argument('-rm', '--real-esrgan-model', dest='REAL_ESRGAN_MODEL_PATH', type=str, help='where to find the model', default=None) #
arg_parser.add_argument('-rp', '--real-esrgan-processor', dest='REAL_ESRGAN_PROCESSOR', type=str, help='RealESRGAN processor (cpu/cuda)', default='cuda')
arg_parser.add_argument('-rd', '--real-esrgan-disable', dest='REAL_ESRGAN_DISABLED', type=str, help='disable RealESRGAN', default=False)
arg_parser.add_argument('-Z', '--debug', dest='DEBUG', type=str, help='Disable loading of ML stuff to debug Gradio stuff quicker', default=False)
args = arg_parser.parse_args()

# globals here we go!    
OUTDIR   = "outputs/" + args.OUTPUT_FOLDER
CONFIG   = "configs/stable-diffusion/v1-inference.yaml"
WEIGHTS  = "models/ldm/stable-diffusion-v1/model.ckpt"
DEBUG    = args.DEBUG

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
    
GFPGAN = None    
REAL_ESRGANS = None
DATABASE = None

# and now the hacks for instant previews during generation, 1-by-1, not at all 3 at once
PREVIEW        = args.INSTANT_PREVIEW   
PREVIEW_IMAGES = []
PREVIEW_EVENTS = []
GENERATING     = None # going to react to image.change later, but deleting causes changes too
    
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
                
                
def get_prompt_and_settings_id(init_image_filename, prompt, seed, steps, width, height, cfg_scale):
    global DATABASE
    global OUTDIR
    global GENERATOR
    
    # try find if the image was already generated
    prompt = prompt.lower()
    try:
        cursor = DATABASE.cursor()
        cursor.execute("SELECT rowid FROM prompt WHERE description=?", [prompt])
        result = cursor.fetchone()
        if result is None:
            cursor = DATABASE.cursor()
            cursor.execute("INSERT INTO prompt(description) VALUES(?)", [prompt])
            DATABASE.commit()
            prompt_id = cursor.lastrowid
        else:
            prompt_id = result[0]

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
        else:
            settings_id = result[0]
    except lite.Error as e:
        message = 'Database exception: ' + e.args[0]
    return prompt_id, settings_id        
                
def get_image_id(filename, prompt_id, settings_id, seed):
    global DATABASE
    cursor = DATABASE.cursor()
    cursor.execute("INSERT INTO image(filename, prompt_id, settings_id, seed) values(?, ?, ?, ?)", [filename, prompt_id, settings_id, seed])
    cursor.execute("SELECT last_insert_rowid();")
    row = cursor.fetchone();
    DATABASE.commit()
    return row[0]
                
# preview hack start ================================================================================================                
def provide_preview(for_index):
    global PREVIEW_IMAGES
    global PREVIEW_EVENTS
    global GENERATING
    si = str(for_index)
    delete_buttons   = [gr.update(variant='secondary'), gr.update(variant='secondary'), gr.update(variant='secondary')]
    enhance_controls = [gr.update(label='Post process'), gr.update(variant='secondary'), gr.update(label='Post process'), gr.update(variant='secondary'), gr.update(label='Post process'), gr.update(variant='secondary')]
    if not GENERATING: # do some nops
        return [gr.update(interactive=False), gr.update(interactive=False), gr.update(variant='secondary')] + delete_buttons + enhance_controls
    while not PREVIEW_EVENTS[for_index].isSet():
        PREVIEW_EVENTS[for_index].wait(1)
    visible = (for_index+1 == SHOW_COUNT)
    for i in range(for_index+1):
        value = "Delete forever ["+str(PREVIEW_IMAGES[i]['id'])+"]"
        delete_buttons[i] = gr.update(visible=visible, value=value)
        enhance_controls[i * 2 + 0] = gr.update(visible=True)
        enhance_controls[i * 2 + 1] = gr.update(visible=True)
    return [PREVIEW_IMAGES[for_index]['filename'], PREVIEW_IMAGES[for_index]['seed'], gr.update(visible=True)] + delete_buttons + enhance_controls
        
def handle_preview(image, seed):
    global OUTDIR
    global PREVIEW_IMAGES
    global PREVIEW_EVENTS
    global GENERATING

    filename = os.path.join(OUTDIR, str(uuid.uuid4()) + ".png")
    image.save(filename)
    
    index = len(PREVIEW_IMAGES)
    id = get_image_id(filename, GENERATING['prompt_id'], GENERATING['settings_id'], seed)
    PREVIEW_IMAGES.append({'filename':filename, 'seed':seed, 'id':id})
    PREVIEW_EVENTS[index].set()
     
def generation_thread(init_image_filename, prompt, seed, steps, width, height, cfg_scale):   
    global DATABASE
    global OUTDIR
    global GENERATOR
    global GENERATING
    global IMAGE_COUNT
    global ITERATIONS
    if init_image_filename:
        results = GENERATOR.prompt2image(prompt=prompt, outdir=OUTDIR, seed=seed, steps=steps, cfg_scale=cfg_scale, init_img=init_image_filename, batch_size=IMAGE_COUNT, iterations=ITERATIONS, image_callback=handle_preview)
    else:
        results = GENERATOR.prompt2image(prompt=prompt, outdir=OUTDIR, seed=seed, steps=steps, width=width, height=height, cfg_scale=cfg_scale, batch_size=IMAGE_COUNT, iterations=ITERATIONS, image_callback=handle_preview)
    GENERATING = None
        
def generate_with_preview(init_image_filename, prompt, seed, steps, width, height, cfg_scale):
    global OUTDIR
    global GENERATOR
    global PREVIEW
    global PREVIEW_IMAGES
    global PREVIEW_EVENTS
    global GENERATING

    PREVIEW_IMAGES = []
    PREVIEW_EVENTS = [threading.Event(), threading.Event(), threading.Event()]
    GENERATING     = None
   
    if prompt.strip() == "":
        return "Please enter a valid prompt first"

    if not init_image_filename:
        save_t2i_session_settings(prompt, seed, steps, cfg_scale, width, height)

    prompt = prompt.lower()
    prompt_id, settings_id = get_prompt_and_settings_id(init_image_filename, prompt, seed, steps, width, height, cfg_scale)
        
    # start the image generation thread
    GENERATING = { 'prompt_id':prompt_id, 'settings_id':settings_id }
    thread = threading.Thread(target=generation_thread, args=(init_image_filename, prompt, seed, steps, width, height, cfg_scale,))        
    thread.start()
                
    # update the message label with a unique message to trigger `change`, and hide the 3 delete buttons and enhance controls
    buttons = [gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)]
    return ["Please wait until all 3 are generated... [hack#: " + str(uuid.uuid4()) + "]"] + buttons
# preview hack end ================================================================================================                
        
        
def generate(init_image_filename, prompt, seed, steps, width, height, cfg_scale):
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

    prompt = prompt.lower()
    prompt_id, settings_id = get_prompt_and_settings_id(init_image_filename, prompt, seed, steps, width, height, cfg_scale)
    # generate the images
    if init_image_filename:
        results = GENERATOR.prompt2image(prompt=prompt, outdir=OUTDIR, seed=seed, steps=steps, cfg_scale=cfg_scale, init_img=init_image_filename, batch_size=IMAGE_COUNT, iterations=ITERATIONS)
    else:
        results = GENERATOR.prompt2image(prompt=prompt, outdir=OUTDIR, seed=seed, steps=steps, width=width, height=height, cfg_scale=cfg_scale, batch_size=IMAGE_COUNT, iterations=ITERATIONS)
    for index, result in enumerate(results):
        filename = os.path.join(OUTDIR, str(uuid.uuid4()) + ".png")
        result[0].save(filename)
        images[index] = asarray(result[0])
        seeds[index] = result[1]
        ids[index] = get_image_id(filename, prompt_id, settings_id, seed)
                
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
        "prompt": "corgi wargaming miniature", \
        "seed": 1604788626, \
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
        if (not filter_mode) or (filter_mode == 'none') or (not filter_text) or (filter_text.strip() == ""):
            cursor.execute("SELECT filename, seed, id FROM image WHERE id>=? ORDER BY id LIMIT ?", [start, amount+1])
        else:   
            cursor.execute('SELECT filename, seed, id FROM image WHERE id>=? AND prompt_id IN (SELECT rowid FROM prompt WHERE description LIKE ''?'') ORDER BY id LIMIT ?', [start, '%'+filter_text+'%', amount+1])
        images = cursor.fetchall()
        count = len(images)
        more = count > amount
        if more:
            start = images[-1][2]
            images.pop() # don't use the last one, it was for knowing if there's more
        for image in images:
            result.append({ 'filename': image[0], 'seed': image[1], 'id': image[2] })
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
    global PREVIEW
    if PREVIEW:
        return generate_with_preview(None, prompt, seed, steps, width, height, cfg_scale)
    else:
        return generate(None, prompt, seed, steps, width, height, cfg_scale)

def generate_i2i(image_numpy, prompt, seed, steps, cfg_scale):
    global OUTDIR
    global PREVIEW
    image = Image.fromarray(image_numpy)    
    filename = OUTDIR + "/temp.png"
    image.save(filename)
    width, height = image.size
    steps = int(steps * 1.34) # no idea why, but the image to image script seems to use 74% of what you provide
    
    if PREVIEW:
        return generate_with_preview(filename, prompt, seed, steps, width, height, cfg_scale)
    else:
        return generate(filename, prompt, seed, steps, width, height, cfg_scale)
    
def gfpgan_image(image):
    if image is None:
        return None
    _, _, restored_img = GFPGAN.enhance(image[:,:,::-1], has_aligned=False, only_center_face=False, paste_back=True)
    return restored_img[:,:,::-1]
                    
def realesrgan_image(image, upscaler):
    if image is None:
        return None
    result, _ = REAL_ESRGANS[upscaler].enhance(image[:,:,::-1])
    return result[:,:,::-1]
                    
def post_process_image(image, process):
    if 'face' in process:
        return gfpgan_image(image)
    else:
        return realesrgan_image(image, process)
                    
def create_generation_tab(title, with_image_input, session):
    global PREVIEW
    global GFPGAN
    global REAL_ESRGANS
    
    post_processes = []
    if GFPGAN:
        post_processes.append('Enhance face')
    if not REAL_ESRGANS is None:
        for instance in REAL_ESRGANS.keys():
            post_processes.append(instance)
    
    with gr.TabItem(title):
        if with_image_input:
            image = gr.Image(label="Guide/initial image (keep small with dimensions that are multiples of 64)", interactive=True)
            if len(post_processes):
                with gr.Row():
                    with gr.Column():
                        top_post_process_choices = gr.Dropdown(label='Post-process', choices=post_processes, value=post_processes[0])
                    with gr.Column():
                        top_post_process_btn = gr.Button(value="Process")
                        top_post_process_btn.click(fn=post_process_image, inputs=[image, top_post_process_choices], outputs=image)
        else:
            image = None
            
        local_prompt = gr.Textbox(label="Prompt        (the sentence to generate the image from)", max_lines=1, value=session["prompt"], placeholder='For inspiration, check out https://lexica.art')
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
                        if with_image_input:
                            local_width = gr.Textbox(label="Width", value=session["width"], interactive=False, visible=False)
                        else:
                            local_width = gr.Dropdown(label="Width", value=session["width"], choices=[256,320,384,448,512,576,640,704,768,832,896,960,1024], interactive=True)
                    with gr.Column():
                        if with_image_input:
                            local_height = gr.Textbox(label="Height", value=session["height"], interactive=False, visible=False)
                        else:
                            local_height = gr.Dropdown(label="Height", value=session["height"], choices=[256,320,384,448,512,576,640,704,768,832,896,960,1024], interactive=True)
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
        if not PREVIEW:
            gr.Markdown(value='(to see images *as* they are generated, you can try running this script with `-ip true`, but this *is experimental*)', show_label=False)
        else:
            gr.Markdown(value='**DO NOT TOUCH ANYTHING** until *ALL* images are done generating', show_label=False)
        local_output         = [None, None, None]
        local_image_seed     = [None, None, None]
        make_current         = [None, None, None]
        gfpgan_btn           = [None, None, None]
        remove_btn           = [None, None, None]
        use_btn              = [None, None, None]
        post_process_choices = [None, None, None]
        post_process_btn     = [None, None, None]
        with gr.Row():
            for column in range(3):
                with gr.Column():
                    local_output[column] = gr.Image(label="Generated image", interactive=False)
                    with gr.Row():
                        with gr.Column():
                            local_image_seed[column] = gr.Textbox(label="Seed", max_lines=1, interactive=False)
                        with gr.Column():
                            make_current[column] = gr.Button(value="Make current seed")
                            make_current[column].click(fn=lambda s: gr.update(value=int(s)), inputs=[local_image_seed[column]], outputs=local_image_seed[column])
                    use_btn[column] = gr.Button(value="Use for image-to-image generation", visible=False)
                    if len(post_processes):
                        with gr.Row():
                            with gr.Column():
                                post_process_choices[column] = gr.Dropdown(label='Post-process', choices=post_processes, value=post_processes[0])
                            with gr.Column():
                                post_process_btn[column] = gr.Button(value="Process")
                                post_process_btn[column].click(fn=post_process_image, inputs=[local_output[column], post_process_choices[column]], outputs=local_output[column])
                    remove_btn[column] = gr.Button(value="Remove forever", visible=False)
                    remove_btn[column].click(fn=remove_image_forever, inputs=remove_btn[column], outputs=[local_output[column], use_btn[column], remove_btn[column]])
        message = gr.Textbox(label="Messages", max_lines=1, interactive=False)
        
        if PREVIEW:
            first  = gr.Variable(value=0)
            second = gr.Variable(value=1)
            third  = gr.Variable(value=2)
            cascading_common = [remove_btn[0], remove_btn[1], remove_btn[2]]
            post_process_outputs = []
            if len(post_processes):
                for i in range(3):
                    post_process_outputs += [post_process_choices[i], post_process_btn[i]]
            cascading_common += post_process_outputs
            message.change(fn=provide_preview, inputs=first, outputs=[local_output[0], local_image_seed[0], use_btn[0]] + cascading_common)
            local_output[0].change(fn=provide_preview, inputs=second, outputs=[local_output[1], local_image_seed[1], use_btn[1]] + cascading_common)
            local_output[1].change(fn=provide_preview, inputs=third, outputs=[local_output[2], local_image_seed[2], use_btn[2]] + cascading_common)
            if len(post_processes):
                outputs_ = [message, remove_btn[0], remove_btn[1], remove_btn[2]] + post_process_outputs
            else:
                outputs_ = [message, remove_btn[0], remove_btn[1], remove_btn[2]]
        else:
            outputs_ = [local_output[0], local_output[1], local_output[2], local_image_seed[0], local_image_seed[1], local_image_seed[2], message]
            outputs_ += [use_btn[0], use_btn[1], use_btn[2], remove_btn[0], remove_btn[1], remove_btn[2]]
        if with_image_input:
            generate_btn.click(fn=generate_i2i, inputs=[image, local_prompt, local_seed, local_steps, local_cfg_scale], outputs=outputs_)
        else:
            generate_btn.click(fn=generate_t2i, inputs=[local_prompt, local_seed, local_steps, local_width, local_height, local_cfg_scale], outputs=outputs_)
        return local_prompt, local_seed, local_steps, local_width, local_height, local_cfg_scale, local_output[0], local_image_seed[0], local_output[1], local_image_seed[1], local_output[2], local_image_seed[2], image, [use_btn[0], use_btn[1], use_btn[2]]
        
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
        def get_page_ids(filter_mode_, filter_text_):
            global DATABASE
            cursor_ = DATABASE.cursor()
            if (not filter_mode_) or (filter_mode_ == 'none') or (filter_text_.strip() == ""):
                cursor_.execute("SELECT id FROM (SELECT id, ROW_NUMBER() OVER(ORDER BY id) AS rownumber FROM image) WHERE (rownumber-1)%25=0")
            else:   
                cursor_.execute('SELECT id FROM( \
                                    SELECT id, filename, ROW_NUMBER() OVER(ORDER BY id) AS rownumber FROM image WHERE prompt_id IN \
                                    (SELECT rowid FROM prompt WHERE description LIKE ''?'') \
                                    ) WHERE (rownumber-1)%25=0', ['%'+filter_text_+'%'])
                                    
            rows_ = cursor_.fetchall()
            result_ = []
            for row_ in rows_:
                result_.append(row_[0])
            return result_
            
            
        def update_image_history(start_, filter_mode_, filter_text_):
            result_ = []
            history_, next_start_, more_ = get_images_history(start_, 25, filter_mode_, filter_text_)
            if more_:
                result_.append(next_start_)
                result_.append(gr.update(visible=True)) # 2 of them 'cos we gots 2 buttons
            elif not history_:
                result_.append(1)
                result_.append(gr.update(visible=False))
            else:
                result_.append(1)
                result_.append(gr.update(value='Restart from beginning', visible=True))
            count_ = len(history_)
            if count_:
                for index_ in range(count_):
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
            for index in range(25-count_):
                result_.append(gr.update(value=None, visible=False))
                result_.append(gr.update(visible=False)) # for the reuse settings button
                result_.append(gr.update(visible=False)) # for the reuse settings button
            if more_:
                result_.append(gr.update(visible=True)) # 2 of them 'cos we gots 2 next page buttons
            elif not history_:
                result_.append(gr.update(visible=False))
            else:
                result_.append(gr.update(value='Restart from beginning', visible=True))
            return result_

        def use_image_history(filename_):
            global DATABASE
            result_ = [None, None, None, None, None, None, None] # seed, steps, width, height, cfg_scale, prompt            
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
                    result_ = [row_[0], row_[1], row_[2], row_[3], row_[4], row_[5], row_[6]]
            except lite.Error as e:
                print(f"Database exception trying to get image history: {e}")
            return result_ + result_ + [result_[0]] # twice because text to image AND image to image are updated, and the seed for a 3rd time for the 1st output on T2I tab

        def do_filter(filter_mode_, filter_text_):
            page_ids_ = get_page_ids(filter_mode_, filter_text_)
            page_numbers_ = arange(1, len(page_ids_)+1).tolist()
            # would be really nice to be able to early out here, but got a whoooole lotta UI's updates to honour
            
            result_ = update_image_history(0, filter_mode_, filter_text_)
            result_.append(page_ids_)
            
            if len(page_numbers_):
                result_.append(gr.update(value=1, choices=page_numbers_, visible=True))
                new_label_ = "of " + str(page_numbers_[-1])
                result_.append(gr.update(value=new_label_, visible=True))
                result_.append(gr.update(visible=True)) # goto page
                result_.append(gr.update(visible=True)) # goto most recent
            else:
                result_.append(gr.update(visible=False))
                result_.append(gr.update(visible=False))
                result_.append(gr.update(visible=False)) # goto page
                result_.append(gr.update(visible=False)) # goto most recent
            result_.append(filter_text_)
            return result_      
            
        def apply_filter(filter_mode_, filter_text_):
            return do_filter(filter_mode_, filter_text_)

        def reset_filter():
            return do_filter(None, None)
            
        def goto_page(filter_mode_, filter_text_, page_number_, page_ids_):
            if page_ids_:
                id_ = page_ids_[page_number_-1] # page numbers are 1..N, arrays are 0..N-1
            else:
                id_ = 0
            return update_image_history(id_, filter_mode_, filter_text_) + [gr.update(value='Go to page')]

        def next_page(filter_mode_, filter_text_, current_page_, page_ids_):
            if current_page_ == len(page_ids_):
                new_page_ = 1
            else:
                new_page_ = current_page_ + 1
            id_ = page_ids_[new_page_ - 1]
            result_ = update_image_history(id_, filter_mode_, filter_text_)
            result_.append(new_page_)
            return result_

        def most_recent(filter_mode_, filter_text_, pages_):
            if pages_:
                index_ = pages_[-1]
            else:
                index_ = 0
            result_ = update_image_history(index_, filter_mode_, filter_text_)
            result_.append(len(pages_))
            return result_
            
        # UI controls    
        ima_start = gr.Variable(value=0)
        with gr.Row():
            with gr.Column():
                ima_filter_mode = gr.Dropdown(label="Filter mode", choices=["none", "contains word"], value="contains word")
            with gr.Column():
                ima_filter_text_variable = gr.Variable(value="")
                ima_filter_text = gr.Textbox(label="Filter text", interactive=True, max_lines=1, placeholder="Type a keyword to search for and press ENTER, or hit Apply")
            with gr.Column():
                ima_apply_filter = gr.Button(value='Apply', variant='primary')
                ima_reset_filter = gr.Button(value='Reset filter (also helps if you deleted a bunch)')
            ima_filter_mode.change(fn=lambda mode: gr.update(interactive=not mode=='none'), inputs=ima_filter_mode, outputs=ima_filter_text)
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        ima_gotopage = gr.Button(value='Do initial load', variant='primary')
                    with gr.Column():
                        page_ids_ = get_page_ids(None, None)
                        ima_page_numbers = arange(1, len(page_ids_)+1).tolist()
                        ima_page_map = gr.Variable(value=page_ids_)
                    with gr.Column():
                        if ima_page_numbers:
                            choices_ = ima_page_numbers
                            selected_ = ima_page_numbers[0]
                        else:
                            choices_ = [1]
                            selected_ = 1
                        ima_pages = gr.Dropdown(show_label=False, choices=choices_, value=selected_, interactive=True)
                    with gr.Column():
                        ima_pagecount_text = gr.Text(value="of " + str(len(choices_)), show_label=False, interactive=False)
            with gr.Column():
                ima_fetch1 = gr.Button(value='Next page (hit Apply if missing new pages)', variant='primary', visible=False)
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
        ima_fetch2 = gr.Button(value='Next page (hit Apply if missing new pages)', variant='primary', visible=False)
        ima_outputs.append(ima_fetch2)
        
        next_outputs = ima_outputs + [ima_pages]
        ima_fetch1.click(fn=next_page, inputs=[ima_filter_mode, ima_filter_text_variable, ima_pages, ima_page_map], outputs=next_outputs)
        ima_fetch2.click(fn=next_page, inputs=[ima_filter_mode, ima_filter_text_variable, ima_pages, ima_page_map], outputs=next_outputs)    
        ima_end.click(fn=most_recent, inputs=[ima_filter_mode, ima_filter_text_variable, ima_page_map], outputs=ima_outputs + [ima_pages])
        ima_gotopage.click(fn=goto_page, inputs=[ima_filter_mode, ima_filter_text_variable, ima_pages, ima_page_map], outputs=ima_outputs + [ima_gotopage])
        ima_moar = ima_outputs + [ima_page_map, ima_pages, ima_pagecount_text, ima_gotopage, ima_end, ima_filter_text_variable]
        ima_apply_filter.click(fn=apply_filter, inputs=[ima_filter_mode, ima_filter_text], outputs=ima_moar)
        ima_filter_text.submit(fn=apply_filter, inputs=[ima_filter_mode, ima_filter_text], outputs=ima_moar)
        ima_reset_filter.click(fn=reset_filter, inputs=None, outputs=ima_moar)
        
def try_init_real_esrgan():
    global REAL_ESRGANS
    global args
    
    if args.REAL_ESRGAN_DISABLED:
        return
    
    import glob
    
    model_possibilities = []
    for model in glob.glob("models/**/RealESRGAN_*.pth", recursive = True):
        model_possibilities.append(model)
    for model in glob.glob("src/**/RealESRGAN_*.pth", recursive = True):
        model_possibilities.append(model)
    if not args.REAL_ESRGAN_MODEL_PATH is None:
        for model in glob.glob(os.path.join(args.REAL_ESRGAN_MODEL_PATH, "/**/RealESRGAN_*.pth"), recursive = True):
            model_possibilities.append(model)
            
    if not len(model_possibilities):
        print("RealESRGAN models not found and will not be used.\n")
        return
    
    try:
        if not args.REAL_ESRGAN_FOLDER is None:
            sys.path.append(os.path.abspath(args.REAL_ESRGAN_FOLDER))
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet

        import torch
        REAL_ESRGANS = {}
        for model_path in model_possibilities:
            if 'anime' in model_path:
                net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            else:
                net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            head, tail = os.path.split(model_path)
            instance = RealESRGANer(scale=2, model_path=model_path, model=net, pre_pad=0, half=False)
            instance.model.name = tail
            instance.device = torch.device(args.REAL_ESRGAN_PROCESSOR)
            REAL_ESRGANS[tail] = instance
        print("RealESRGAN loaded successfully\n")
    except ImportError as e:
        print(f'Failed to import RealESRGAN: {e}\n')
        pass
    
                
def try_init_gfpgan():
    global GFPGAN
    global args
    
    if args.GFP_DISABLED:
        return
    
    model_possibilities = [os.path.join(args.GFP_MODEL_PATH, 'GFPGANv1.3.pth'), './src/gfpgan/experiments/pretrained_models/GFPGANv1.3.pth']
    model_path = None
    for possibility in model_possibilities:
        print(f"Trying GFPGAN model path: {possibility}... ", end='')
        if os.path.isfile(possibility):
            print("found!")
            model_path = possibility
            break
        else:
            print('x')
    if model_path is None:
        print('Could not find model, GFPGAN disabled\n')
        return
            
    src_possibilities = [os.path.join(args.GFP_FOLDER, 'gfpgan'), './src']
    for src_path in src_possibilities:
        print(f"Trying GFPGAN source path: {src_path}... ", end='')
        if os.path.isdir(src_path):
            print('found!')
            sys.path.append(src_path)
            try:
                from gfpgan import GFPGANer
                GFPGAN = GFPGANer(model_path=model_path, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)
                print('Using GFPGAN.\n\n *** NOTE: Any face enhancements you make are NOT saved to disk, if you like the changes, right click and save the image.')
                print('If you want to reapply the enhancements in future, browse the image history and reuse the image, which lets you enhance it from the T2I or I2I tabs.\n')
                return
            except ImportError as e:
                print(f'Failed to import GFPGAN: {e}\n')
                pass
        else:
            print('x')
    print("\nCould not find GFPGAN\n")
    
def main():    
    global OUTDIR
    global DEBUG
    global GENERATOR
        
    #RealESRGAN check    
    try_init_real_esrgan()
        
    #GFPGAN check    
    try_init_gfpgan()
        
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
            create_image_history_tab([tti_seed, tti_steps, tti_width, tti_height, tti_cfg_scale, tti_prompt, tti_output1, iti_seed, iti_steps, iti_width, iti_height, iti_cfg_scale, iti_prompt, iti_image, tti_seed1])
            # Prompt History tab    
            create_prompt_history_tab(tti_prompt, iti_prompt)
       
    # TODO:
    # * make this unload after e.g., 5 minutes of inactivity, then reload when used again
    if not DEBUG:
        GENERATOR = T2I(width=512,
                    height=512,
                    batch_size=IMAGE_COUNT,
                    iterations=ITERATIONS,
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
    