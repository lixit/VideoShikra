"""
Adapted from: https://github.com/Vision-CAIR/MiniGPT-4/blob/main/demo.py 
"""
import argparse
import re
import cv2
import os
root_dir = os.path.join(os.getcwd(), "..")
import sys
sys.path.append(root_dir)

import torch
import gradio as gr

import decord
decord.bridge.set_bridge('torch')

from vtimellm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from vtimellm.conversation import conv_templates, SeparatorStyle
from vtimellm.model.builder import load_pretrained_model
from vtimellm.utils import disable_torch_init
from vtimellm.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, VideoExtractor
from PIL import Image
from transformers import TextStreamer
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
import clip

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--model_base", type=str, required=True, help="Path to your vicuna-7b-v1.5 huggingface checkpoint")
    parser.add_argument("--clip_path", type=str, default=os.path.join(root_dir, "checkpoints/clip/ViT-L-14.pt"))
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default=os.path.join(root_dir, "checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin"))
    parser.add_argument("--stage2", type=str, default=os.path.join(root_dir, "checkpoints/vtimellm-vicuna-v1-5-7b-stage2"))
    parser.add_argument("--stage3", type=str, default=os.path.join(root_dir, "checkpoints/vtimellm-vicuna-v1-5-7b-stage3"))
    parser.add_argument("--share", action='store_true')
    args = parser.parse_args()
    return args

# ========================================
#             Model Initialization
# ========================================

args = parse_args()
device = f'cuda:{args.gpu_id}'

disable_torch_init()
tokenizer, model, context_len = load_pretrained_model(args, args.stage2, args.stage3)
model = model.to(torch.float16).to(device)

clip_model, _ = clip.load(args.clip_path)
clip_model.eval()
clip_model = clip_model.to(device)

transform = Compose([
    Resize(224, interpolation=BICUBIC),
    CenterCrop(224),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

print('Initialization Finished')

# ========================================
#             Gradio Setting
# ========================================

TEXT_PLACEHOLDER = 'Upload your video first, or directly click the examples at the bottom of the page.'

def gradio_reset(chat_state, video_features_state, conv_state):
    if chat_state is not None:
        chat_state.messages = []
    video_features_state = None
    conv_state = {}
    return None, gr.update(value=None, interactive=True), gr.update(value='', placeholder=TEXT_PLACEHOLDER, interactive=False), gr.update(value="Upload & Start Chat", interactive=True), chat_state, video_features_state, conv_state

def upload_video(gr_video, chat_state, video_features_state, conv_state, chatbot):
    if not gr_video:
        return None, None, gr.update(interactive=True), chat_state, video_features_state, conv_state, None
    else:
        print(gr_video)
        video_loader = VideoExtractor(N=100)
        _, images = video_loader.extract({'id': None, 'video': gr_video})

        transform = Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        # print(images.shape) # <N, 3, H, W>
        images = transform(images / 255.0)
        images = images.to(torch.float16)
        with torch.no_grad():
            video_features_state = clip_model.encode_image(images.to(device))

        chatbot = chatbot + [((gr_video,), None)]
        chat_state = conv_templates["v1"].copy()
        conv_state['first'] = True

        return gr.update(interactive=False), \
            gr.update(interactive=True, placeholder='Type and press Enter'), \
            gr.update(value="Start Chatting", interactive=False), \
            chat_state, video_features_state, conv_state, chatbot

def gradio_ask(user_message, chatbot, chat_state, conv_state):
    if len(user_message) == 0:
        conv_state['should_answer'] = False
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state, conv_state
    conv_state['should_answer'] = True
    chatbot = chatbot + [[user_message, None]]
    if conv_state['first']:
        user_message = DEFAULT_IMAGE_TOKEN + '\n' + user_message
        conv_state['first'] = False
    chat_state.append_message(chat_state.roles[0], user_message)
    chat_state.append_message(chat_state.roles[1], None)
    return '', chatbot, chat_state, conv_state


def gradio_answer(chatbot, chat_state, video_features_state, conv_state, temperature):
    if not conv_state['should_answer']:
        return chatbot, chat_state
    prompt = chat_state.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
    stop_str = chat_state.sep if chat_state.sep_style != SeparatorStyle.TWO else chat_state.sep2 # plain:sep(###) v1:sep2(None)
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=video_features_state[None,].to(device),
            do_sample=True,
            temperature=temperature,
            max_new_tokens=1024,
            streamer=streamer,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    chat_state.messages[-1][-1] = outputs

    chatbot[-1][1] = outputs
    print(chat_state.get_prompt())
    print(chat_state)
    return chatbot, chat_state

def extract_frame_and_position(output: str):
    pattern = re.compile(r'(\d+):\[(\d+),(\d+),(\d+),(\d+)\]')
    matches = pattern.findall(output)
    
    frame_to_bbox = {}
    # Loop through matches to populate the dictionary
    for match in matches:
        frame_number = int(match[0])
        bbox = [int(coord) for coord in match[1:]]
        frame_to_bbox[frame_number] = bbox

    return frame_to_bbox


def display_images(video_path, chat_state, N=100):
    answer = chat_state.messages[-1][-1]
    frame_to_bbox = extract_frame_and_position(answer)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_in_memory = []
    while True:
        curr_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        ret, frame = cap.read()
        
        if not ret:
            break  # Break the loop if there are no frames left to read
        
        # Process the frame (example: convert to RGB)
        imagine_frame = round(curr_frame / total_frames * 100, 0)
        if imagine_frame in list(frame_to_bbox.keys()):
            
            bbox = frame_to_bbox[imagine_frame]
            height, width = frame.shape[:2]
            xmin = int(bbox[0] * width / 100)
            ymin = int(bbox[1] * height / 100)
            xmax = int(bbox[2] * width / 100)
            ymax = int(bbox[3] * height / 100)
            frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        frames_in_memory.append(frame)
        
    output_filename = 'output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    frame_size = (width, height)  # Frame size, adjust to your needs
    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)
    
    for frame in frames_in_memory:
        video_writer.write(frame)
        
    video_writer.release()
    return output_filename



with gr.Blocks() as demo:
    gr.Markdown('''# Demo for VTimeLLM''')

    with gr.Row():
        with gr.Column():
            video = gr.Video()

            upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
            clear = gr.Button("Reset")
            
            temperature = gr.Slider(
                minimum=0,
                maximum=2.0,
                value=0.05,
                step=0.01,
                interactive=True,
                label="Temperature",
            )
        with gr.Column():
            chat_state = gr.State()
            video_features_state = gr.State()
            conv_state = gr.State({})
            chatbot = gr.Chatbot(label='VTimeLLM')
            text_input = gr.Textbox(label='User', placeholder=TEXT_PLACEHOLDER, interactive=False)
            

    # with gr.Column():
    #     gr.Examples(examples=[
    #         [os.path.join(root_dir, f"images/demo.mp4"), "Explain why the video is funny."],
    #     ], inputs=[video, text_input])
        
    with gr.Column():
        video1 = gr.Video()
        
    upload_button.click(upload_video, [video, chat_state, video_features_state, conv_state, chatbot],
                        [video, text_input, upload_button, chat_state, video_features_state, conv_state, chatbot])
    text_input.submit(gradio_ask, [text_input, chatbot, chat_state, conv_state], 
                      [text_input, chatbot, chat_state, conv_state])\
                          .then(gradio_answer, 
                                [chatbot, chat_state, video_features_state, conv_state, temperature], 
                                [chatbot, chat_state])\
                            .then(display_images, [video, chat_state], [video1])
    clear.click(gradio_reset, [chat_state, video_features_state, conv_state], 
                [chatbot, video, text_input, upload_button, chat_state, video_features_state, conv_state], queue=False)

demo.queue().launch(share=args.share)
