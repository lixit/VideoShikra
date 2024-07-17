import random
import copy
import json
import torch
import transformers
from transformers import CLIPImageProcessor, CLIPVisionModel
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import pathlib 
import re

from vtimellm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from vtimellm import conversation as conversation_lib
from vtimellm.mm_utils import tokenizer_image_token, extract_frames


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    val_data_path: Optional[str] = field(default=None,
                            metadata={"help": "Path to the validation data."})
    lazy_preprocess: bool = False
    feat_folder: Optional[str] = field(default=None)
    val_feat_folder: Optional[str] = field(default=None)

def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_glm(
    sources,
    tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    
    input_ids = []
    targets = []

    for source in sources:
        tokens, loss_masks = [tokenizer.get_command("[gMASK]"), tokenizer.get_command("sop")], [0, 0]
        def _update(_tokens: List[int], value: int = 1):
            value = int(value)
            tokens.extend(_tokens)
            loss_masks.extend([value] * len(_tokens))
        
        for conv in source:
            if conv["from"] == 'human':
                role_token = tokenizer.get_command("<|user|>")
                loss = False
            else:
                role_token = tokenizer.get_command("<|assistant|>")
                loss = True
                
            token_id = [role_token] + tokenizer_image_token(conv['value'], tokenizer)[2:]
            _update(token_id, loss)
        _update([tokenizer.eos_token_id], False)

        loss_masks = [False] + loss_masks[:-1]
        labels = [(t if m else IGNORE_INDEX) for t, m in zip(tokens, loss_masks)]

        input_ids.append(tokens)
        targets.append(labels)

        # print("Sanity Check >>>>>>>>>>>>>")
        # for t, m in zip(tokens, labels):
        #     decoded =  tokenizer.tokenizer.index_special_tokens[t] \
        #         if t in tokenizer.tokenizer.index_special_tokens \
        #         else tokenizer.decode([t])
        #     print("%20s: %6d -> %6d" % (repr(decoded), t, m))
        # print("<<<<<<<<<<<<< Sanity Check")

    return dict(
        input_ids=torch.tensor(input_ids),
        labels=torch.tensor(targets),
    )

def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # print(conversations)
    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )



def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN + '\n' # completely remove the question
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

# example = {"img_path": "COCO_train2014_000000310289.jpg", "expression": "the giant doughnut with white icing and red , white , and blue sprinkles", "bbox": [334.72, 298.08, 522.88, 450.23999999999995], "dataset_name": "refcocog", "height": 480, "width": 640}

# example_return =  {
#     "conversations": [
#       {
#         "from": "human",
#         "value": "Write a terse but informative summary of the picture.\n<image>"
#       },
#       {
#         "from": "gpt",
#         "value": "a grey watch with an army style strap"
#       }
#     ]
# }

def normalize_bbox(bbox, width, height):
    """
    Normalize the bbox
    """
    xmin, ymin, xmax, ymax = bbox
    xmin = int(round(xmin / width, 2) * 100)
    ymin = int(round(ymin / height, 2) * 100)
    xmax = int(round(xmax / width, 2) * 100)
    ymax = int(round(ymax / height, 2) * 100)
    bbox = [xmin, ymin, xmax, ymax]
    return re.sub(r'\s+', '', str(bbox))

def get_conversation(example: dict, template: list):
    example_return = {
        "image": "00223/002239345.jpg",
        "conversations": [
        {
            "from": "human",
            "value": "Write a terse but informative summary of the picture.\n<image>"
        },
        {
            "from": "gpt",
            "value": ""
        }
        ]
    }
    # generate question
    expression = example["expression"]
    tmp = random.choice(template)
    question = tmp.replace("<expr>", expression)
    
    # generate answer
    answer = normalize_bbox(example["bbox"], example["width"], example["height"])
    
    example_return["image"] = example["img_path"]
    example_return["conversations"][0]["value"] = question
    example_return["conversations"][1]["value"] = answer
    return example_return


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 feat_folder: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 template_path: str = "config/_base_/dataset/template/REC.json"):
        super(LazySupervisedDataset, self).__init__()

        self.tokenizer = tokenizer
        if data_path.endswith('.jsonl'):
            self.list_data_dict = list(open(data_path, "r"))
            self.is_jsonl = True
        else:
            self.list_data_dict = json.load(open(data_path, "r"))
            self.is_jsonl = False
            
        self.feat_folder = feat_folder
        self.templates = json.load(open(template_path, 'r', encoding='utf8'))

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # convert to vtimellm style
        if self.is_jsonl:
            orginal_source = json.loads(self.list_data_dict[i])
            source = copy.deepcopy(get_conversation(orginal_source, self.templates))
        else:
            source = copy.deepcopy(self.list_data_dict[i])

        data_type = 'video'
        if '<image>' in source['conversations'][0]['value']:
            source['conversations'][0]['value'] = source['conversations'][0]['value'].replace('<image>', '<video>')
            data_type = 'image'

        # if 'meta' in source:
        #     def convert(duration, x):
        #         x = x / duration * 100
        #         x = str(min(round(x), 99))
        #         if len(x) == 1:
        #             x = "0" + x
        #         return x

        #     replace_set = []
        #     for k, v in source['meta']['token'].items():
        #         if isinstance(v, list):
        #             replace_set.append((k, str(v)))
        #         else:
        #             replace_set.append((k, convert(source['meta']['duration'], v)))
        #     for l in range(len(source['conversations'])):
        #         for x1, x2 in replace_set:
        #             source['conversations'][l]['value'] = source['conversations'][l]['value'].replace(x1, x2)
        # image = torch.zeros((100 if data_type == 'video' else 1, 768), dtype=torch.float16)
        
        if data_type == 'video':
            vid_path = pathlib.Path(self.feat_folder) / source["meta"]["vid_path"]
            vid_path = vid_path.as_posix()
            begin_fid = source["meta"]["begin_fid"]
            end_fid = source["meta"]["end_fid"]
            asked_frames = source["meta"]["asked_frames"]
            # Nx3xHxW
            images = extract_frames(vid_path, begin_fid, end_fid, asked_frames)
        else:
            image_path = '{}/{}'.format(self.feat_folder, source["image"])
            images = Image.open(image_path)
            
        inputs = processor(images=images, return_tensors="pt")
        # Move inputs to the defined device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        select_hidden_state = outputs.hidden_states[-2]
        image_features = select_hidden_state[:, 1:]
            

        # try:
        #     feature_path = '{}/{}.npy'.format(self.feat_folder, source['id'])
        #     image = np.load(feature_path) # <N, 768> float16
        #     image = torch.from_numpy(image)
        #     if data_type == 'image' and len(image.shape) == 1: # <768>
        #         image = image.unsqueeze(0) # torch.Size([1, 768])
        # except Exception as e:
        #     print(e)
        #     return random.choice(self)

        if getattr(self.tokenizer, 'name', None) == 'GLMTokenizer':
            data_dict = preprocess_glm([source["conversations"]], self.tokenizer)
        else:
            data_dict = preprocess(
                [source["conversations"]],
                self.tokenizer,
                has_image=True)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        data_dict['image'] = image_features
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args: DataArguments) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(data_path=data_args.data_path,
                                          feat_folder=data_args.feat_folder,
                                          tokenizer=tokenizer)
    eval_dataset = LazySupervisedDataset(data_path=data_args.val_data_path,
                                        feat_folder=data_args.val_feat_folder,
                                        tokenizer=tokenizer) if data_args.val_data_path else None
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)

