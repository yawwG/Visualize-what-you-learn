import os
import torch
import numpy as np
import copy
import random
import pandas as pd
import segmentation_models_pytorch as smp
from . import builder
from . import utils
from . import constants
from .models.vision_model import PretrainedImageClassifier
from typing import Union, List

np.random.seed(6)
random.seed(6)
_MODELS = {
    "VSWL_resnet50": "./pretrained/resnet50.ckpt",
    "VSWL1_resnet50": "./pretrained/resnet50.ckpt",
    "VSWL2_resnet50": "./pretrained/resnet50.ckpt",
}


_SEGMENTATION_MODELS = {
    "VSWL_resnet50": "./pretrained/resnet50.ckpt",
    "VSWL1_resnet50": "./pretrained/resnet50.ckpt",
    "VSWL2_resnet50": "./pretrained/resnet50.ckpt",
}


_FEATURE_DIM = {"VSWL_resnet50": 2048, "VSWL1_resnet50": 2048,"VSWL2_resnet50": 2048}


def available_models() -> List[str]:
    """Returns the names of available VSWL models"""
    return list(_MODELS.keys())


def available_segmentation_models() -> List[str]:
    """Returns the names of available VSWL models"""
    return list(_SEGMENTATION_MODELS.keys())


def load_VSWL(
    name: str = "VSWL_resnet50",
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Load a VSWL model

    Parameters
    ----------
    name : str
        A model name listed by `VSWL2.available_models()`, or the path to a model checkpoint containing the state_dict
    device : Union[str, torch.device]
        The device to put the loaded model

    Returns
    -------
    VSWL_model : torch.nn.Module
        The VSWL model
    """

    # warnings
    if name in _MODELS:
        ckpt_path = _MODELS[name]
    elif os.path.isfile(name):
        ckpt_path = name
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}"
        )

    ckpt = torch.load(ckpt_path)
    # ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    cfg = ckpt["hyper_parameters"]
    ckpt_dict = ckpt["state_dict"]

    fixed_ckpt_dict = {}
    for k, v in ckpt_dict.items():
        new_key = k.split("VSWL2.")[-1]
        fixed_ckpt_dict[new_key] = v
    ckpt_dict = fixed_ckpt_dict

    VSWL_model = builder.build_VSWL_model(cfg).to(device)
    VSWL_model.load_state_dict(ckpt_dict)

    return VSWL_model


def load_img_classification_model(
    name: str = "VSWL_resnet50",
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    num_cls: int = 1,
    freeze_encoder: bool = True,
):
    """Load a VSWL pretrained classification model

    Parameters
    ----------
    name : str
        A model name listed by `VSWL2.available_models()`, or the path to a model checkpoint containing the state_dict
    device : Union[str, torch.device]
        The device to put the loaded model
    num_cls: int
        Number of output classes
    freeze_encoder: bool
        Freeze the pretrained image encoder

    Returns
    -------
    img_model : torch.nn.Module
        The VSWL pretrained image classification model
    """

    # load pretrained image encoder
    VSWL_model = load_VSWL(name, device)
    image_encoder = copy.deepcopy(VSWL_model.img_encoder)
    del VSWL_model

    # create image classifier
    feature_dim = _FEATURE_DIM[name]
    img_model = PretrainedImageClassifier(
        image_encoder, num_cls, feature_dim, freeze_encoder
    )

    return img_model
    # return VSWL_model

def load_img_segmentation_model(
    name: str = "VSWL_resnet50",
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Load a VSWL pretrained classification model

    Parameters
    ----------
    name : str
        A model name listed by `VSWL2.available_models()`, or the path to a model checkpoint containing the state_dict
    device : Union[str, torch.device]
        The device to put the loaded model

    Returns
    -------
    img_model : torch.nn.Module
        The VSWL pretrained image classification model
    """

    # warnings
    if name in _SEGMENTATION_MODELS:
        ckpt_path = _SEGMENTATION_MODELS[name]
        base_model = name.split("_")[-1]
    elif os.path.isfile(name):
        ckpt_path = name
        base_model = "resnet50"  # TODO
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_segmentation_models()}"
        )

    # load base model
    seg_model = smp.Unet(base_model, encoder_weights=None, activation=None)

    # update weight
    ckpt = torch.load(ckpt_path)
    ckpt_dict = {}
    for k, v in ckpt["state_dict"].items():
        if k.startswith("VSWL2.img_encoder.model"):
            k = ".".join(k.split(".")[3:])
            ckpt_dict[k] = v
        ckpt_dict["fc.bias"] = None
        ckpt_dict["fc.weight"] = None
    # seg_dict = seg_model.encoder.state_dict()
    seg_model.encoder.load_state_dict(ckpt_dict)

    return seg_model.to(device)

def get_similarities(VSWL_model, imgs, txts, similarity_type="both"):
    """Load a VSWL pretrained classification model

    Parameters
    ----------
    VSWL_model : str
        VSWL model, load via VSWL2.load_models()
    imgs:
        processed images using VSWL_model.process_img
    txts:
        processed text using VSWL_model.process_text
    similartiy_type
        Either local, global or both

    Returns
    -------
    similarities :
        similartitie between each imgs and text
    """

    # warnings
    if similarity_type not in ["global", "local", "both"]:
        raise RuntimeError(
            f"similarity type should be one of ['global', 'local', 'both']"
        )
    if type(txts) == str or type(txts) == list:
        raise RuntimeError(
            f"Text input not processed - please use VSWL_model.process_text"
        )
    if type(imgs) == str or type(imgs) == list:
        raise RuntimeError(
            f"Image input not processed - please use VSWL_model.process_img"
        )

    # get global and local image features
    with torch.no_grad():
        img_emb_l, img_emb_g,  x_, e1, e2, e3 = VSWL_model.image_encoder_forward2(imgs)
        text_emb_l, text_emb_g, sents,text_mlp1, text_mlp2 = VSWL_model.text_encoder_forward(
            txts["caption_ids"], txts["attention_mask"], txts["token_type_ids"]
        )

    # get similarities
    global_similarities = VSWL_model.get_global_similarities(img_emb_g, text_emb_g)
    local_similarities, attn_maps = VSWL_model.get_local_similarities(
        img_emb_l, text_emb_l, txts["cap_lens"]
    )

#plot
    # for i in range(len(attn_maps)):
    #     attenmap_tmp =[]
    #     imgs_tmp = []
    #
    #     # words_num =  txts["cap_lens"][i]
    #     word = sents[i]
    #
    #     # word = [word] * attn_maps[i].size(0)#atten[8,1,10,32,32],sent[8,97]
    #
    #
    #     # print(word.size)
    #     # word = torch.from_numpy(word)
    #     # word = torch.stack(word)
    #      # [1, 768, 25]
    #     # word = word.repeat(attn_maps[i].size(0), 1, 1)
    #     attenmap_tmp.append(attn_maps[i][0:3, :, :, :])
    #     # attenmap_tmp.append(attn_maps[0:3, :, :, :])
    #     attenmap_tmp.append(attn_maps[i][10:13, :, :, :])
    #     attenmap_tmp.append(attn_maps[i][20:23, :, :, :])
    #     attenmap_tmp = torch.cat((attenmap_tmp[0],attenmap_tmp[1],attenmap_tmp[2]),0)
    #     # imgs_tmp.append(imgs[0:3, :, :, :])
    #     # imgs_tmp.append(imgs[10:13, :, :, :])
    #     # imgs_tmp.append(imgs[20:23, :, :, :])
    #     imgs_tmp = torch.cat((imgs[0:3, :, :, :],imgs[10:13, :, :, :],imgs[20:23, :, :, :]),0)
    #     word = [word] * attenmap_tmp.size(0)  # atten[8,1,10,32,32],sent[8,97]
    #     attenmap_tmp = torch.unsqueeze(attenmap_tmp,1)
    #
    #     VSWL_model.plot_attn_maps(attenmap_tmp, imgs_tmp.cpu(), word, "zeroshot_"+str(i)+str(word[0][0]))

    similarities = (local_similarities + global_similarities) / 2

    if similarity_type == "global":
        return global_similarities.detach().cpu().numpy()
    elif similarity_type == "local":
        return local_similarities.detach().cpu().numpy()
    else:
        return similarities.detach().cpu().numpy()

def zero_shot_classification(VSWL_model, imgs, cls_txt_mapping):
    """Load a VSWL pretrained classification model

    Parameters
    ----------
    VSWL_model : str
        VSWL model, load via VSWL2.load_models()
    imgs:
        processed images using VSWL_model.process_img
    cls_txt_mapping:
        dictionary of class to processed text mapping. Each class can have more than one associated text

    Returns
    -------
    cls_similarities :
        similartitie between each imgs and text
    """

    # get similarities for each class
    class_similarities = []
    for cls_name, cls_txt in cls_txt_mapping.items():
        similarities = get_similarities(
            VSWL_model, imgs, cls_txt, similarity_type="both"
        )
        cls_similarity = similarities.max(axis=1)  # average between class prompts
        class_similarities.append(cls_similarity)
    class_similarities = np.stack(class_similarities, axis=1)

    # normalize across class
    if class_similarities.shape[0] > 1:
        class_similarities = utils.normalize(class_similarities)
    class_similarities = pd.DataFrame(
        class_similarities, columns=cls_txt_mapping.keys()
    )

    return class_similarities

def generate_INB_class_prompts(n: int = 3):
    """Generate text prompts for each classification task

    Parameters
    ----------
    n:  int
        number of prompts per class

    Returns
    -------
    class prompts : dict
        dictionary of class to prompts
    """

    prompts = {}
    for k, v in constants.INB_CLASS_PROMPTS.items():
        cls_prompts = []
        keys = list(v.keys())

        # severity
        for k0 in v[keys[0]]:
            # subtype
            for k1 in v[keys[1]]:
                # location
                for k2 in v[keys[2]]:
                    cls_prompts.append(f"{k0} {k1} {k2}")

        prompts[k] = random.sample(cls_prompts, n)
    return prompts