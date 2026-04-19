import numpy as np
import torch
from groundingdino.util.inference import load_image, predict
from huggingface_hub import hf_hub_download
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from torchvision.ops import box_convert


def load_GDINO():
    repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swint_ogc.pth"
    ckpt_config_filename = "GroundingDINO_SwinT_OGC.cfg.py"
    device='cuda'

    cache_config_file = hf_hub_download(
        repo_id=repo_id, filename=ckpt_config_filename)
    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device
    cache_file = hf_hub_download(repo_id=repo_id, filename=ckpt_filenmae)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(
        checkpoint['model']), strict=False)
    print("Reward Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model


def get_reward(dino_model, image, caption, cap_bbox, BOX_THRESHOLD=.35, WH=512):
    # iter through each glob prt and local prt
    reward = []
    for idx, (cur_cap, cur_box, cur_img) in enumerate(zip(caption, cap_bbox, image)):
        cur_reward = 0
        for ii in range(len(cur_box)):
            ins_cap, ins_box = cur_cap[ii], cur_box[ii]
            import ipdb;ipdb.set_trace()
            # using dino to predict bbox and confi
            boxes, logits, phrases = predict(
                model=dino_model, image=cur_img, caption=ins_cap, 
                box_threshold=BOX_THRESHOLD, text_threshold=.25
            )
            bbox = boxes[0]*WH
            boxes_xyxy = box_convert(
                boxes=bbox, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            confi = logits[0].item()
            iou_scores = IoU(ins_box, boxes_xyxy)
            cur_reward += confi + iou_scores

    return reward


def IoU(box, boxes):
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    inter = w * h
    iou = inter / (box_area + area - inter)
    return iou
