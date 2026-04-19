import numpy as np
import torch
from groundingdino.util.inference import Model
from groundingdino.util.inference import load_image, predict
from PIL import Image
from huggingface_hub import hf_hub_download
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from torchvision.ops import box_convert


def load_model(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, device='cuda'):
    cache_config_file = hf_hub_download(repo_id=ckpt_repo_id, filename=ckpt_config_filename)
    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device
    cache_file = hf_hub_download(repo_id=ckpt_repo_id, filename=ckpt_filenmae)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model    


def gen_dino_score(final_caps, bboxes, image, model, WH=512):
    BOX_THRESHOLD = 0.5
    reward_score = []
    for (gt_prts, gt_boxes) in zip(final_caps, bboxes):
        reward = 0
        for (cur_prt, cur_b, im) in zip(gt_prts, gt_boxes, image):
            if sum(cur_b)==0: continue
            boxes, logits, phrases = predict(
                model=model, image=im, caption=cur_prt, box_threshold=BOX_THRESHOLD, 
                text_threshold=.5
            )
            # cannot find
            if len(boxes)==0: continue
            bbox = boxes[0]*WH
            boxes_xyxy = box_convert(boxes=bbox, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            confi = logits[0].item()
            cur_b = [val*WH for val in cur_b]
            reward += confi
            reward += IoU(boxes_xyxy.tolist(), cur_b)
        reward_score.append(reward)
    return reward_score



def IoU(box, boxes):
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[2] - boxes[0] + 1) * (boxes[3] - boxes[1] + 1)
    xx1 = np.maximum(box[0], boxes[0])
    yy1 = np.maximum(box[1], boxes[1])
    xx2 = np.minimum(box[2], boxes[2])
    yy2 = np.minimum(box[3], boxes[3])
    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    inter = w * h
    iou = inter / (box_area + area - inter)
    return iou


