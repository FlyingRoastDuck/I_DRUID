
import torch
import os 
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.bbox_visualization import bbox_visualization,scale_boxes
from PIL import Image

from ldm.modules.druid.druid_sd3_layers import AdapterLayoutSD3Transformer2DModel
from ldm.modules.druid.druid_sd3_pipeline import DRUIDSD3Pipeline

from datasets import load_dataset

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import ast
import numpy as np

def adjust_and_normalize_bboxes(bboxes, orig_width, orig_height):
    normalized_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        x1_norm = round(x1 / orig_width,3)  
        y1_norm = round(y1 / orig_height,3)
        x2_norm = round(x2 / orig_width,3)
        y2_norm = round(y2 / orig_height,3)
        normalized_bboxes.append([x1_norm, y1_norm, x2_norm, y2_norm])
    
    return normalized_bboxes

class BboxDataset(Dataset):
    def __init__(self, dataset, resolution=1024):
        self.dataset = dataset #.select(range(10))
        self.resolution = resolution
        self.transform = transforms.Compose([
            transforms.Resize(
                (resolution,resolution), interpolation=transforms.InterpolationMode.BILINEAR 
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        image = self.transform(image)
        height = int(item['height'])
        width = int(item['width'])
        global_caption = item['global_caption']
        region_bboxes_list = item['bbox_list']
        detail_region_caption_list = item['detail_region_captions']
        region_caption_list = item['region_captions']
        file_name = item['file_name']

        region_bboxes_list = ast.literal_eval(region_bboxes_list)
        region_bboxes_list = adjust_and_normalize_bboxes(region_bboxes_list,width,height)
        region_bboxes_list = np.array(region_bboxes_list, dtype=np.float32)
    
        region_caption_list = ast.literal_eval(region_caption_list)
        detail_region_caption_list = ast.literal_eval(detail_region_caption_list)
        
        return {
            'image': image,
            'global_caption': global_caption,
            'detail_region_caption_list': detail_region_caption_list,
            'region_bboxes_list': region_bboxes_list,
            'region_caption_list': region_caption_list,
            'file_name': file_name,
            'height': height,
            'width': width
        }


def gen_spa_mask(bbox, lat_shape, max_box=10):
    B, _, W, H = lat_shape
    mask = torch.zeros((B, max_box, H, W))
    
    for idx, cur_id_box in enumerate(bbox):
        for idx_b, cur_bbox in enumerate(cur_id_box):
            if sum(cur_bbox)==0: continue
            x0, y0, x1, y1 = cur_bbox
            cur_mask = torch.zeros(H, W)
            cur_mask[int(y0*H):int(y1*H), int(x0*W):int(x1*W)] = 1
            mask[idx, idx_b, :, :] = cur_mask
    return mask

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--lora_path", type=str)
    parser.add_argument("--save_root", type=str)
    args = parser.parse_args()

    # DRUID infer
    transformer = AdapterLayoutSD3Transformer2DModel(
        sample_size=128, patch_size=2, in_channels=16, num_layers=24, attention_head_dim=64, 
        num_attention_heads=24, joint_attention_dim=4096, caption_projection_dim=1536, pooled_projection_dim=2048,
        out_channels=16, pos_embed_max_size=192, attention_type="layout", max_boxes_per_image=10, 
    )
    
    transformer = transformer.from_pretrained(args.lora_path)
    print(f"Lora {args.lora_path} Loaded")
    pipe = DRUIDSD3Pipeline.from_pretrained(args.model_path, transformer=transformer, torch_dtype=torch.bfloat16)


    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    pipe = pipe.to(device)
    test_dataset = load_dataset(args.dataset_path, split='test')
    prompt_len = len(test_dataset)

    start_pid=rank*prompt_len//world_size
    end_pid=(1+rank)*prompt_len//world_size
    print(f"infering: {start_pid} to {end_pid}")
    test_dataset = test_dataset.select(range(start_pid, end_pid))

    test_dataset = BboxDataset(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    seed = 42
    batch_size = 1
    num_inference_steps = 50
    height = 1024
    width = 1024
    guidance_scale = 7.5

    
    img_save_root = os.path.join(args.save_root,"images")
    os.makedirs(img_save_root,exist_ok=True)
    img_with_layout_save_root = os.path.join(args.save_root,"images_with_layout")
    os.makedirs(img_with_layout_save_root,exist_ok=True)

    #generation
    for i, batch in enumerate(tqdm(test_dataloader)):
        global_caption = batch["global_caption"]
        region_caption_list = [t[0] for t in batch["detail_region_caption_list"]]
        region_bboxes_list = batch["region_bboxes_list"][0].numpy().tolist()
        filename = batch["file_name"][0]
        # spatial_mask = None
        spatial_mask = gen_spa_mask([region_bboxes_list], (2 if guidance_scale>0 else 1, 10, 64, 64)).to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            images = pipe(
                prompt = global_caption*batch_size, generator = torch.Generator(device=device).manual_seed(seed),
                num_inference_steps = num_inference_steps, guidance_scale = guidance_scale, 
                bbox_phrases = [region_caption_list], bbox_raw = [region_bboxes_list], 
                height = height, width = width, spatial_mask=spatial_mask
            )

        image=images.images[0]

        image.save(os.path.join(img_save_root,filename)) 

        img_with_layout_save_name=os.path.join(img_with_layout_save_root,filename)

        white_image = Image.new('RGB', (width, height), color='rgb(256,256,256)')
        show_input = {"boxes":scale_boxes(region_bboxes_list,width,height),"labels":region_caption_list}

        bbox_visualization_img = bbox_visualization(white_image,show_input)
        image_with_bbox = bbox_visualization(image ,show_input)

        total_width = width*2
        total_height = height

        new_image = Image.new('RGB', (total_width, total_height))
        new_image.paste(bbox_visualization_img, (0, 0))
        new_image.paste(image_with_bbox, (width, 0))
        new_image.save(img_with_layout_save_name)