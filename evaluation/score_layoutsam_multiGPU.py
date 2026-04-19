import os
import json
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import ast
import numpy as np
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import argparse
from datasets import load_dataset


def adjust_and_normalize_bboxes(bboxes, orig_width, orig_height):
    normalized_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        x1_norm = round(x1 / orig_width, 3)
        y1_norm = round(y1 / orig_height, 3)
        x2_norm = round(x2 / orig_width, 3)
        y2_norm = round(y2 / orig_height, 3)
        normalized_bboxes.append([x1_norm, y1_norm, x2_norm, y2_norm])
    return normalized_bboxes


class BboxDataset(Dataset):
    def __init__(self, dataset, resolution=1024):
        self.dataset = dataset
        self.resolution = resolution
        self.transform = transforms.Compose([
            transforms.Resize(
                (resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR
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
        region_bboxes_list = adjust_and_normalize_bboxes(region_bboxes_list, width, height)
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


def setup_distributed():
    """初始化分布式环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://',
                           world_size=world_size, rank=rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def gather_results_to_rank0(results_dict, world_size):
    """收集所有进程的结果到 rank 0"""
    all_results = [None] * world_size
    
    # 每个进程将自己的结果序列化
    if results_dict:
        json_str = json.dumps(results_dict)
        json_bytes = json_str.encode('utf-8')
    else:
        json_bytes = b''
    
    # 收集每个进程的数据长度
    local_size = torch.tensor([len(json_bytes)], dtype=torch.long).cuda()
    size_list = [torch.zeros(1, dtype=torch.long).cuda() for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    max_size = max(s.item() for s in size_list)
    
    if max_size == 0:
        return all_results
    
    # padding 以便 gather
    if len(json_bytes) < max_size:
        json_bytes = json_bytes + b'\x00' * (max_size - len(json_bytes))
    
    tensor_list = [torch.zeros(max_size, dtype=torch.uint8).cuda() for _ in range(world_size)]
    local_tensor = torch.from_numpy(np.frombuffer(json_bytes, dtype=np.uint8)).cuda()
    dist.all_gather(tensor_list, local_tensor)
    
    # 解码
    for i, tensor in enumerate(tensor_list):
        size = size_list[i].item()
        if size > 0:
            json_str = tensor[:size].cpu().numpy().tobytes().decode('utf-8')
            all_results[i] = json.loads(json_str)
    
    return all_results


def process_batch(args, model, tokenizer, batch, generate_path, temp_root, resolution):
    """处理单个 batch"""
    global_caption = batch["global_caption"]
    detial_region_caption_list = [t[0] for t in batch["detail_region_caption_list"]]
    region_caption_list = [t[0] for t in batch["region_caption_list"]]
    region_bboxes_list = batch["region_bboxes_list"][0]
    filename = batch["file_name"][0]

    generated_img = os.path.join(generate_path, filename)
    temp_save_root = os.path.join(temp_root, filename.replace('.jpg', ''))
    os.makedirs(temp_save_root, exist_ok=True)

    bbox_count = len(region_caption_list)

    # Initialize scores
    img_score_spatial = 0
    img_score_color = 0
    img_score_texture = 0
    img_score_shape = 0

    for i, (bbox, detial_region_caption, region_caption) in enumerate(
        zip(region_bboxes_list, detial_region_caption_list, region_caption_list)
    ):
        x1, y1, x2, y2 = bbox
        # 缩放到 0~1
        x1 = int(x1 * resolution)
        y1 = int(y1 * resolution)
        x2 = int(x2 * resolution)
        y2 = int(y2 * resolution)

        img = Image.open(generated_img)
        cropped_img = img.crop((x1, y1, x2, y2))

        # save crop img
        description = region_caption.replace('/', '')
        detail_description = detial_region_caption.replace('/', '')
        cropped_img_path = os.path.join(temp_save_root, f'{description}.jpg')
        cropped_img.save(cropped_img_path)

        # spatial
        question = f'Is the subject "{description}" present in the image? Strictly answer with "Yes" or "No", without any irrelevant words.'

        msgs = [{'role': 'user', 'content': [cropped_img, question]}]

        res = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
            seed=42
        )

        if "Yes" in res or "yes" in res:
            score_spatial = 1.0
        else:
            score_spatial = 0.0

        score_color, score_texture, score_shape = 0.0, 0.0, 0.0
        # attribute
        if score_spatial == 1.0:
            # color
            question_color = f'Is the subject in "{description}" in the image consistent with the color described in the detailed description: "{detail_description}"? Strictly answer with "Yes" or "No", without any irrelevant words. If the color is not mentioned in the detailed description, the answer is "Yes".'
            msgs_color = [{'role': 'user', 'content': [cropped_img, question_color]}]

            color_attribute = model.chat(
                image=None,
                msgs=msgs_color,
                tokenizer=tokenizer,
                seed=42
            )

            if "Yes" in color_attribute or "yes" in color_attribute:
                score_color = 1.0
        # texture
        if score_spatial == 1.0:
            question_texture = f'Is the subject in "{description}" in the image consistent with the texture described in the detailed description: "{detail_description}"? Strictly answer with "Yes" or "No", without any irrelevant words. If the texture is not mentioned in the detailed description, the answer is "Yes".'
            msgs_texture = [{'role': 'user', 'content': [cropped_img, question_texture]}]

            texture_attribute = model.chat(
                image=None,
                msgs=msgs_texture,
                tokenizer=tokenizer,
                seed=42
            )
            if "Yes" in texture_attribute or "yes" in texture_attribute:
                score_texture = 1.0
        # shape
        if score_spatial == 1.0:
            question_shape = f'Is the subject in "{description}" in the image consistent with the shape described in the detailed description: "{detail_description}"? Strictly answer with "Yes" or "No", without any irrelevant words. If the shape is not mentioned in the detailed description, the answer is "Yes".'
            msgs_shape = [{'role': 'user', 'content': [cropped_img, question_shape]}]

            shape_attribute = model.chat(
                image=None, msgs=msgs_shape, tokenizer=tokenizer, seed=42
            )

            if "Yes" in shape_attribute or "yes" in shape_attribute:
                score_shape = 1.0

        # Update total scores
        img_score_spatial += score_spatial
        img_score_color += score_color
        img_score_texture += score_texture
        img_score_shape += score_shape

    # Store image stats
    return os.path.basename(filename), {
        "bbox_count": bbox_count,
        "score_spatial": img_score_spatial,
        "score_color": img_score_color,
        "score_texture": img_score_texture,
        "score_shape": img_score_shape,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--generate_path', type=str, default="outputs/DRUID_SAM_AdapterRL/images")
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    # 初始化分布式环境
    rank, world_size, local_rank = setup_distributed()
    is_main_process = (rank == 0)

    # 加载模型（每个进程都会加载自己的模型副本）
    model_id = args.model_id
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True,
        attn_implementation='sdpa', torch_dtype=torch.bfloat16)
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # 加载数据集
    dataset_path = args.dataset_path
    test_dataset = load_dataset(dataset_path, split='test')
    test_dataset = BboxDataset(test_dataset, resolution=args.resolution)

    # 使用 DistributedSampler
    sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        sampler=sampler
    )

    generate_path = args.generate_path
    resolution = args.resolution

    if is_main_process:
        print(f"processing: {generate_path}")
        print(f"World size: {world_size}")

    save_json_path = generate_path.replace("images", "minicpm-vqa.json")
    temp_root = generate_path.replace("images", "images-perarea")
    os.makedirs(temp_root, exist_ok=True)

    # 每个进程独立处理分配到的数据
    local_image_stats = {}

    # 使用 tqdm 进度条
    if is_main_process:
        pbar = tqdm(total=len(test_dataloader), desc="Processing")
    
    for i, batch in enumerate(test_dataloader):
        filename, stats = process_batch(
            args, model, tokenizer, batch, 
            generate_path, temp_root, resolution
        )
        local_image_stats[filename] = stats
        
        if is_main_process:
            pbar.update(1)
        
        # 同步所有进程，确保进度一致
        dist.barrier()

    if is_main_process:
        pbar.close()

    # 等待所有进程完成处理
    dist.barrier()

    # 收集所有进程的结果到 rank 0
    all_results = gather_results_to_rank0(local_image_stats, world_size)

    # 只在 rank 0 上写入文件
    if is_main_process:
        # 合并所有进程的结果
        image_stats = {}
        for result in all_results:
            if result is not None:
                image_stats.update(result)

        # 保存 JSON 文件
        with open(save_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(image_stats, json_file, indent=4)

        print(f"Image statistics saved to {save_json_path}")

        # 计算总分
        score_save_path = save_json_path.replace('minicpm-vqa.json', 'minicpm-vqa-score.txt')

        total_num = 0
        total_bbox_num = 0
        total_score_spatial = 0
        total_score_color = 0
        total_score_texture = 0
        total_score_shape = 0

        miss_match = 0
        for key, value in image_stats.items():
            total_num += value["bbox_count"]
            total_score_spatial += value["score_spatial"]
            total_score_color += value["score_color"]
            total_score_texture += value["score_texture"]
            total_score_shape += value["score_shape"]

            if (value["bbox_count"] != value["score_spatial"] or 
                value["bbox_count"] != value["score_color"] or 
                value["bbox_count"] != value["score_texture"] or 
                value["bbox_count"] != value["score_shape"]):
                print(key, value["bbox_count"], value["score_spatial"], 
                      value["score_color"], value["score_texture"], value["score_shape"])
                miss_match += 1

        print(miss_match)

        # 保存分数
        with open(score_save_path, "w") as f:
            f.write(f"Total number of bbox: {total_num}\n")
            f.write(f"Total score of spatial: {total_score_spatial}; Average score of spatial: {round(total_score_spatial/total_num, 4)}\n")
            f.write(f"Total score of color: {total_score_color}; Average score of color: {round(total_score_color/total_num, 4)}\n")
            f.write(f"Total score of texture: {total_score_texture}; Average score of texture: {round(total_score_texture/total_num, 4)}\n")
            f.write(f"Total score of shape: {total_score_shape}; Average score of shape: {round(total_score_shape/total_num, 4)}\n")

    # 清理分布式环境
    cleanup_distributed()


if __name__ == "__main__":
    main()
