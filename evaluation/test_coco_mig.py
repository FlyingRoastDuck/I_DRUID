import torch
import os 
from PIL import Image
from ldm.modules.druid.druid_sd3_layers import AdapterLayoutSD3Transformer2DModel
from ldm.modules.druid.druid_sd3_pipeline import DRUIDSD3Pipeline
from tqdm import tqdm
import argparse

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--lora_path", type=str)
    parser.add_argument("--save_root", type=str)
    args = parser.parse_args()

    bench_name = args.save_root
    path_name = f'{bench_name}/images'
    anno_name = os.path.join(f'{bench_name}/anno')

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    # DRUID infer
    transformer = AdapterLayoutSD3Transformer2DModel(
        sample_size=128, patch_size=2, in_channels=16, num_layers=24, attention_head_dim=64, 
        num_attention_heads=24, joint_attention_dim=4096, caption_projection_dim=1536, pooled_projection_dim=2048,
        out_channels=16, pos_embed_max_size=192, attention_type="layout", max_boxes_per_image=10, 
    )
    transformer = transformer.from_pretrained(args.lora_path)

    pipe = DRUIDSD3Pipeline.from_pretrained(args.model_path, transformer=transformer, torch_dtype=torch.bfloat16)
    print(f"{args.lora_path} loaded")
    pipe = pipe.to(device)

    seed = 42
    batch_size = 1
    num_inference_steps = 50
    guidance_scale = 7
    height = 1024
    width = 1024
    prompt_per_input = 1

    import yaml
    bench_file_path = 'bench_file/mig_bench.txt'
    annotation_path = 'bench_file/mig_bench_anno.yaml'
    with open(annotation_path, 'r') as f:
        cfg = f.read()
        annatation_data = yaml.load(cfg, Loader=yaml.FullLoader)
    num_iter = 1

    os.makedirs(path_name, exist_ok=True)
    os.makedirs(anno_name, exist_ok=True) 


    with open(bench_file_path, 'r') as f:
        lines = f.readlines()

    prompt_len = len(lines)
    start_pid = rank*prompt_len//world_size
    end_pid = (1+rank)*prompt_len//world_size
    # print(f"infering: {start_pid} to {end_pid}")
    lines = lines[start_pid:end_pid]

    for i, prompt_line in enumerate(tqdm(lines)):
        region_bboxes_list = []
        prompt = prompt_line.split('\n')[0]

        img_prompt = prompt 
        instance_prompts = [] 
        global_caption = [prompt]
        region_caption_list = []
        if prompt in annatation_data:
            for phase in annatation_data[prompt]:
                if phase == 'coco_id':
                    continue
                bbox_list = annatation_data[prompt][phase]
                for _ in range(len(bbox_list)):
                    img_prompt += ',' + phase
                    instance_prompts.append(phase)
                for bbox in bbox_list:
                    region_bboxes_list.append(bbox)
                    region_caption_list.append(phase)
        coco_id = annatation_data[prompt]['coco_id']
        # spatial_mask = None
        spatial_mask = gen_spa_mask([region_bboxes_list], (2 if guidance_scale>0 else 1, 10, 64, 64)).to(device)

        for idx in range(prompt_per_input):

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                images = pipe(
                    prompt = global_caption*batch_size,
                    generator = torch.Generator(device=device).manual_seed(seed),
                    num_inference_steps = num_inference_steps, guidance_scale = guidance_scale, 
                    bbox_phrases = [region_caption_list], 
                    bbox_raw = [region_bboxes_list], height = height, width = width,
                    spatial_mask=spatial_mask
                )
            images=images.images
            filename = f"{coco_id}_{idx}.jpg"
            anno_file_name = f"anno_{coco_id}_{idx}_{bench_name}.jpg"
            
            for j, image in enumerate(images):   
                image.save(os.path.join(path_name, filename))

                bbox_image = pipe.draw_box_desc(
                    image, region_bboxes_list, region_caption_list, format="x0y0x1y1"
                )
                image.save(os.path.join(anno_name, anno_file_name))
