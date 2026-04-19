IMG_RL_DIR="{Your MIGC infer images here}" 

CUDA_VISIBLE_DEVICES=1 python -m evaluation.eval_mig --need_miou_score\
 --need_instance_sucess_ratio --metric_name 'eval'\
 --image_dir ${IMG_RL_DIR} --need_clip_score --need_local_clip 
