

LAYOUTSAM_EVAL_PATH="./hf_datasets/layoutsam_eval/data"
MINICPM="./hf_models/MiniCPM-V-2_6"
SD3_PATH="./hf_models/sd3_mid"
lora_path="./hf_models/I_DRUID"

SAVE_COCO_ROOT="./outputs/DRUID_LayoutRL"

# eva on COCO-MIG
torchrun -m --nproc_per_node=8 --master_port=10046 evaluation.test_coco_mig --save_root ${SAVE_COCO_ROOT} --model_path ${SD3_PATH} --lora_path ${lora_path}


SAVE_ROOT="./outputs/DRUID_SAM_AdapterRL"

# eva on LayoutSAM-eval
torchrun -m --nproc_per_node=8 --master_port=1046 evaluation.test_druid_layoutsam_benchmark --model_path ${SD3_PATH} --dataset_path ${LAYOUTSAM_EVAL_PATH} --lora_path ${lora_path} --save_root ${SAVE_ROOT} 
torchrun -m --nproc_per_node=8 --master_port=1046 evaluation.score_layoutsam_multiGPU --dataset_path ${LAYOUTSAM_EVAL_PATH} --model_id ${MINICPM} --generate_path ${SAVE_ROOT}/images
