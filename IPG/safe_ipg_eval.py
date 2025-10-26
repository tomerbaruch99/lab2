# safe_ipg_eval_verbose.py
import torch
import os
from basicsr.models import build_model
from basicsr.data import build_dataloader, build_dataset
import yaml
from tqdm import tqdm
import cv2

# --- CONFIGURATION ---
config_path = "/home/student/project/IPG/options/test_IPG_BasicSR_x2.yml"
weights_path = "/home/student/project/weights/IPG_SRx2.pth"

VALTILE = 32
VALTILE_PAD = 8
TOP_K = 8
SAMPLE_SIZE = 2
SAMPLING_METHOD = 1

# --- LOAD CONFIG ---
with open(config_path, 'r') as f:
    opt = yaml.safe_load(f)

# Add missing keys for testing
opt['is_train'] = False   # important!
opt['datasets'] = opt.get('datasets', {})  # ensure datasets exist

# Update memory-safe parameters
opt['val']['tile'] = VALTILE
opt['val']['tile_pad'] = VALTILE_PAD
opt['network_gtop_k'] = TOP_K
opt['network_g_sample_size'] = SAMPLE_SIZE
opt['network_sampling_method'] = SAMPLING_METHOD
opt['val_use_amp'] = True

# --- BUILD MODEL ---
model = build_model(opt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.model_to_device(model.net_g)

# --- LOAD DATASETS ---
datasets = {}
loaders = {}
for phase, dataset_opt in opt['datasets'].items():
    datasets[phase] = build_dataset(dataset_opt)
    loaders[phase] = build_dataloader(datasets[phase], dataset_opt)

# --- OUTPUT DIRECTORY ---
output_dir = "/home/student/project/IPG/safe_outputs"
os.makedirs(output_dir, exist_ok=True)

# --- INFERENCE LOOP ---
for phase, dataloader in loaders.items():
    print(f"Processing dataset: {phase} ({len(dataloader)} images)")
    for i, data in enumerate(tqdm(dataloader, desc=phase)):
        img = data['lq'].to(device)
        with torch.no_grad():
            output = model.net_g(img)

        # Save image
        sr_img = output[0].cpu().permute(1,2,0).numpy()
        sr_img = (sr_img * 255).clip(0,255).astype('uint8')
        filename = os.path.join(output_dir, f"{phase}_{i+1}.png")
        cv2.imwrite(filename, sr_img)

        # Free GPU memory
        del img, output
        torch.cuda.empty_cache()

print("✅ All datasets processed safely, outputs saved in:", output_dir)