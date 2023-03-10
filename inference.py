import os
import yaml  # type: ignore
import argparse

import numpy as np
import torch

from spadesr import Discriminator, Generator, InverseEncoder

TEST_CFG = {
    "filename": "SPADE_SR_64_G400_I200",
    "G_Z_DIM": 128,
    "G_CONV_CH": 128,
    "G_CONV_THERMAL_CH": 64,
    'E_CONV_CH': 64,
    'E_USE_GSP': False,  # Global Sum Pooling,
    "IMG_SHAPE_X": 64,
    "IMG_SHAPE_Y": 40,
    "THERMAL_PATH": "data/inference/thermals.npy",
    "RGB_PATH": "data/inference/rgbs.npy",
    "RELU_NEG_SLOPE": 0.01,
    "BATCH_SIZE": 64,
    'G_PTH': 'experiments/SPADE_SR_64_G400_I200/generator.pth',
    'I_Pth': 'experiments/SPADE_SR_64_G400_I200/inverter.pth',
}

# GPU Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    # Read config
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default="DEFAULT", help="Path to cfg file")
    args = parser.parse_args()
    if args.cfg_path == "DEFAULT":
        print("Using default config")
        cfg = TEST_CFG
    else:
        with open(args.cfg_path, "r") as stream:
            cfg = yaml.safe_load(stream)

    # Create log folders
    folder_name = os.path.join('outputs', cfg['filename'])
    os.makedirs(folder_name, exist_ok=True)

    # Save config
    with open(os.path.join(folder_name, 'cfg.yaml'), 'a') as file:
        yaml.dump(cfg, file, sort_keys=False)

    # Load models
    generator = Generator(cfg).to(device)
    inverter = InverseEncoder(cfg).to(device)

    # generator.load_state_dict(torch.load(cfg['G_Pth'], map_location=device))
    # inverter.load_state_dict(torch.load(cfg['I_Pth'], map_location=device))

    generator.eval()
    inverter.eval()

    # Load data
    thermals = np.load(cfg['THERMAL_PATH']).transpose(0, 3, 1, 2)
    rgbs = np.load(cfg['RGB_PATH']).transpose(0, 3, 1, 2)

    # Inference
    for i in range(0, len(thermals), cfg['BATCH_SIZE']):
        thermal_batch = torch.tensor(thermals[i:i + cfg['BATCH_SIZE']]).to(device)
        rgb_batch = torch.tensor(rgbs[i:i + cfg['BATCH_SIZE']]).to(device)

        with torch.no_grad():
            zs = inverter(rgb_batch)
            gen_rgbs = generator(thermal_batch, zs)

        for j in range(len(gen_rgbs)):
            gen_rgb = gen_rgbs[j].cpu().numpy().transpose(1, 2, 0)

            # Save the image using torchvision.utils.save_image
            import matplotlib.pyplot as plt
            plt.imsave(os.path.join(folder_name, f"{i + j}.png"), gen_rgb / 2 + 0.5)
