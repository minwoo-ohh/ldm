import os
import glob
import argparse
import numpy as np
from PIL import Image
import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def load_model(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    model = instantiate_from_config(config.model)
    model.load_state_dict(pl_sd["state_dict"], strict=False)
    model.cuda()
    model.eval()
    return model


def prepare_mask(path, device):
    m = Image.open(path).convert("L")
    m = np.array(m).astype(np.float32) / 255.0
    m = torch.from_numpy(m)[None, None]
    m = m.to(device) * 2.0 - 1.0
    return m


def main(opt):
    config = OmegaConf.load(opt.config)
    model = load_model(config, opt.ckpt)
    sampler = DDIMSampler(model)
    os.makedirs(opt.outdir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(opt.indir, "*")))
    for path in files:
        cond = prepare_mask(path, model.device)
        h, w = cond.shape[-2:]
        shape = (model.first_stage_model.embed_dim, h // 8, w // 8)
        samples, _ = sampler.sample(S=opt.steps, conditioning=cond,
                                    batch_size=1, shape=shape, eta=opt.ddim_eta, verbose=False)
        x = model.decode_first_stage(samples)
        x = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
        x = 255. * x.cpu().permute(0, 2, 3, 1).numpy()[0]
        Image.fromarray(x.astype(np.uint8)).save(os.path.join(opt.outdir, os.path.basename(path)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="models/ldm/shoeprint_ldm/config.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--indir", type=str, required=True, help="directory with sole masks")
    parser.add_argument("--outdir", type=str, default="outputs/sole2footprint")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    opt = parser.parse_args()
    main(opt)
