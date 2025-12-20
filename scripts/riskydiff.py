import importlib
import torch
import numpy as np
import PIL
from omegaconf import OmegaConf
from PIL import Image
from tqdm import trange
import os
import time
from torch import autocast
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from einops import rearrange, repeat
from pytorch_lightning import seed_everything
from contextlib import nullcontext
import argparse
import pickle
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
import misc
from datastuff import get_dataset_class


VERSION2SPECS = {
    "unCLIP-L": {"H": 768, "W": 768, "C": 4, "f": 8},
    "unOpenCLIP-H": {"H": 768, "W": 768, "C": 4, "f": 8},
}

device = "cuda" if torch.cuda.is_available() else "cpu"


lr=1e-3
num_epochs = 20
num_iters_vector = 10000
num_iters_vector_cpt = 1000
feat_dim = 1024
# CLAMP_VALUE = 20

parser = argparse.ArgumentParser()
parser.add_argument('--model_version', type=str, default="unOpenCLIP-H", choices=list(VERSION2SPECS.keys()))
parser.add_argument('--input_img', action="store_true")
parser.add_argument('--prompt', type=str, default="")
parser.add_argument('--outdir', type=str, default="outputs")
parser.add_argument('--neg_prompt', type=str, default="")
parser.add_argument('--cfg_scale', type=float, default=10.)
parser.add_argument('--nrow', type=int, default=2)
parser.add_argument('--ncol', type=int, default=2)
parser.add_argument('--steps', type=int, default=20)
parser.add_argument('--eta', type=float, default=0.)
parser.add_argument('--force_fp32', action="store_true")
parser.add_argument('--H', type=int, default=512)
parser.add_argument('--W', type=int, default=512)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--sampler', type=str, default="DDIM", choices=["DDIM", "DPM"])
parser.add_argument('--noise_level_input', type=float, default=0)
parser.add_argument('--noise_level_avg', type=float, default=50)
parser.add_argument('--num_inputs', type=int, default=1)    
parser.add_argument('--weight_input', type=float, default=1.)
parser.add_argument('--noise_embedding_mix', type=bool, default=True)
# risky region
parser.add_argument('--dataset', type=str, default="NICOpp")
parser.add_argument('--category', type=str, default="cat")
parser.add_argument('--pkldir', type=str, default="pkl")
parser.add_argument('--loss_k', type=int, default=50)
parser.add_argument('--gradient_scale', type=float, default=0)
parser.add_argument('--match_scale', type=float, default=0)
parser.add_argument('--std_scale', type=float, default=1)
parser.add_argument('--text_prompt_range', type=float, default=1.0, help="between 0 and 1, indicating the fraction of steps using text prompt as condition")
parser.add_argument('--gradient_range', type=float, default=1.0, help="between 0 and 1, indicating the fraction of steps using risky gradient")
parser.add_argument('--conf_threshold', type=float, default=0.5)
parser.add_argument('--risky_model_path', type=str, default="/home/hanyu/stuff/SubpopBench/train_output/debug_attrNo/NICOpp_ERM_hparams0_seed59/model.pkl")
parser.add_argument('--risky_model_arch', type=str, default="")
parser.add_argument('--no_filter', action="store_true")
parser.add_argument('--max_filter_time', type=float, default=120)
parser.add_argument('--save_model_eval', action="store_true")
parser.add_argument('--load_model_eval', action="store_true")
parser.add_argument('--holdout', action="store_true")
parser.add_argument('--num_candidates', type=int, default=5)
parser.add_argument('--drop_prompt', action="store_true")
parser.add_argument("--save_grid", action='store_true', help="Save images as grid")


args = parser.parse_args()

seed = args.seed
seed_everything(seed)


def check_range(value, inf, sup):
    if not inf <= value <= sup:
        print("Argument out of range!")
        raise ValueError
        

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def load_img(img_path):
    if img_path == "":
        return None
    image = Image.open(img_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h})")
    w, h = map(lambda x: x - x % 64, (w, h))
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.

def sample(
        model,
        prompt,
        n_runs=3,
        n_samples=2,
        H=512,
        W=512,
        C=4,
        f=8,
        scale=10.0,
        ddim_steps=50,
        ddim_eta=0.0,
        callback=None,
        skip_single_save=False,
        save_grid=True,
        ucg_schedule=None,
        negative_prompt="",
        adm_cond=None,
        adm_uc=None,
        use_full_precision=False,
        only_adm_cond=False,
        **kwargs
):
    batch_size = n_samples
    precision_scope = autocast if not use_full_precision else nullcontext
    # decoderscope = autocast if not use_full_precision else nullcontext
    if use_full_precision: print(f"Warning: Running {model.__class__.__name__} at full precision.")
    if isinstance(prompt, str):
        prompt = [prompt]
    prompts = batch_size * prompt

    # outputs = st.empty()

    with precision_scope("cuda"):
        with model.ema_scope():
            all_samples = list()
            for n in trange(n_runs, desc="Sampling"):
                shape = [C, H // f, W // f]
                if not only_adm_cond:
                    uc = None
                    if scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [negative_prompt])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)

                if adm_cond is not None:
                    if adm_cond.shape[0] == 1:
                        adm_cond = repeat(adm_cond, '1 ... -> b ...', b=batch_size)
                    if adm_uc is None:
                        print("Warning: Not guiding via c_adm")
                        adm_uc = adm_cond
                    else:
                        if adm_uc.shape[0] == 1:
                            adm_uc = repeat(adm_uc, '1 ... -> b ...', b=batch_size)
                    if not only_adm_cond:
                        c = {"c_crossattn": [c], "c_adm": adm_cond}
                        uc = {"c_crossattn": [uc], "c_adm": adm_uc}
                    else:
                        c = adm_cond
                        uc = adm_uc
                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                 conditioning=c,
                                                 batch_size=batch_size,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=uc,
                                                 eta=ddim_eta,
                                                 x_T=None,
                                                 callback=callback,
                                                 ucg_schedule=ucg_schedule,
                                                 **kwargs
                                                 )
                x_samples = model.decode_first_stage(samples_ddim)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                # print("Debug:")
                # for x_sample in x_samples:
                #     probs = F.softmax(risky_model(custom_tr(x_sample.unsqueeze(0)).to(device)), dim=1)
                #     print(torch.max(probs, dim=1)[0])
                # print("Oracle:")
                if not skip_single_save:
                    base_count = len(os.listdir(os.path.join(SAVE_PATH, "samples")))
                    for x_sample in x_samples:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        image = Image.fromarray(x_sample.astype(np.uint8))
                        image.save(os.path.join(SAVE_PATH, "samples", f"{base_count:09}.png"))
                        # probs = F.softmax(risky_model(dataset.transform(image).to(device).unsqueeze(0)), dim=1)
                        # print(torch.max(probs, dim=1)[0])
                        base_count += 1

                all_samples.append(x_samples)

                # get grid of all samples
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n h) (b w) c')
                # outputs.image(grid.cpu().numpy())

            # additionally, save grid
            grid = Image.fromarray((255. * grid.cpu().numpy()).astype(np.uint8))
            if save_grid:
                grid_count = len(os.listdir(SAVE_PATH)) - 1
                grid.save(os.path.join(SAVE_PATH, f'grid-{grid_count:06}.png'))

    return x_samples


def make_oscillating_guidance_schedule(num_steps, max_weight=15., min_weight=1.):
    schedule = list()
    for i in range(num_steps):
        if float(i / num_steps) < 0.1:
            schedule.append(max_weight)
        elif i % 2 == 0:
            schedule.append(min_weight)
        else:
            schedule.append(max_weight)
    print(f"OSCILLATING GUIDANCE SCHEDULE: \n {schedule}")
    return schedule


def torch2np(x):
    x = ((x + 1.0) * 127.5).clamp(0, 255).to(dtype=torch.uint8)
    x = x.permute(0, 2, 3, 1).detach().cpu().numpy()
    return x

#st.cache?
def init(version="unCLIP-L"):
    state = dict()
    if not "model" in state:
        if version == "unCLIP-L":
            config = "configs/stable-diffusion/v2-1-stable-unclip-l-inference.yaml"
            ckpt = "checkpoints/sd21-unclip-l.ckpt"

        elif version == "unOpenCLIP-H":
            config = "configs/stable-diffusion/v2-1-stable-unclip-h-inference.yaml"
            ckpt = "checkpoints/sd21-unclip-h.ckpt"
        else:
            raise ValueError(f"version {version} unknown!")

        config = OmegaConf.load(config)
        model, msg = load_model_from_config(config, ckpt, vae_sd=None)
        state["msg"] = msg
        state["model"] = model
        state["ckpt"] = ckpt
        state["config"] = config
    return state


def load_model_from_config(config, ckpt, verbose=False, vae_sd=None):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    msg = None
    if "global_step" in pl_sd:
        msg = f"This is global step {pl_sd['global_step']}. "
    if "model_ema.num_updates" in pl_sd["state_dict"]:
        msg += f"And we got {pl_sd['state_dict']['model_ema.num_updates']} EMA updates."
    global_step = pl_sd.get("global_step", "?")
    sd = pl_sd["state_dict"]
    if vae_sd is not None:
        for k in sd.keys():
            if "first_stage" in k:
                sd[k] = vae_sd[k[len("first_stage_model."):]]

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    print(f"Loaded global step {global_step}")
    return model, msg


if __name__ == "__main__":
    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))
    dataset = get_dataset_class(args.dataset)() 
    torch.set_grad_enabled(False)                
    version = args.model_version
    state = init(version=version)
    
    if args.gradient_scale > 0 or args.drop_prompt:
        risky_model = misc.load_model(dataset, args.risky_model_arch)
        risky_model.to(device)
        risky_model.eval() 
        misc.freeze_all_parameters(risky_model)
        assert not misc.has_trainable_parameters(risky_model)
        print("Succeed in loading the risky model!")
    else:
        risky_model = None
    
    if not args.no_filter:
        with open(os.path.join(args.pkldir, f"{args.dataset}-{args.risky_model_arch}-merge.pkl"), "rb") as f:
            results = pickle.load(f)
        with open(os.path.join(args.pkldir, f"{args.dataset}-features-merge.pkl"), "rb") as f:
            features = pickle.load(f)
        if args.holdout:
            phases = ["val", "test"]    
        else:
            phases = ["all"]
            results["all"] = dict()
            for key in  ["labels", "losses", "error"]:
                results["all"][key] = np.concatenate([results[phase][key] for phase in ["val", "test"]])
            features["all"] = np.concatenate([features[phase] for phase in ["val", "test"]])
            

        output_dim = 2
        input_dim = feat_dim
        class_weight = [1, (results[phases[0]]["error"]==0).sum()/(results[phases[0]]["error"]==1).sum()]
        # print(class_weight)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weight, device=device).float())    
        if args.dataset != "PACS":
            model_eval = misc.MLP(input_dim, output_dim, [input_dim*2, input_dim])
        else:
            model_eval = misc.MLP(input_dim, output_dim)        
        if args.load_model_eval:
            model_eval.load_state_dict(torch.load(f"model_eval/{args.dataset}-{args.risky_model_arch}-MLP.pth", map_location="cpu"))
            model_eval.to(device)
        else:
            # train MLP
            torch.set_grad_enabled(True)
            model_eval.to(device)
            dataloaders = dict()
            for phase in phases:
                print("Phase %s numpy dataset construction..." % phase)
                cur_dataset = misc.NumpyDataset(features[phase], results[phase]["error"], discrete=True)
                dataloaders[phase] = DataLoader(cur_dataset, batch_size=64, shuffle=(phase!="test"), num_workers=8)    

            ## Training loop
            model_eval.train()
            optimizer = optim.Adam(model_eval.parameters(), lr=lr)
            for epoch in range(num_epochs):
                for i, (X, Y) in enumerate(dataloaders[phases[0]]):
                    X, Y = X.to(device), Y.to(device)
                    outputs = model_eval(X)
                    loss = criterion(outputs, Y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if (i+1) % 20 == 0:
                        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloaders[phases[0]])}], Loss: {loss.item():.4f}')
            torch.set_grad_enabled(False)
            if args.save_model_eval:
                os.makedirs("model_eval", exist_ok=True)
                torch.save(model_eval.state_dict(), f"model_eval/{args.dataset}-{args.risky_model_arch}-MLP.pth")
            
            ## Testing the model
            model_eval.eval()  # Set model to evaluation mode
            for phase in phases:
                precision, recall, all_preds, all_labels = misc.calculate_precision_recall(model_eval, dataloaders[phase], device, threshold=-1, discrete=True)
                print("Error prediction on %s data: precision=%.4f, recall=%.4f" % (phase, precision, recall))
                if phase == "test":
                    category_valid = []
                    for idx in range(dataset.num_classes):
                        cate_filter = results["test"]["labels"]==idx
                        preds_cate = all_preds[cate_filter]
                        labels_cate = all_labels[cate_filter]
                        precision_cate = precision_score(labels_cate, preds_cate, average='binary')
                        recall_cate = recall_score(labels_cate, preds_cate, average='binary')
                        print("%s: precision=%.4f, recall=%.4f, number of predicted error=%d, number of true error=%d" % (dataset.label2class[idx], precision_cate, recall_cate, (preds_cate==1).sum(), (labels_cate==1).sum()))
                        if precision_cate > 0:
                            category_valid.append(dataset.label2class[idx])
                    os.makedirs("valid_categories", exist_ok=True)
                    with open(os.path.join("valid_categories", f"{args.dataset}-{args.risky_model_arch}.txt"), "w") as f:
                        for category in category_valid:
                            f.write(f"\"{category}\" ")
    else:
        model_eval = None
    
    # search for risky vectors
    vector_list = []
    
    with open(os.path.join(args.pkldir, f"{args.dataset}-features.pkl"), "rb") as f:
        features_cate = pickle.load(f)
    features_cate = np.concatenate([features_cate[phase][args.category] for phase in ["val", "test"]])
    
    # Employ Gaussian distribution whose mean is the mean of a category's all samples
    mean = torch.tensor(features_cate.mean(axis=0), device=device, requires_grad=True).reshape((1,-1))
    std = torch.tensor(features_cate.std(axis=0), device=device, requires_grad=True).reshape((1,-1))
    # enumerate until a fixed number of samples are reached
    cnt = 0
    candidates_list = []
    progress_bar = tqdm(total=args.num_candidates)
    start_time = time.time()
    while cnt < args.num_candidates:
        candidates = torch.randn(10000*args.num_candidates, feat_dim, device=device)*std*args.std_scale+mean
        if not args.no_filter:
            output = model_eval(candidates)
            probs = F.softmax(output, dim=1)
            confs, preds = torch.max(probs, 1)
            error_filter = (preds==1)*(confs>args.conf_threshold)
            candidates_list.append(candidates[error_filter])
            num_update = int(error_filter.sum())
        else:
            candidates_list.append(candidates)
            num_update = len(candidates)
        progress_bar.update(num_update)
        cnt += num_update
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time > args.max_filter_time:
            print("Time exceeded!")
            os.makedirs("failed_categories", exist_ok=True)
            with open(os.path.join("failed_categories", f"{args.dataset}-{args.risky_model_arch}.txt"), "a") as f:
                f.write(f"\"{args.category}\" ")
            break
    if cnt < args.num_candidates:
        candidates_list.append(torch.randn(args.num_candidates-cnt, feat_dim, device=device)*std*args.std_scale+mean)
        cnt = args.num_candidates
    vector_list = torch.cat(candidates_list)
    if cnt > args.num_candidates:
        vector_list = vector_list[:args.num_candidates]
    assert vector_list.shape[0] == args.num_candidates
    
    # Generate risky examples
    prompt = args.prompt
    negative_prompt = args.neg_prompt
    scale = args.cfg_scale
    number_rows = args.nrow
    number_cols = args.ncol
    steps = args.steps
    eta = args.eta
    check_range(scale, -100., 100.)
    check_range(number_rows, 1, 10)
    check_range(number_cols, 1, 10)
    check_range(steps, 1, 1000)
    check_range(eta, 0., 1.)
    force_full_precision = args.force_fp32  # TODO: check if/where things break.
    H = args.H
    W = args.W
    check_range(H, 64, 2048)
    check_range(W, 64, 2048)
    C = VERSION2SPECS[version]["C"]
    f = VERSION2SPECS[version]["f"]

    SAVE_PATH = args.outdir
    os.makedirs(os.path.join(SAVE_PATH, "samples"), exist_ok=True)

    ucg_schedule = None
    sampler = args.sampler
    if sampler == "DPM":
        sampler = DPMSolverSampler(state["model"])
    elif sampler == "DDIM":
        sampler = DDIMSampler(state["model"])
    else:
        raise ValueError(f"unknown sampler {sampler}!")

    num_inputs = args.num_inputs


    def make_conditionings_from_vector(vector):
        with torch.no_grad():
            adm_cond = vector
            adm_cond = adm_cond.repeat(number_cols).reshape(number_cols, -1)
            weight = args.weight_input
            check_range(weight, inf=-10., sup=10.)
            if state["model"].noise_augmentor is not None:
                noise_level = args.noise_level_input
                check_range(noise_level, 0, state["model"].noise_augmentor.max_noise_level - 1)
                c_adm, noise_level_emb = state["model"].noise_augmentor(adm_cond, noise_level=repeat(
                    torch.tensor([noise_level]).to(state["model"].device), '1 -> b', b=number_cols))
                adm_cond = torch.cat((c_adm, noise_level_emb), 1) * weight
            adm_uc = torch.zeros_like(adm_cond)
        return adm_cond, adm_uc, weight
    start = time.time()
    for idx, cond_vector in enumerate(vector_list):
        adm_cond, adm_uc = None, None
        adm_inputs = list()
        weights = list()
        for n in range(num_inputs):
            adm_cond, adm_uc, w = make_conditionings_from_vector(vector=cond_vector)
            weights.append(w)
            adm_inputs.append(adm_cond)
        adm_cond = torch.stack(adm_inputs).sum(0) / sum(weights)
        if num_inputs > 1:
            if args.noise_embedding_mix:
                noise_level = args.noise_level_avg
                check_range(noise_level, 0, state["model"].noise_augmentor.max_noise_level - 1)
                c_adm, noise_level_emb = state["model"].noise_augmentor(
                    adm_cond[:, :state["model"].noise_augmentor.time_embed.dim],
                    noise_level=repeat(
                        torch.tensor([noise_level]).to(state["model"].device), '1 -> b', b=number_cols))
                adm_cond = torch.cat((c_adm, noise_level_emb), 1)

        print("Sampling")


        def t_callback(t):
            pass
            # print("Step %d/%d" % (t, steps))

        samples = sample(
            state["model"],
            prompt,
            n_runs=number_rows,
            n_samples=number_cols,
            H=H, W=W, C=C, f=f,
            scale=scale,
            ddim_steps=steps,
            ddim_eta=eta,
            callback=t_callback,
            ucg_schedule=ucg_schedule,
            negative_prompt=negative_prompt,
            adm_cond=adm_cond, adm_uc=adm_uc,
            use_full_precision=force_full_precision,
            only_adm_cond=False,
            risk_predictor=risky_model,
            gradient_scale=args.gradient_scale,
            match_scale=args.match_scale,
            category_label=dataset.class2label[args.category],
            category_name=args.category,
            text_prompt_range=args.text_prompt_range,
            gradient_range=args.gradient_range,
            outdir=args.outdir,
            drop_prompt=args.drop_prompt,
            decode_tr=dataset.decode_tr,
            save_grid=args.save_grid
        )
    duration = time.time()-start
    print(f"Total time cost: {duration:.2f}")
    duration_perimage = duration/(args.num_candidates*args.nrow*args.ncol)
    print(f"Time per image: {duration_perimage:.2f})")