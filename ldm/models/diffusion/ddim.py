"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms
from copy import deepcopy
import os
import open_clip

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor


postprocess = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", device=torch.device("cuda"), **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.device = device

    def pgd_attack(self, model, images, labels, eps=0.1, alpha=0.01, iters=40, **kwargs):
        ori_images = images.data
        for i in range(iters):    
            images.requires_grad = True
            x_decode = self.model.decode_first_stage(images, requires_grad=True)
            x_decode = torch.clamp((x_decode + 1.0) / 2.0, min=0.0, max=1.0)
            x_decode = kwargs["decode_tr"](x_decode)
            outputs = model(x_decode)
            preds = outputs.argmax(1)
            if torch.sum(torch.abs(preds-labels)) != 0:
                break
            model.zero_grad()
            cost = F.cross_entropy(outputs, labels)
            cost.backward()

            adv_images = images + alpha*images.grad.sign()
            eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
            images = (ori_images+eta).detach_()
            # images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
        return images.detach()

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != self.device:
                attr = attr.to(self.device)
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None, # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               dynamic_threshold=None,
               ucg_schedule=None,
               **kwargs
               ):
        
        # build empty condition
        self.empty_cond = self.model.get_learned_conditioning(batch_size*[""])
        self.category_embedding = self.model.cond_stage_model.model.encode_text(open_clip.tokenize(kwargs["category_name"]).to(self.device))
        self.disable_prompt = False
        self.clip_tr = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list): ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            elif isinstance(conditioning, list):
                for ctmp in conditioning:
                    if ctmp.shape[0] != batch_size:
                        print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    dynamic_threshold=dynamic_threshold,
                                                    ucg_schedule=ucg_schedule,
                                                    **kwargs
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      ucg_schedule=None,
                      **kwargs):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            if ucg_schedule is not None:
                assert len(ucg_schedule) == len(time_range)
                unconditional_guidance_scale = ucg_schedule[i]
            
            # enable or disable prompt
            cond_new = deepcopy(cond)
            # if index < total_steps*(1-kwargs["text_prompt_range"]):
            if self.disable_prompt:
                if isinstance(cond_new, dict):
                    cond_new["c_crossattn"] = [self.empty_cond,]
                else:
                    cond_new = self.empty_cond
            outs = self.p_sample_ddim(img, cond_new, ts, index=index, total_steps=total_steps, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      dynamic_threshold=dynamic_threshold,
                                      **kwargs)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, total_steps, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None, **kwargs):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            model_output = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            if isinstance(c, dict):
                assert isinstance(unconditional_conditioning, dict)
                c_in = dict()
                for k in c:
                    if isinstance(c[k], list):
                        c_in[k] = [torch.cat([
                            unconditional_conditioning[k][i],
                            c[k][i]]) for i in range(len(c[k]))]
                    else:
                        c_in[k] = torch.cat([
                                unconditional_conditioning[k],
                                c[k]])
            elif isinstance(c, list):
                c_in = list()
                assert isinstance(unconditional_conditioning, list)
                for i in range(len(c)):
                    c_in.append(torch.cat([unconditional_conditioning[i], c[i]]))
            else:
                c_in = torch.cat([unconditional_conditioning, c])
            model_uncond, model_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == "eps", 'not implemented'
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)


        # calculate risky gradient
        if kwargs["gradient_scale"] > 0 and index < int(total_steps*kwargs["gradient_range"]):
            torch.set_grad_enabled(True)
            if kwargs["category_label"] is not None:
                pseudo_y = torch.tensor(kwargs["category_label"], device=device, dtype=torch.long).reshape((1,))
            grad_list = []
            prob_list = []
            for idx in range(x.shape[0]):
                x_hasgrad = x[idx:idx+1].clone()
                x_hasgrad.requires_grad_()
                if self.model.parameterization != "v":
                    pred_x0_temp = (x_hasgrad - sqrt_one_minus_at[idx:idx+1] * e_t[idx:idx+1]) / a_t[idx:idx+1].sqrt()
                else:
                    pred_x0_temp = self.model.predict_start_from_z_and_v(x_hasgrad, t[idx:idx+1], model_output[idx:idx+1])
                if quantize_denoised:
                    pred_x0_temp, _, *_ = self.model.first_stage_model.quantize(pred_x0_temp)                
                x_decode = self.model.decode_first_stage(pred_x0_temp, requires_grad=True)
                # add transforms
                x_decode = torch.clamp((x_decode + 1.0) / 2.0, min=0.0, max=1.0)
                if kwargs["match_scale"] > 0:
                    x_for_clip = self.clip_tr(x_decode)
                x_decode = kwargs["decode_tr"](x_decode)
                logits = kwargs["risk_predictor"](x_decode)
                if kwargs["category_label"] is not None:
                    risk_score = F.cross_entropy(logits, pseudo_y)
                    prob_list.append(float(torch.exp(-risk_score)))
                else:
                    probs = F.softmax(logits, dim=1)
                    risk_score = -torch.sum(probs*torch.log(probs+1e-20))
                if kwargs["match_scale"] > 0:
                    image_embedding = self.model.embedder.encode_with_vision_transformer(x_for_clip, preprocess=False)
                    match_score = torch.sum(image_embedding*self.category_embedding)
                else:
                    match_score = 0
                score = risk_score+kwargs["match_scale"]*match_score
                score.backward()
                grad_list.append(x_hasgrad.grad/torch.sqrt(torch.sum(x_hasgrad.grad**2)+1e-10))
            # with open("log-%d.txt"%(int(kwargs["gradient_scale"])), "a") as f:
            #     f.write("%d: "%index+str(prob_list)+"\n")
            torch.set_grad_enabled(False)
            risk_grad = torch.cat(grad_list)
            e_t = e_t - kwargs["gradient_scale"]*sqrt_one_minus_at*risk_grad
        elif "drop_prompt" in kwargs and kwargs["drop_prompt"] and not self.disable_prompt:
            if kwargs["category_label"] is not None:
                pseudo_y = torch.tensor(kwargs["category_label"], device=device, dtype=torch.long).reshape((1,))
            prob_list = []
            pred_list = []
            for idx in range(x.shape[0]):
                if self.model.parameterization != "v":
                    pred_x0_temp = (x[idx:idx+1] - sqrt_one_minus_at[idx:idx+1] * e_t[idx:idx+1]) / a_t[idx:idx+1].sqrt()
                else:
                    pred_x0_temp = self.model.predict_start_from_z_and_v(x[idx:idx+1], t[idx:idx+1], model_output[idx:idx+1])
                if quantize_denoised:
                    pred_x0_temp, _, *_ = self.model.first_stage_model.quantize(pred_x0_temp)                
                x_decode = self.model.decode_first_stage(pred_x0_temp)
                x_decode = torch.clamp((x_decode + 1.0) / 2.0, min=0.0, max=1.0)
                
                logits = kwargs["risk_predictor"](postprocess(x_decode.squeeze(0)).unsqueeze(0))
                preds = logits.argmax(1)
                pred_list.append(int(preds))
                if kwargs["category_label"] is not None:
                    score = F.cross_entropy(logits, pseudo_y)
                    prob_list.append(float(torch.exp(-score)))
                else:
                    probs = F.softmax(logits, dim=1)
                    score = -torch.sum(probs*torch.log(probs+1e-20))
            if np.sum(np.array(pred_list) != int(pseudo_y)) == 0:
            # if index < total_steps*3//4:
                self.disable_prompt = True
                with open(os.path.join(kwargs["outdir"], "log.txt"), "a") as f:
                    f.write("Idx=%d, prob=%s\n"%(int(index), str(prob_list)))
        # current prediction for x_0
        if self.model.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            if kwargs["gradient_scale"] > 0 and index < int(total_steps*kwargs["gradient_range"]):
                v_update = model_output-kwargs["gradient_scale"]*sqrt_one_minus_at/a_t.sqrt()*risk_grad
                pred_x0 = self.model.predict_start_from_z_and_v(x, t, v_update)
            else:
                pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        if dynamic_threshold is not None:
            raise NotImplementedError()

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        if "PGD" in kwargs and kwargs["PGD"]:
            torch.set_grad_enabled(True)
            if kwargs["category_label"] is not None:
                pseudo_y = torch.tensor(kwargs["category_label"], device=device, dtype=torch.long).reshape((1,))
            x_prev_list = []
            for idx in range(x_prev.shape[0]):
                x_prev_single = x_prev[idx:idx+1]
                x_prev_new = self.pgd_attack(kwargs["risk_predictor"], x_prev_single, pseudo_y, eps=kwargs["PGD_scale"]*(1-self.ddim_alphas[index]), iters=kwargs["PGD_iters"], **kwargs)
                x_prev_list.append(x_prev_new)
            x_prev = torch.cat(x_prev_list)
            torch.set_grad_enabled(False)
        return x_prev, pred_x0

    @torch.no_grad()
    def encode(self, x0, c, t_enc, use_original_steps=False, return_intermediates=None,
               unconditional_guidance_scale=1.0, unconditional_conditioning=None, callback=None):
        num_reference_steps = self.ddpm_num_timesteps if use_original_steps else self.ddim_timesteps.shape[0]

        assert t_enc <= num_reference_steps
        num_steps = t_enc

        if use_original_steps:
            alphas_next = self.alphas_cumprod[:num_steps]
            alphas = self.alphas_cumprod_prev[:num_steps]
        else:
            alphas_next = self.ddim_alphas[:num_steps]
            alphas = torch.tensor(self.ddim_alphas_prev[:num_steps])

        x_next = x0
        intermediates = []
        inter_steps = []
        for i in tqdm(range(num_steps), desc='Encoding Image'):
            t = torch.full((x0.shape[0],), i, device=self.model.device, dtype=torch.long)
            if unconditional_guidance_scale == 1.:
                noise_pred = self.model.apply_model(x_next, t, c)
            else:
                assert unconditional_conditioning is not None
                e_t_uncond, noise_pred = torch.chunk(
                    self.model.apply_model(torch.cat((x_next, x_next)), torch.cat((t, t)),
                                           torch.cat((unconditional_conditioning, c))), 2)
                noise_pred = e_t_uncond + unconditional_guidance_scale * (noise_pred - e_t_uncond)

            xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
            weighted_noise_pred = alphas_next[i].sqrt() * (
                    (1 / alphas_next[i] - 1).sqrt() - (1 / alphas[i] - 1).sqrt()) * noise_pred
            x_next = xt_weighted + weighted_noise_pred
            if return_intermediates and i % (
                    num_steps // return_intermediates) == 0 and i < num_steps - 1:
                intermediates.append(x_next)
                inter_steps.append(i)
            elif return_intermediates and i >= num_steps - 2:
                intermediates.append(x_next)
                inter_steps.append(i)
            if callback: callback(i)

        out = {'x_encoded': x_next, 'intermediate_steps': inter_steps}
        if return_intermediates:
            out.update({'intermediates': intermediates})
        return x_next, out

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False, callback=None):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return x_dec