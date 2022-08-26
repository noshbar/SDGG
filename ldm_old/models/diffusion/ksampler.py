'''wrapper around part of Karen Crownson's k-duffsion library, making it call compatible with other Samplers'''
import k_diffusion as K
import torch
import torch.nn as nn
import accelerate

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale

class KSampler(object):
    def __init__(self,model,schedule="lms", **kwargs):
        super().__init__()
        self.model        = K.external.CompVisDenoiser(model)
        self.accelerator  = accelerate.Accelerator()
        self.device       = self.accelerator.device
        self.schedule = schedule

        def forward(self, x, sigma, uncond, cond, cond_scale):
            x_in = torch.cat([x] * 2)
            sigma_in = torch.cat([sigma] * 2)
            cond_in = torch.cat([uncond, cond])
            uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
            return uncond + (cond - uncond) * cond_scale


    # most of these arguments are ignored and are only present for compatibility with
    # other samples
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
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):

        sigmas = self.model.get_sigmas(S)
        if x_T:
            x = x_T
        else:
            x = torch.randn([batch_size, *shape], device=self.device) * sigmas[0] # for GPU draw
        model_wrap_cfg = CFGDenoiser(self.model)
        extra_args = {'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': unconditional_guidance_scale}
        return (K.sampling.sample_lms(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not self.accelerator.is_main_process),
                None)

    def gather(samples_ddim):
        return self.accelerator.gather(samples_ddim)
