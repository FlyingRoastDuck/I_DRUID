from typing import Optional, Tuple, Union
from diffusers import DDIMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput
from diffusers.utils.torch_utils import randn_tensor
import torch
from torch.distributions import Normal
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
import math


class DDIMSchedulerExtended(DDIMScheduler):
    def _get_variance_logprob(self, timestep, prev_timestep):
        alpha_prod_t = self.alphas_cumprod.to(timestep.device)[timestep]
        mask_a = (prev_timestep >= 0).int().to(timestep.device)
        mask_b = 1 - mask_a
        alpha_prod_t_prev = (
            self.alphas_cumprod.to(timestep.device)[prev_timestep] * mask_a
            + self.final_alpha_cumprod.to(timestep.device) * mask_b
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev / beta_prod_t) * (
            1 - alpha_prod_t / alpha_prod_t_prev
        )
        return variance

    def step_logprob(
        self,  model_output,
        timestep, sample, eta = 1.0,
        use_clipped_model_output = False,
        generator=None, variance_noise = None,
        return_dict = True,
    ):
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps'"
                " after creating the scheduler"
            )

        prev_timestep = (
            timestep - self.config.num_train_timesteps // self.num_inference_steps
        )

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod.to(timestep.device)[timestep]
        # alpha_prod_t = alpha_prod_t.to(torch.float16)
        mask_a = (prev_timestep >= 0).int().to(timestep.device)
        mask_b = 1 - mask_a
        alpha_prod_t_prev = (
            self.alphas_cumprod.to(timestep.device)[prev_timestep] * mask_a
            + self.final_alpha_cumprod.to(timestep.device) * mask_b
        )
        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.config.prediction_type == "epsilon":
            # pred x0
            pred_original_sample = (
                sample - beta_prod_t ** (0.5) * model_output
            ) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (
                beta_prod_t**0.5
            ) * model_output
            # predict V
            model_output = (alpha_prod_t**0.5) * model_output + (
                beta_prod_t**0.5
            ) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one"
                " of `epsilon`, `sample`, or `v_prediction`"
            )

        # 4. Clip "predicted x_0"
        if self.config.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance_logprob(timestep, prev_timestep).to(
            dtype=sample.dtype
        )
        std_dev_t = (eta * variance ** (0.5)).to(dtype=sample.dtype)

        if use_clipped_model_output:
            # the model_output is always re-derived from the clipped x_0 in Glide
            model_output = (
                sample - alpha_prod_t ** (0.5) * pred_original_sample
            ) / beta_prod_t ** (0.5)

        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (
            0.5
        ) * model_output

        # pylint: disable=line-too-long
        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = (
            alpha_prod_t_prev ** (0.5) * pred_original_sample
            + pred_sample_direction
        )

        if eta > 0:
            device = model_output.device
        if variance_noise is not None and generator is not None:
            raise ValueError(
                "Cannot pass both generator and variance_noise. Please make sure"
                " that either `generator` or `variance_noise` stays `None`."
            )

        if variance_noise is None:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=device,
                dtype=model_output.dtype,
            )
        variance = std_dev_t * variance_noise
        dist = Normal(prev_sample, std_dev_t)
        prev_sample = prev_sample.detach().clone() + variance
        log_prob = (
            dist.log_prob(prev_sample.detach().clone())
            .mean(dim=-1)
            .mean(dim=-1)
            .mean(dim=-1)
            .detach()
            .cpu()
        )
        if not return_dict: return (prev_sample,)
        # return  xt-1 latent and get log_prob for optim
        return (
            DDIMSchedulerOutput(
                prev_sample=prev_sample, pred_original_sample=pred_original_sample
            ),
            log_prob,
        )

    def step_forward_logprob(
        self, model_output, timestep, sample, next_sample, eta = 1.0, use_clipped_model_output = False,
        generator=None, variance_noise = None, return_dict = True,
    ): 
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps'"
                " after creating the scheduler"
            )

        prev_timestep = (
            timestep - self.config.num_train_timesteps // self.num_inference_steps
        )

        alpha_prod_t = self.alphas_cumprod.to(timestep.device)[timestep]
        mask_a = (prev_timestep >= 0).int().to(timestep.device)
        mask_b = 1 - mask_a
        alpha_prod_t_prev = (
            self.alphas_cumprod.to(timestep.device)[prev_timestep] * mask_a
            + self.final_alpha_cumprod.to(timestep.device) * mask_b
        )
        beta_prod_t = 1 - alpha_prod_t

        if self.config.prediction_type == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t ** (0.5) * model_output
            ) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (
                beta_prod_t**0.5
            ) * model_output
            # predict V
            model_output = (alpha_prod_t**0.5) * model_output + (
                beta_prod_t**0.5
            ) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one"
                " of `epsilon`, `sample`, or `v_prediction`"
            )

        # 4. Clip "predicted x_0"
        if self.config.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance_logprob(timestep, prev_timestep).to(
            dtype=sample.dtype
        )
        std_dev_t = (eta * variance ** (0.5)).to(dtype=sample.dtype)

        if use_clipped_model_output:
            # the model_output is always re-derived from the clipped x_0 in Glide
            model_output = (
                sample - alpha_prod_t ** (0.5) * pred_original_sample
            ) / beta_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (
            0.5
        ) * model_output

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = (
            alpha_prod_t_prev ** (0.5) * pred_original_sample
            + pred_sample_direction
        )

        if eta > 0:
            device = model_output.device
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure"
                    " that either `generator` or `variance_noise` stays `None`."
                )

            if variance_noise is None:
                variance_noise = randn_tensor(
                    model_output.shape,
                    generator=generator,
                    device=device,
                    dtype=model_output.dtype,
                )
                
            variance = std_dev_t * variance_noise
            dist = Normal(prev_sample, std_dev_t)
            log_prob = (
                dist.log_prob(next_sample.detach().clone())
                .mean(dim=-1)
                .mean(dim=-1)
                .mean(dim=-1)
            )

        return log_prob

    def get_x0(self, xt, pred_noise, t):
        alpha_prod_t = self.alphas_cumprod[t]
        beta_prod_t = 1 - alpha_prod_t
        x0 = (xt - beta_prod_t.view(-1,1,1,1) ** (0.5) * pred_noise) / alpha_prod_t.view(-1,1,1,1) ** (0.5)
        return x0


class FlowSchedulerExtended(FlowMatchEulerDiscreteScheduler):
    def sde_step_with_logprob(
        self, model_output, timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor, noise_level = 0.7,
        prev_sample: Optional[torch.FloatTensor] = None, generator: Optional[torch.Generator] = None,
    ):
        # bf16 can overflow here when compute prev_sample_mean, we must convert all variable to fp32
        model_output=model_output.float()
        sample=sample.float()
        if prev_sample is not None:
            prev_sample=prev_sample.float()

        step_index = [self.index_for_timestep(t) for t in timestep]
        prev_step_index = [step+1 for step in step_index]
        sigma = self.sigmas[step_index].view(-1, *([1] * (len(sample.shape) - 1)))
        sigma_prev = self.sigmas[prev_step_index].view(-1, *([1] * (len(sample.shape) - 1)))
        sigma_max = self.sigmas[1].item()
        dt = sigma_prev - sigma
        std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma)))*noise_level
        
        # our sde
        prev_sample_mean = sample*(1+std_dev_t**2/(2*sigma)*dt)+model_output*(1+std_dev_t**2*(1-sigma)/(2*sigma))*dt
        
        if prev_sample is None:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1*dt) * variance_noise

        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1*dt))**2))
            - torch.log(std_dev_t * torch.sqrt(-1*dt))
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
        # mean along all but batch dimension
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return prev_sample, log_prob, prev_sample_mean, std_dev_t

    def compute_log_prob(self, noise_pred, sample, time):
        # compute the log prob of next_latents given latents under the current model
        prev_sample, log_prob, prev_sample_mean, std_dev_t = self.sde_step_with_logprob(
            noise_pred.float(), time, sample.float(), prev_sample=None, noise_level=0.7,
        )
        return prev_sample, log_prob, prev_sample_mean, std_dev_t
    