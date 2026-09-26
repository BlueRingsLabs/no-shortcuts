# Module 31: Generative Models

*Part IV: Deep Learning · about 40 hours*

Teaching machines to make things, and understanding the math instead of just the pretty samples.

## Self-check

If you can answer these without looking anything up, skim the lessons and go straight to the labs.
If you can't, good, that's what the module is for.

- Why can't you backpropagate through sampling without the reparameterization trick?

## When you finish it, you can

- Derive the ELBO and implement a VAE with the reparameterization trick.
- Explain the GAN minimax game, its failure modes, and the fixes (WGAN-GP, spectral norm).
- Derive and implement a DDPM diffusion model.
- Explain flow matching, latent diffusion, and classifier-free guidance.

## Lessons

31.1. [Autoencoders and VAEs](31.1-autoencoders-vae.md)

31.2. [GANs](31.2-gans.md)

31.3. [Diffusion models](31.3-diffusion.md)

31.4. [Flow matching, latent diffusion and guidance](31.4-flow-matching-latent-diffusion.md)

## Labs

Run them from the repository root, for example:

```bash
python part-4-deep-learning/31-generative-models/labs/lab_31_1_autoencoders_vae.py
```

- [`lab_31_1_autoencoders_vae.py`](labs/lab_31_1_autoencoders_vae.py)
- [`lab_31_2_gans.py`](labs/lab_31_2_gans.py)
- [`lab_31_3_diffusion.py`](labs/lab_31_3_diffusion.py)
- [`lab_31_4_flow_matching.py`](labs/lab_31_4_flow_matching.py)

Back to [Part IV](../) · [Syllabus](../../SYLLABUS.md)
