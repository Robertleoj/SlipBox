---
tags: 
 - deep # deep, diffusion, etc
 - diffusion
 - AI

type: paper
link: "https://arxiv.org/abs/2112.10752"

creation date: 2022-11-20 17:19
modification date: Sunday 20th November 2022 17:19:11
---

The stable diffusion paper. Legendary.

Instead of performing the diffusion directly in pixel space (which is large), perform the diffusion in the latent space of the decoder of a pretrained autoencoder. This space is much smaller, and leads to much faster and less consuming training. 

They claim that thye have created powerful and flexible generators for general conditioning inputs (test, bounding boxes, etc)

## Introduction
* Diffusion models heavily utilize parameter sharing. I haven't thought about it that way. They say that this is what enables them to achieve very good results without being huge in terms of parameters. 
* The reason DMs are so computationally intensive is that they operate in the huge pixel space
* the high dimensional pixel space also makes inference expensive
* They emphasize the point that they do not need to rely on excessive spatial compression in the autoencoder because the will perform the diffusion there, which exhibit better scaling properties w.r.t. the spatial dimensionality (??) 
* They can train the autoencoder only once, and then train many models to do different tasks in its latent space. I'm guessing conditioning on different information. They mention text-to-image, and image-to-image tasks (how do they work?). 
* For the text-to-image task, they connect transformers to the DM's UNet backbone. 
* They mention (stochastic) super-resolution, and image inpainting. How do those work within their framework?
* They say that they need very little regularization of the latent space of the autoencoder. How do they do that?
* Their conditioning mechanism is based on cross-attention. How does that work?
* They train layout-to-image models. What is that?

## Related work
* They mention flow-based models. What are those? Are they still relevant?
* Also mention autoregressive models. I'm pretty sure that this means they are choosing some amount of pixels at a time, and conditioning on the previous outputs. 
* Previous work has used ARMs to model images in latent space
* Inductive bias (built-in assumptions of learner) of UNet is perfect for image-like data
* They are free to choose the level of compression that does not leave the model to do too much perceptual compression. They can thus test until they find the optimal amount of compression. 
* Jointly learning the autoencoder and the generative model is too hard, and outperformed by not doing that. 
## Method
* What they are doing is an explicit separation between the compressive and the generative learning phase. This pretty much means that you only have to do the compression and decompression once, and the generative process happens in between, *inside* the compressed space. 
* They train the autoencoder with a combination of a perceptual loss, and a patch-based loss. What are these? This apparently enforces local realism and avoids bluriness. I marked the paper for the perceptual loss, but look at the references for the adversarial objective.
* They used some interesting penalties to avoid arbitrarily high-variance latent space. Check out the loss functions in Appendix G
* Cross-attention: Realllllly coooool. make a permanent note from it.  You basically only use the Unet's data as the query, and the conditional info is the keys and values. So the Unet can **query** the fucking conditional data. That's fucking amazing. 





They mention that since pixel-space DMs are likelihood-based, they are more likely to spend way too much capacity modeling imperceptible details of the data. (page 1). linked an article in the reading list for this. They also mention high-frequency details. What does that mean?









## NoteLinks