---
tags: 
 - deep # deep, diffusion, etc
 - attention
topics: 
 - "The first attention mechanism"
 
creation date: 2022-11-18 11:00
modification date: Friday 18th November 2022 11:00:49
---
Reference:  [[Attention mechanism from scratch]]

This is the first attention mechanism for deep learning, specifically for seq2seq applications, where we use an encoder and a decoder with RNNs, or any kind of recurrent structure. It works as follows:

1. Compute the alignment scores. If $\vc{h}_i$ are the encoder hidden states and $\vc{s}_{t - 1}$ the decoder's previous output, we compute the alignment score $e_{t, i} = a(\vc{s}_{t - 1}, \vc{h}_i)$,  which describes how much attention we want to put on $\vc{h}_i$ for computing $s_{t}$. Usually, $a$ is a neural network.
2. Turn these alignment scores into normalized weights with $\vc{\alpha}_t = \softmax(\vc{e}_t)$.
3. Create the context vector$$\vc{c}_t = \vc{H}\vc{\alpha}_t$$which can then be fed into the decoder to produce the next output. 

It can be seen that there is no inherent sequential connection between the encoder hidden states in this formulation. We feed each of them into $a$ separately, and then take a weighted sum of them, and there is no temporal relation between them. 

However, this formulation forces a sequential relation between the decoder outputs, which must be produced one after the other, as the previous output must be fed into $a$ to produce the next one. 


## NoteLinks
