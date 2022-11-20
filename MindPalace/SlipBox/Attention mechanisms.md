---
summary: ""
tags: 
 - deep # deep, diffusion, etc
 - attention
 - AI
 
creation date: 2022-11-18 10:53
modification date: Friday 18th November 2022 10:53:00
---

Reference: [[Attention mechanism from scratch]]

Attention mechanisms in deep learning are a method to allow your network to selectively focus on some aspect of the input. This seems to be done by computing weights for each input, and allowing the inputs through the layer scaled by their weights, where you can optionally aggregate. 

There are many different attention mechanisms.

A simple one which also happens to be the first one is [[Bahdanau's Attention mechanism]]. However, it only works when the decoder output is sequentially related.

A generalization is the [[The General Attention Mechanism]], often called the dot-product attention. It can be used for any seq2seq application, even if there is not a sequential nature in the input, nor the output. A modification of it is [[Scaled dot-product attention]].

An improvement is [[Multi-headed attention]], used in the transformer architecture.

A problem is that the dot-product attention runs in quadratic memory and space. To mitigate this, the [[Efficient Attention Mechanism]] was introduced, which is an approximate dot-product attention, but has way less complexity. 

## NoteLinks
[[Analysis of multi-head attention implementation for images]]






