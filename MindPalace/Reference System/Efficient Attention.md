---
tags: 
 - deep # deep, diffusion, etc
 - attention
 - AI

type: paper
title: "Efficient attention"
summary: "Attention with linear complexity"

link: "https://arxiv.org/pdf/1812.01243.pdf"

creation date: 2022-11-20 11:26
modification date: Sunday 20th November 2022 11:27:32
---

The authors describe a Linear Attention Mechanism which is approximate to the dot-product attention, but with less complexity. 

In the abstract, they claim that their formulation is equivalent to dot-product attention, but it is not in the softmax case, and they admit this later, saying that it is equivalent, "with a caveat". 

I read this because I am finding this implemented in all diffusion models I can find. 

## NoteLinks