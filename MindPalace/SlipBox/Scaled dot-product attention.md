---
tags: 
 - deep # deep, diffusion, etc
 - attention
 - AI

creation date: 2022-11-18 13:00
modification date: Friday 18th November 2022 13:00:38
---
Reference: [[Attention is all you need]]

Scaled dot-product attention is a modification of the regular dot-product attention ([[The General Attention Mechanism]]), where the only difference is simply scaling the scores before turning them into weights. Let $d_k$ be the dimension of the keys and the queries, and let $d_v$ be the dimension of the values. Then scaled dot-product attention is
$$\begin{align}
\attention(\vc{Q},\vc{K},\vc{V} ) = \softmax(\frac{\vc{Q}\vc{K}^\top}{\sqrt{d_k}})\vc{V}
\end{align}$$
The regular dot-product attention is
$$\begin{align}
\attention(\vc{Q},\vc{K},\vc{V} ) = \softmax(\vc{Q}\vc{K}^\top)\vc{V}
\end{align}$$
So the only modification is the scaling of the scores by $1/\sqrt{d_k}$.  The scaling empirically seems to outperform regular dot-product attention. The inventors give the explanation that when the dimension of the keys grows large, the magnitude of the dot-products grows large as well, causing the softmax to have small gradients, so scaling by the dimension counteracts this. 

It seems like this specific scaling value is not special at all, and that we could use many functions that shrink with $d_k$. The authors most likely tested some functions, and found that this one worked well enough. 

## NoteLinks
