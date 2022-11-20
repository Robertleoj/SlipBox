---
tags: 
 - deep # deep, diffusion, etc
 - attention
 - AI

creation date: 2022-11-18 12:58
modification date: Friday 18th November 2022 12:58:27
---
Reference: [[Attention is all you need]]

An improvement over [[The General Attention Mechanism]] is the multi-headed attention mechanism. 

The idea is to linearly project the queries, keys and values a few ways with learned projections, and then apply [[Scaled dot-product attention]] to each querys-keys-values combination, and finally combine the information from the attentions for a final output. 

What this aims to achieve is to allow the model to attend to the data from different representation subspaces and perspectives, as a "single-headed" attention will only allow the data to attend to very few places in the input at once due to the weighed average of the values. 

Formally, we linearly project the queries, keys and values $h$ times into dimensions $d_k$, $d_k$, and $d_v$, respectively. Then, the multi-headed attention is computed as follows:
$$
\opname{MultiHead}(\vc{Q}, \vc{K}, \vc{V}) 
= \opname{Concat}(\vc{H}_{i},\ldots,\vc{H}_{h})\vc{W}^O
$$
Where the heads are
$$
\vc{H}_{i} 
= \opname{attention}(
   \vc{Q}\vc{W}_i^Q, 
   \vc{K}\vc{W}_i^K,
   \vc{V}\vc{W}_i^V
)
$$
All the matrices $\vc{W}$ are learned weights. Their shapes are
$$\begin{align}
\vc{W}_i^Q &\in \RR^{d_{model}\times d_k}\\
\vc{W}_i^K &\in \RR^{d_{model} \times d_k}\\
\vc{W}_i^V &\in \RR^{d_{model} \times d_v}\\
\vc{W}^O &\in \RR^{h\cdot d_v \times d_{model}}
\end{align}$$
A common thing to do is to use $d_k = d_v = d_{model} / h$. This leads to a similar computational complexity as regular dot-product attention.

Here is an illustration:
![[media/Pasted image 20221118154528.png]]


## NoteLinks
[[Analysis of multi-head attention implementation for images]]














