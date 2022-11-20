---
tags: 
 - deep # deep, diffusion, etc
 - attention
 - AI
topics: 
 - "A generalized attention mechanism for seq2seq applications"

creation date: 2022-11-18 11:36
modification date: Friday 18th November 2022 11:36:44
---
Reference: [[Attention mechanism from scratch]]

In the General Attention mechanism, we have three components: the keys $\vc{K}$, the values $\vc{V}$, and the queries $\vc{Q}$. To describe it shortly, we compute weights for the values $\vc{V}$ for each query vector $\vc{q_i}$ based on the query and the keys $\vc{K}$, and then use the weights to compute a weightes sum of the values. 

Arbitrarily, let each of these matricies contain their values as rows, so the $i$th key is the $i$th row in $\vc{K}$, denoted $\vc{k}_i$

The keys are the component we use to compute the weights for the values, in accordance with the query. 

Now let's describe the operations formally.
1. For each query, compute a score for each key with $e_{i, j} = \vc{q}_i \cdot \vc{k}_i$. The scores are then a vector $\vc{e}_i =\vc{q}\vc{K}^\top$, and the whole score matrix is $\vc{E} = \vc{Q}\vc{K}^\top$. The score vector for $\vc{q}_i$ is the $i$th row of $E$. 
2. Create normalized weights for each query. The weight computed from a key and a query will be the weight of the value corresponding to the key in the weighted sum of the values for that specific query. For this, we use a row-wise softmax over $\vc{E}$, that is $\vc{W} = \softmax(\vc{E})$. 
3. For each query, the output of the attention is the weighte sum of the values, using the weights computed in the previous step. That is, $$
	\begin{align}
	\vc{r}_i &= \sum_{j = 1}^{N}w_{i, j}\vc{v}_j\\
	&= \vc{w}_{i} \vc{V}
	\end{align}$$So the result $\vc{r}_i$ is a row vector for each query. The complete result matrix is $\vc{R} = \vc{W}\vc{V}$
	
The final result can thus be summarized with the single equation $$
\begin{align}
\operatorname{attention}(\vc{Q}, \vc{K}, \vc{V}) 
&= \vc{W}\vc{V}\\
&= \softmax(\vc{E})\vc{V}\\
&= \softmax(\vc{Q}\vc{K}^\top)\vc{V}
\end{align}
$$Quite a simple equation, isn't it?

This mechanism is problematic in that it runs in quadratic time and space. [[Linear Attention Mechanism]] mitigates this. 

Edits:
* This attention mechanism is also called dot-product attention

## NoteLinks
[[Scaled dot-product attention]]
[[Efficient Attention Mechanism]]