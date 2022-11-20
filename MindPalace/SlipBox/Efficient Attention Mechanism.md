---
summary: "An attention mechanism that runs in linear complexity"
tags: 
 - deep # deep, diffusion, etc
 - attention
 - AI

creation date: 2022-11-20 11:34
modification date: Sunday 20th November 2022 11:34:25
---
Reference: [[Efficient Attention]]

The dot-product attention runs in quadratic time and space, which is quite large. The Efficient Attention Mechanism  (EAM) claims to have created an equivalent attention mechanism that runs in linear time instead, and has the same representation power. 

> This raises a question. If EAM is more efficient and is just as good, why would one ever use regular dot-product attention?

EAM is based on the simple observation of matrix multiplication associativity. Regular dot-product attention is
$$
D(\vc{Q}, \vc{K}, \vc{V}) = \rho(\vc{Q}\vc{K}^\top) \vc{V}
$$
Where $\rho$ is a normalization function applied row-wise. Two functions are named: a simple scaling one:
$$
\rho(\vc{X}) = \frac{\vc{X}}{n}
$$
and the row-wise softmax. 

Disregarding $\rho$ for a moment, we have $(\vc{Q}\vc{K}^\top)\vc{V}$. In this case, we can use associativity to equate this to $\vc{Q}(\vc{K}^\top \vc{V})$. Since $\vc{Q}, \vc{K} \in \RR^{n \times d_k}$, we know that $\vc{Q}\vc{K}^\top$ is $n\times n$, leading to the quadratic complexity. However, $\vc{K}^\top\vc{V}$ is only $d_k\times d_v$, and so the largest matrix we calculate in $\vc{Q}(\vc{K}^\top \vc{V})$ is $n\times \max(d_k, d_v)$, leading to linear complexity. 

From this observation, EAM is stated as
$$
E(\vc{Q}, \vc{K}, \vc{V}) = \rho_q(\vc{Q})(\rho_k(\vc{K})^\top \vc{V})
$$
where $\rho_q$ and $\rho_k$ are two normalization functions. The paper names two choices for them. The first is a scaling one:
$$
\rho_q(\vc{X}) = \rho_k(\vc{X}) = \frac{\vc{X}}{\sqrt{n}}
$$
and the softmax:
$$\begin{align}
\rho_q(\vc{X}) &= \sigma_{\row}(\vc{X})\\
\rho_k(\vc{X}) &= \sigma_{\col}(\vc{X})
\end{align}$$
Now, in the scaling one, it is obvious that the dot-product attention and EAM are exactly equivalent. However, in the softmax case, they are not. The paper states that the only important property of the softmax in this case is that the rows of $\sigma_{\row}(\vc{Q}\vc{K}^\top)$ are normalized (sum to one). They claim that $\sigma_{\row}(\vc{Q})\sigma_{\col}(\vc{K})^\top$ shares this property without proof. 

I'll prove it here. Let $\vc{Q}' = \sigma_{\row}(\vc{Q})$,  $\vc{K}'= \sigma_{\col}(\vc{K})$, and $\vc{M} = \vc{Q}'\vc{K}'^\top$ Then 
$$
\sum_{j}\vc{Q'}_{i, j} = 1
$$
and 
$$
\sum_j \vc{K}'_{j, i} = 1
$$
So we have
$$\begin{align}
\sum_{k}\vc{M}_{i, k} &= \sum_{k} \sum_{j} \vc{Q}'_{i, j} \vc{K}'^\top_{j, k}\\
&= \sum_{k} \sum_{j} \vc{Q}'_{i, j} \vc{K}'_{k, j}\\
&= \sum_{j} \vc{Q}'_{i, j}\sum_k \vc{K}'_{k, j}\\
&= \sum_{j} \vc{Q}'_{i, j} = 1
\end{align}$$
which proves the statement. 

This shows that the EAM is pretty close in meaning to the dot-product attention. 

My intuition is that since in dot-product attention, we are producing a $n \times d$ matrix, we should not need to produce an intermediary $n \times n$ matrix, since the operations are not very complicated. Since EAM seems to exhibit most of the features of dot-product attention, it should be equally expressive in power, which they empirically show (which is not absolute proof of course, which would need to be mathematical). But this is convincing enough for my purposes. 

Now we look at an interpretation of EAM. Disregarding the ideas of regular dot-product attention, we interpret the columns $\vc{k}_j^\top$ of $\vc{K}$ as $d_k$  global attention maps, each corresponding to a semantic aspect of the input. It might be comfortable to think of the transposed $\vc{V}^\top \vc{k}_j$: the global attention maps $\vc{k}_j^\top$ are weights for a linear combination of the values, so each of them can be an attention map to extract some semantic aspect of $\vc{V}$. 

Then each of the $\vc{k}_j^T$ extract a global context vector $\vc{g}_j$, the result of the linear combination of the values. Note that $\vc{g}_j \in \RR^{d_v}$, and there are $d_k$ of them. To then extract the final output of the attention, we use the query vectors as weights over the $\vc{g}_j$. This means that the queries are placing weights on which semantic aspects of the values they want to pay attention to. 

To summarize: the keys are attention maps over the values, extracting semantic aspects of them. They produce global context vectors, each representing some semantic aspect of the values. Then the query vectors choose which of the semantic aspects they want to pay attention to, using again a linear combination of the global context vectors. 

Here my intuition tells me that this is simply more efficient than the dot-product one, but should have equal expressive power. In the dot-product attention, each query has to make its own semantic representation of the values. This seems redundant, and EAM solves that redundancy by first extracting semantic aspects, and then querying the semantic aspects.

## NoteLinks
[[The General Attention Mechanism]] (the dot-product attention)