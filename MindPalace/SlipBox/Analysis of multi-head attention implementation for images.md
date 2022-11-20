---
tags: 
 - deep # deep, diffusion, etc
 - diffusion
 - code
 - attention
 - AI

creation date: 2022-11-18 13:57
modification date: Friday 18th November 2022 13:57:48
---
Reference: [[DDPM github]]

The following definition of an attention class can be found in the file `denoising-diffusion-pytorch.py` in the reference:

```python
class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(
	        lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads),
	         qkv
	    )

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

```
This code is used as an attention layer at the bottom of a U-net for denoising diffusion for image generation. I want to understand this code, so let's analyze it to know what's going on. 

The code is implementing a [[Multi-headed attention]] with the [[Scaled dot-product attention]] as the attention in the multi-head. The input to the layer is a four-dimensional tensor $\vc{X}$, with dimentions $B \times C \times H \times W$. It is not immediately obvious how you would implement attention on this kind of input. 

Now, they seem to be using $d_k = d_v$, equal to the `dim_head` argument to the `__init__` function. For simplicity, let $d=d_k=d_v$. This explains the scaling factor, as `dim_head ** -0.5` is then equal to $1/\sqrt{d_k}$.  

Next, we see them storing `hidden_dim` as $d \cdot h$. This corresponds to the fact that we concatenate the outputs of each attention before it enters the last layer. 

We now look at the `forward` function. The first thing done is storing the dimensions of the input:
```python
b, c, h, w = x.shape
```

However, the next action performed is rather peculiar:
```python
qkv = self.to_qkv(x).chunk(3, dim = 1)
```
The input is passed through a convolutional layer, where the number of output channels is $d \cdot h \cdot 3$, following a chunking into three pieces over the channels, each piece with $d \cdot h$ channels. 

What we thus have so far is a tuple of three tensors of shape $B\times d \cdot h \times H \times W$. The name of the variable holding these clearly indicates that they represent $\vc{Q}$, $\vc{K}$, and $\vc{V}$. 

In the original paper [All you need is attention](Attention%20is%20all%20you%20need.md), the authors suggested linearly projecting these $h$ times. However, in this implementation, they do not produce $\vc{Q}$, $\vc{K}$, and $\vc{V}$ and then linearly project them in different ways, but instead use the convolution to create a tensor for each with $d \cdot h$ channels, and then split each of then into $h$ tensors. The learned convolution ensures that each of the splits is unique. 

In short, they create one transformations which creates the queries, keys and values at the same time in one tensor, which is then split. The next expression does exactly this:
```python
q, k, v = map(
	lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), 
	qkv
)
```
Each of the $B\times d \cdot h \times H \times W$ tensors are now split up into the individual heads. In addition, the height and width dimensions are combined, or flattened. Thus, each of the tensors in the output now has the dimension $B \times h \times d \times HW$.  

Now, we shall not operate between elements in the batch dimension: each data point in a batch must remain independent. Also, when we do the attention multiplication now, it must be done independent inside the $h$ axis. Thus the $d \times HW$ parts are what we will be operating on in the attention. 

You first index into a batch, then a head, and there you find your $\vc{Q}$, $\vc{K}$, or $\vc{V}$ matrix. Note however that in contrast to the example in the notes, the columns here are the individual elements, and thus instead of performing the multiplication $\vc{Q}\vc{K}^\top$, we perform $\vc{Q}^\top \vc{K}$.

Next the scaling factor is applied:
```python
q = q * self.scale
```
Linearity explains why this can be performed now:
$$
\left(\frac{1}{\sqrt{d}} \vc{Q}^\top\right)\vc{K} = \frac{\vc{Q}^\top\vc{K}}{\sqrt{d}}
$$
A reson to perform this now is that $\vc{Q}$ is smaller than $\vc{Q}^\top\vc{K}$, resulting in fewer multiplications. 

What remains now is to perform the matrix multiplication $\vc{Q}^\top \vc{K}$. This is done with the line
```python
sim = einsum('b h d i, b h d j -> b h i j', q, k)
```
As can be seen in the Einstein sum, the batch and head dimensions are left alone in the multiplication, and we only multiply the inner query and key matrices. We can also see that we are multiplying along the first axis of both matrices. This is exactly equivalent to transposing the $\vc{Q}$ matrix and then performing a matrix multiplication with $\vc{K}$.

After this operation, the `sim` variable represents the similarity scores. It has dimension $B \times h \times HW \times HW$, and the rows represent the scores for a query.

The next line performs the softmax of the similarity scores to produce the weight matrix
```python
attn = sim.softmax(dim = -1)
```
The weights for each query are along the rows, so we perform the softmax over the last dimension, which are exactly the rows. The variable `attn` corresponds to the $\vc{W}$ matrix in the general attention note. 

Finally, we obtain the weighted sum of the values. This means we need to compute $\vc{W}\vc{V}^\top$:
```python
out = einsum('b h i j, b h d j -> b h i d', attn, v)
```
We can see that in the einsum, we are summing over the row index of $\vc{W}$, and also in $\vc{V}$. Thus this is equivalent to computing $\vc{W}\vc{V}^\top$. 

Now `out` stores the output of each head of the attention, each of dimension $B \times h \times d \times HW$. We therefore want to now combine the heads, and expand the height and width dimensions again. This is done in the next line with a simple rearrange:
```python
out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
```
After this operation, the shape of `out` is $B \times h \cdot d \times H \times W$.  We finish off the layer with a convolution which returns the number of channels to what it was on input:
```python
return self.to_out(out)
```

## NoteLinks
[[The Einstein Sum]]