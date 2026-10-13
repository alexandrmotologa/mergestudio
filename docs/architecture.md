# Architecture and Mathematical Specifications

MergeStudio implements mathematical tensor merging, layer surgery, and memory mapped streaming for neural language models.

## 1. Spherical Linear Interpolation (SLERP)

Linear interpolation between two high dimensional weight vectors $v_0$ and $v_1$ ($v(t) = (1-t)v_0 + tv_1$) reduces the magnitude of the resulting vector in high dimensional spaces, causing weight dilution.

SLERP interpolates along the great circle on the hypersphere, maintaining geometric angular velocity:

$$\theta = \arccos\left(\frac{\langle v_0, v_1 \rangle}{\|v_0\| \|v_1\|}\right)$$

$$v(t) = \frac{\sin((1-t)\theta)}{\sin\theta} \hat{v}_0 + \frac{\sin(t\theta)}{\sin\theta} \hat{v}_1$$

When magnitude preservation is enabled, the unit directional vector is scaled by the interpolated norm:

$$\|v(t)\| = (1-t)\|v_0\| + t\|v_1\|$$

$$v_{final}(t) = \|v(t)\| \cdot v(t)$$

### Collinear and Degenerate Fallback

If $\sin\theta \to 0$ or $\cos\theta \approx \pm 1$, standard trigonometric evaluation produces division by zero. MergeStudio detects near collinear vectors within tolerance $\epsilon = 10^{-7}$ and falls back to standard linear interpolation.

---

## 2. TIES-Merging (TRIM, ELECT SIGN, MERGE)

TIES resolves parameter interference across fine tuned models (Yadav et al., 2023):

1. **Task Vector Calculation**:
   $$\tau_i = \theta_i - \theta_{base}$$
2. **Trim (Top-k Density)**:
   For each task vector, keep the top $k\%$ elements with the largest absolute magnitude and set the rest to 0:
   $$\hat{\tau}_i = \tau_i \odot \mathbb{I}(|\tau_i| \ge \text{quantile}_{1-k}(|\tau_i|))$$
3. **Elect Sign**:
   Calculate the majority direction for each parameter:
   $$\gamma = \operatorname{sgn}\left(\sum_{i=1}^N \hat{\tau}_i\right)$$
4. **Disjoint Merge**:
   Mask out discordant deltas ($\operatorname{sgn}(\hat{\tau}_i) \ne \gamma$), then compute the average across models with non zero contributions:
   $$\theta_{merged} = \theta_{base} + \lambda \cdot \tau_{merged}$$

---

## 3. DARE (Drop And REscale)

DARE (Yu et al., 2024) randomly drops fine tuned parameter deltas using a Bernoulli mask with drop rate $p$, then rescales surviving deltas:

$$\tau_i^{(DARE)} = \frac{1}{1 - p} (\tau_i \odot M), \quad M \sim \operatorname{Bernoulli}(1 - p)$$

This preserves the mathematical expectation $\mathbb{E}[\tau_i^{(DARE)}] = \tau_i$ while eliminating up to 90% of parameter shifts, mitigating catastrophic interference.

---

## 4. Zero-RAM Streaming Engine

```
[ safetensors on Disk ] --( mmap )--> [ Single Tensor Read ]
                                            |
                                            v
                                 [ Math Merge Buffer ]
                                            |
                                            v
[ Safetensors Shard Writer ] <--( Stream )--+
```

MergeStudio maps files into virtual memory using `safetensors.safe_open`. Only the parameters corresponding to the currently processed key are brought into RAM. After merging, the resulting tensor is passed to `ShardedSafetensorsWriter`, which writes to temporary shards and releases references, keeping memory usage constant regardless of model depth.
