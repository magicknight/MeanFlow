import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence


class SinusoidalPosEmb(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, time):
        half_dim = self.dim // 2
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return emb


class ResnetBlock(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x, time_emb):
        h = nn.GroupNorm(num_groups=32)(x)
        h = nn.silu(h)
        h = nn.Conv(features=self.dim, kernel_size=(3, 3), padding="SAME")(h)

        time_emb = nn.silu(time_emb)
        time_emb = nn.Dense(features=self.dim)(time_emb)
        h = h + time_emb[:, None, None, :]

        h = nn.GroupNorm(num_groups=32)(h)
        h = nn.silu(h)
        h = nn.Conv(features=self.dim, kernel_size=(3, 3), padding="SAME")(h)

        return h + x


class Attention(nn.Module):
    dim: int
    heads: int = 4

    @nn.compact
    def __call__(self, x):
        head_dim = self.dim // self.heads
        scale = head_dim**-0.5
        b, h, w, c = x.shape

        qkv = nn.Conv(features=self.dim * 3, kernel_size=(1, 1))(x)
        qkv = qkv.reshape(b, h * w, self.heads, 3 * head_dim)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        sim = jnp.einsum("b i h d, b j h d -> b h i j", q, k) * scale
        attn = nn.softmax(sim, axis=-1)

        out = jnp.einsum("b h i j, b j h d -> b i h d", attn, v)
        out = out.reshape(b, h, w, c)
        return nn.Conv(features=self.dim, kernel_size=(1, 1))(out) + x


class UNet(nn.Module):
    dim: int
    channels: int
    dim_mults: Sequence[int]
    num_res_blocks: int

    @nn.compact
    def __call__(self, z, r, t):
        # 1. Time embeddings for r and t
        time_dim = self.dim * 4
        r_emb = SinusoidalPosEmb(self.dim)(r)
        t_emb = SinusoidalPosEmb(self.dim)(t)

        time_emb = jnp.concatenate([r_emb, t_emb], axis=-1)
        time_emb = nn.Sequential([nn.Dense(features=time_dim), nn.gelu, nn.Dense(features=time_dim)])(time_emb)

        # 2. Network architecture
        init_conv = nn.Conv(features=self.dim, kernel_size=(7, 7), padding="SAME")(z)
        x = init_conv

        skips = [x]

        # Downsampling
        dims = [self.dim] + [self.dim * m for m in self.dim_mults]
        for i, (dim_in, dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            is_last = i == (len(dims) - 2)
            for _ in range(self.num_res_blocks):
                x = ResnetBlock(dim=dim_in)(x, time_emb)
                skips.append(x)
            x = Attention(dim=dim_in)(x)
            if not is_last:
                x = nn.Conv(features=dim_out, kernel_size=(4, 4), strides=(2, 2), padding="SAME")(x)

        # Middle
        mid_dim = dims[-1]
        x = ResnetBlock(dim=mid_dim)(x, time_emb)
        x = Attention(dim=mid_dim)(x)
        x = ResnetBlock(dim=mid_dim)(x, time_emb)

        # Upsampling
        for i, (dim_in, dim_out) in enumerate(zip(dims[::-1][:-1], dims[::-1][1:])):
            is_last = i == (len(dims) - 2)
            for _ in range(self.num_res_blocks + 1):
                x = jnp.concatenate([x, skips.pop()], axis=-1)
                x = ResnetBlock(dim=dim_out)(x, time_emb)
            x = Attention(dim=dim_out)(x)
            if not is_last:
                x = nn.ConvTranspose(features=dim_in, kernel_size=(4, 4), strides=(2, 2), padding="SAME")(x)

        # Final layer
        final_conv = nn.Conv(features=self.channels, kernel_size=(1, 1))(x)
        return final_conv
