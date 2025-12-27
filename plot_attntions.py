import math
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_2d_tensor(
    ax: plt.Axes,
    mat2d,
    *,
    title=None,
    vmin=None,
    vmax=None,
    cmap="viridis",
    show_colorbar=False,
    show_ticks=True,
    tick_step=20,
    tick_fontsize=6,
):
    """
    Reusable middle function: plot a single 2D tensor/matrix with token index ticks.
    """
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    if isinstance(mat2d, torch.Tensor):
        mat2d = mat2d.detach()
        if mat2d.is_cuda:
            mat2d = mat2d.cpu()
        mat2d = mat2d.float().numpy()
    else:
        mat2d = np.asarray(mat2d)

    im = ax.imshow(mat2d, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)

    T_y, T_x = mat2d.shape

    if show_ticks:
        xticks = np.arange(0, T_x, tick_step)
        yticks = np.arange(0, T_y, tick_step)

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(xticks, fontsize=tick_fontsize)
        ax.set_yticklabels(yticks, fontsize=tick_fontsize)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    if title:
        ax.set_title(title, fontsize=9)

    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)



def _auto_grid(num_heads: int) -> Tuple[int, int]:
    """
    Choose a near-square grid (rows, cols) for heads.
    """
    cols = math.ceil(math.sqrt(num_heads))
    rows = math.ceil(num_heads / cols)
    return rows, cols


def save_vit_attention_heads_per_layer(
    attn: torch.Tensor,
    out_dir: Union[str, Path],
    *,
    prefix: str = "layer",
    ext: str = "png",
    batch_index: int = 0,
    grid: Optional[Tuple[int, int]] = None,
    dpi: int = 200,
    cmap: str = "viridis",
    robust_percentile: Tuple[float, float] = (1.0, 99.0),
    share_color_scale_within_layer: bool = True,
    log1p: bool = False,
    show_colorbar: bool = False,
) -> None:
    """
    Inputs:
      - attn: [num_layers, batch_size, num_heads, num_tokens, num_tokens] tensor
      - out_dir: output directory path

    Output:
      - For each layer, saves one image in out_dir. Each image contains all heads
        laid out in a grid, with spacing between subplots.

    Notes:
      - robust_percentile controls vmin/vmax via percentiles (helps avoid "all black" plots).
      - share_color_scale_within_layer=True makes heads comparable within the same layer.
      - log1p=True applies log(1+x) before plotting (useful if values are very peaky).
    """
    if not isinstance(attn, torch.Tensor):
        raise TypeError("attn must be a torch.Tensor")
    if attn.ndim != 5:
        raise ValueError(f"attn must have 5 dims [L,B,H,T,T], got shape {tuple(attn.shape)}")

    L, B, H, T1, T2 = attn.shape
    if T1 != T2:
        raise ValueError(f"Expected square attention matrices, got {T1}x{T2}")

    if not (0 <= batch_index < B):
        raise ValueError(f"batch_index out of range: {batch_index} for batch_size={B}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Work on CPU float for plotting
    A = attn.detach()
    if A.is_cuda:
        A = A.cpu()
    A = A.float()

    # Fix batch
    A = A[:, batch_index]  # [L, H, T, T]

    if grid is None:
        rows, cols = _auto_grid(H)
    else:
        rows, cols = grid
        if rows * cols < H:
            raise ValueError(f"grid {grid} too small for num_heads={H}")

    p_lo, p_hi = robust_percentile
    if not (0.0 <= p_lo < p_hi <= 100.0):
        raise ValueError("robust_percentile must satisfy 0 <= lo < hi <= 100")

    for layer in range(L):
        layer_attn = A[layer]  # [H, T, T]

        if log1p:
            layer_plot = torch.log1p(torch.clamp(layer_attn, min=0))
        else:
            layer_plot = layer_attn

        if share_color_scale_within_layer:
            flat = layer_plot.reshape(-1).numpy()
            vmin = float(np.percentile(flat, p_lo))
            vmax = float(np.percentile(flat, p_hi))
        else:
            vmin = vmax = None

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(cols * 3.0, rows * 3.0),
            squeeze=False,
        )

        # Create visible gaps between heads
        fig.subplots_adjust(wspace=0.25, hspace=0.30)

        for h in range(rows * cols):
            ax = axes[h // cols][h % cols]
            if h < H:
                head_mat = layer_plot[h]  # [T, T]

                # If not sharing scale, compute per-head robust vmin/vmax
                if not share_color_scale_within_layer:
                    head_flat = head_mat.reshape(-1).numpy()
                    hvmin = float(np.percentile(head_flat, p_lo))
                    hvmax = float(np.percentile(head_flat, p_hi))
                else:
                    hvmin, hvmax = vmin, vmax

                plot_2d_tensor(
                    ax,
                    head_mat,
                    title=f"Head {h}",
                    vmin=hvmin,
                    vmax=hvmax,
                    cmap=cmap,
                    tick_step = 200,  
                    show_colorbar=show_colorbar,
                )
            else:
                ax.axis("off")

        fig.suptitle(f"Layer {layer} | tokens={T1} | heads={H}", fontsize=12)

        out_path = out_dir / f"{prefix}_{layer:02d}.{ext}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

def plot_attentions(model, dataloader, output_dir):
    model.eval()
    device = next(model.parameters()).device

    attns = []
    for samples, _ in dataloader:
        samples = samples.to(device)
        attn = model.get_attentions(samples)
        attns.append(attn)
        
    for attn in attns:
        save_vit_attention_heads_per_layer(attn, output_dir)