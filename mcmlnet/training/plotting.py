"""
This module contains functions for visualizing training results,
such as parameter marginals and spectra.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from mcmlnet.utils.logging import setup_logging
from mcmlnet.utils.process_spectra import coeff_of_variation
from mcmlnet.utils.tensor import TensorType, array_conversion_decorator

logger = setup_logging(level="info", logger_name=__name__)


@array_conversion_decorator
def display_parameter_marginals(
    real: TensorType, generated: TensorType | None = None
) -> plt.figure:
    """
    Create matplotlib histogram of input parameter marginals (of every tissue layer).

    Args:
        real: Original parameter values.
        generated: Synthesized parameter values (optional).

    Returns:
        Histogram of input parameter marginals.

    Raises:
        AssertionError: If the shapes of real and generated do not match.
    """
    # Shape assertions
    if generated is None:
        generated = real
    if real.shape != generated.shape:
        raise AssertionError(
            "Real and generated data must have same shape, "
            f"but found {real.shape} and {generated.shape}!"
        )

    # Determine grid layout (for default three layer model)
    if real.shape[1] % 3 == 0:
        n_params_per_layer = real.shape[1] // 3
        rows = 3
    else:
        n_params_per_layer = real.shape[1]
        rows = 1

    # Create subplots
    fig, axs = plt.subplots(
        figsize=(rows * n_params_per_layer, n_params_per_layer),
        nrows=rows,
        ncols=n_params_per_layer,
        sharex=False,
        sharey=False,
    )

    # Plot histograms
    for i in range(n_params_per_layer):
        for j in range(rows):
            # Determine which axis to use
            if rows > 1 and n_params_per_layer > 1:
                ax = axs[j, i]
            elif n_params_per_layer > 1:
                ax = axs[i]
            else:
                ax = axs

            # Get data for this parameter
            param_idx = i * rows + j if rows > 1 else i
            real_data = real[:, param_idx]
            gen_data = generated[:, param_idx]

            # Check if data has any variation
            real_range = real_data.max() - real_data.min()
            gen_range = gen_data.max() - gen_data.min()

            # Plot histograms
            bins = max(10, len(real_data) // 10)

            # Real data histogram
            if real_range != 0:
                heights, edges = np.histogram(real_data, bins=bins)
                centers = (edges[:-1] + edges[1:]) / 2
                ax.bar(
                    centers,
                    heights,
                    width=(edges[1] - edges[0]),
                    align="center",
                    color="tab:blue",
                    label="Real",
                    alpha=0.7,
                )

            # Generated data histogram
            if gen_range != 0:
                heights, edges = np.histogram(gen_data, bins=bins)
                centers = (edges[:-1] + edges[1:]) / 2
                ax.bar(
                    centers,
                    heights,
                    width=(edges[1] - edges[0]),
                    align="center",
                    color="tab:red",
                    label="Generated",
                    alpha=0.7,
                )

            if i == 0 and j == 0:  # Add legend to first subplot only
                ax.legend()

    plt.tight_layout()
    return fig


@array_conversion_decorator
def display_spectra(
    real: TensorType,
    generated: TensorType,
    n_rows: int = 1,
    n_cols: int = 5,
) -> plt.figure:
    """
    Plot reflectance spectra for visual assessment during training.

    Args:
        real: Original spectra.
        generated: Synthesized spectra.
        n_rows: Number of rows in the resulting grid.
        n_cols: Number of columns in the resulting grid.

    Returns:
        Figure containing the image spectra grid.
    """
    # Content and shape assertion
    if len(real) != n_rows * n_cols:
        raise AssertionError(
            "Given tensors must contain enough entries to fill a grid of n_rows*n_cols."
        )
    if real.ndim != 2 or generated.ndim != 2:
        raise AssertionError(
            "Given tensors are required to be a batch of spectra of shape [bs, lambda]."
        )
    if real.shape != generated.shape:
        raise AssertionError("'real' and 'generated' tensors do not match in shape.")

    # add scale and create figure
    scale = 5
    fig, ax = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(n_cols * scale, n_rows * scale)
    )
    x_vals = range(real.shape[1])

    for index, _ in enumerate(real):
        x = int(index // n_cols)
        y = int(index % n_cols)

        if n_rows == 1:
            # plot real spectra
            ax[y].plot(x_vals, real[index], label="original", c="tab:blue")
            # plot generated spectra
            ax[y].plot(x_vals, generated[index], label="generated", c="tab:red")
            ax[y].legend()
        else:
            # plot real spectra
            ax[x, y].plot(x_vals, real[index], label="original", c="tab:blue")
            # plot generated spectra
            ax[x, y].plot(x_vals, generated[index], label="generated", c="tab:red")
            ax[x, y].legend()

    return fig


@array_conversion_decorator
def plot_coeff_of_variation_vs_relative_error(
    gen: torch.Tensor, output: torch.Tensor
) -> plt.Figure:
    """
    Plot the theoretical coefficient of variation vs. the relative error
    of the generated reflectance.

    Args:
        gen: Generated reflectance data.
        output: Output reflectance data.

    Returns:
        Figure containing the plot of coefficient of variation vs. relative error.
    """
    # Define the number of photons
    n_photons = 10**6

    # Clip gen and warn if clipping was necessary
    min_val, max_val = 1e-8, 1
    affected = (gen < min_val) | (gen > max_val)
    if np.any(affected):
        logger.warning(
            f"Clipping {np.sum(affected)} values in 'gen' to the range "
            f"[{min_val}, {max_val}]"
        )
    gen = np.clip(gen, min_val, max_val)

    # Get the relative error of the reflectance
    rel_error = 100 * np.abs(output - gen) / output

    # Compute the boundaries for the return probability of photons (reflectance)
    p_low = min(gen.min(), output.min())
    p_high = min(gen.max(), output.max())
    p = torch.linspace(p_low, p_high, 1000)

    fig = plt.figure()
    # Plot the relative error as a hexbin plot against the reflectance values
    plt.hexbin(
        gen.flatten() + 1e-8,
        rel_error.flatten() + 1e-8,
        xscale="log",
        yscale="log",
        bins="log",
        marginals=True,
        gridsize=100,
        cmap="Blues",
    )
    # Plot the coefficient of variation for different photon counts
    plt.plot(
        p.numpy(),
        100 * coeff_of_variation(p, n_photons),
        color="red",
        linestyle="--",
        label="Coefficient of Variation 1 Mio Photons",
    )
    plt.plot(
        p.numpy(),
        100 * coeff_of_variation(p, 100 * n_photons),
        color="gray",
        linestyle="--",
        label="Coefficient of Variation 100 Mio Photons",
    )
    # plot 1% relative error line
    plt.plot(
        p.numpy(),
        [1.0] * len(p),
        color="black",
        linestyle="--",
        alpha=0.8,
        label="1% Relative Error",
    )
    plt.xlabel("Monte Carlo Reflectance Values")
    plt.ylabel("Surrogate Model Relative Error [%]")
    plt.title(
        "Surrogate Model Relative Error\n"
        f"Mean {rel_error.mean():.2f}%, 2.5/97.5% Percentile "
        f"{np.percentile(rel_error, 2.5):.2f}/{np.percentile(rel_error, 97.5):.2f}%"
    )
    plt.legend(loc="lower left")

    return fig
