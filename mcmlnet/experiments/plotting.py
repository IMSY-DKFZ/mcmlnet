import os

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from matplotlib import cm
from matplotlib.colors import to_hex

from mcmlnet.experiments.utils import polynomial_func

load_dotenv()

dash_styles = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"]


def get_simulation_color(sim_name: str) -> str:
    """Get the color associated with a simulation name."""
    colors = [to_hex(c) for c in cm.get_cmap("tab10").colors]
    color_map = {
        "generic_sims": colors[0],
        "jacques_sims": colors[1],
        "tsui_sims": colors[2],
        "manoj_sims": colors[3],
        "lan_sims": colors[4],
        "jacques_sims_artificial": colors[5],
        "generic_sims_8400k": colors[6],
        "generic_sims_420k": colors[7],
        "generic_sims_21k": colors[8],
    }
    return color_map.get(sim_name, "#000000")  # type: ignore [no-any-return]


def get_simulation_color_with_alpha(sim_name: str, alpha: float) -> str:
    """Get the color associated with a simulation name with alpha."""
    color = get_simulation_color(sim_name)
    rgb_tuple = tuple(int(color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))

    return f"rgba({rgb_tuple[0]}, {rgb_tuple[1]}, {rgb_tuple[2]}, {alpha})"


def construct_filename(
    config: dict[str, dict[str, str | bool | int | list[float]]],
    plot_name: str,
) -> str:
    # Construct filename for recall plots
    specular_suffix = "_specular" if config["data"]["specular"] else ""
    overlap_suffix = "_minimal_overlap" if config["recall"]["minimal_overlap"] else ""
    surrogate_suffix = "_surrogate" if config["recall"]["use_surrogate_model"] else ""
    ablation_suffix = (
        "_ablation" if config["recall"]["do_surrogate_model_ablation"] else ""
    )
    aggregation = config["recall"]["aggregation_level"]
    metric = config["recall"]["metric"]
    filename = (
        f"{config['data']['real_data']}{surrogate_suffix}{ablation_suffix}_{plot_name}_"
        f"{aggregation}_{metric}{specular_suffix}{overlap_suffix}.svg"
    )
    return os.path.join(os.environ["plot_save_dir"], filename)


def format_param_latex(param: float, decimals: int = 2) -> str:
    """
    Math formatting of resulting parameters, including rounding and exponent options.

    Args:
        param: float, parameter to format.
        decimals: int, determines the number of decimals to keep.
            decimals < 0 additionally rounds to 10**abs(decimals)
            and shows no decimals e.g., -1 is rounded to the nearest 10.

    Returns:
        str, math-formatted parameter string.
    """
    if decimals < 0:
        param = round(param, decimals)
    abs_val = abs(param)
    # Do not display decimals when using negative 'decimals' argument
    frac_digits = 0 if decimals < 0 else decimals

    # Add base and exponent for very small or very large numbers
    if abs_val < 0.01 or abs_val >= 10000:
        exp = int(np.floor(np.log10(abs(param))))
        mantissa = param / (10**exp)
        return rf"${mantissa:.{frac_digits}f} \cdot 10^{{{exp}}}$"
    else:
        return f"${param:.{frac_digits}f}$"


def plot_scaling_fit(
    ax: plt.Axes,
    x_data: np.ndarray,
    y_data: np.ndarray,
    popt: np.ndarray,
    perr: np.ndarray,
    label: str,
    color: str,
    decimals: dict[str, int] | None = None,
) -> None:
    """
    Plot scaling fit with uncertainty bands.

    Args:
        ax: Matplotlib axes to plot on
        x_data: Original x data points
        y_data: Original y data points
        popt: Fitted parameters
        perr: Parameter errors
        label: Label for the plot
        color: Color for the plot
        decimals: Dict of parameter: n_decimals
    """
    if decimals is None:
        decimals = {"a": -1, "c": 1}

    # Generate the fit and uncertainty bands
    x_fit = np.linspace(np.min(x_data), np.max(x_data), 100)
    y_fit = polynomial_func(x_fit, *popt)
    y_fit_min = polynomial_func(x_fit, *(popt - perr))
    y_fit_max = polynomial_func(x_fit, *(popt + perr))

    # Plot the fit and uncertainty bands
    ax.plot(
        x_fit,
        y_fit,
        linestyle="--",
        label=(
            rf"Poly. Fit: {format_param_latex(popt[0], decimals["a"])} "
            rf"$\cdot x^{{{popt[1]:.2f}}}$ + "
            rf"{format_param_latex(popt[2], decimals["c"])}"
        ),
        color=color,
    )
    ax.fill_between(x_fit, y_fit_min, y_fit_max, alpha=0.5, color=color)

    # Plot the data
    ax.scatter(
        x_data,
        y_data,
        color=color,
        label=label,
        s=10,
        marker="x",
    )


def setup_plot_style() -> None:
    """Set up consistent plot styling for all figures."""
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["axes.labelsize"] = 8
    plt.rcParams["legend.fontsize"] = 8
    plt.rcParams["xtick.labelsize"] = 8
    plt.rcParams["ytick.labelsize"] = 8

    # Additional LaTeX settings
    plt.rcParams["font.serif"] = [
        "Times New Roman",
        "DejaVu Serif",
        "Computer Modern Roman",
    ]
    plt.rcParams["mathtext.fontset"] = "cm"


def add_theoretical_bounds(
    ax: plt.Axes,
    mle_value: float,
    ci_bounds: tuple[float, float] | None = None,
    color: str = "navy",
    label_prefix: str = "Theoretical MC Standard Deviation",
) -> None:
    """
    Add theoretical bounds to a plot.

    Args:
        ax: Matplotlib axes to plot on
        mle_value: MLE value for horizontal line
        ci_bounds: Optional CI bounds for shaded region
        color: Color for the bounds
        label_prefix: Prefix for the label
    """
    # Add theoretical lower bound
    ax.axhline(
        mle_value,
        color=color,
        label=rf"{label_prefix} $\hat{{\sigma}} / n_{{phot.}}$",
    )

    # Add CI bounds if provided
    if ci_bounds is not None:
        ax.axhspan(
            ci_bounds[0],
            ci_bounds[1],
            color="black",
            label="Empirical Lower MAE Bound (95% CI)",
        )
