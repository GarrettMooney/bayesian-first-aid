from typing import Any, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm


def bayes_prop_test(
    x: Union[list[int], np.ndarray],
    n: Union[list[int], np.ndarray] = None,
    comp_theta: Union[float, list[float]] = None,
    cred_mass: float = 0.95,
    n_iter: int = 15000,
    progress_bar: bool = True,
    random_seed: int = None,
) -> dict[str, Any]:
    """
    Bayesian alternative to a test of proportions.

    Parameters
    ----------
    x : array-like
        Vector of counts of successes or a 2D array with counts of successes and failures.
    n : array-like, optional
        Vector of counts of trials. Ignored if x is a 2D array.
    comp_theta : float or array-like, optional
        Fixed relative frequencies of success to compare with. Must be between 0 and 1.
    cred_mass : float, default=0.95
        The amount of probability mass that will be contained in reported credible intervals.
    n_iter : int, default=15000
        The number of iterations to run the MCMC sampling.
    progress_bar : bool, default=True
        Whether to display a progress bar during sampling.
    random_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        A dictionary containing the results of the analysis.
    """
    # Process inputs similar to R's prop.test
    if isinstance(x, np.ndarray) and x.ndim == 2:
        if x.shape[1] != 2:
            raise ValueError("If 'x' is a 2D array, it must have 2 columns")
        n = x.sum(axis=1)
        x = x[:, 0]
    elif n is None:
        raise ValueError("If 'x' is not a 2D array, 'n' must be provided")

    x = np.asarray(x)
    n = np.asarray(n)

    if len(x) != len(n):
        raise ValueError("'x' and 'n' must have the same length")

    if np.any(n <= 0):
        raise ValueError("Elements of 'n' must be positive")

    if np.any(x < 0):
        raise ValueError("Elements of 'x' must be non-negative")

    if np.any(x > n):
        raise ValueError("Elements of 'x' must not be greater than those of 'n'")

    # Process comp_theta (comparison values)
    if comp_theta is not None:
        if isinstance(comp_theta, (int, float)):
            comp_theta = np.repeat(comp_theta, len(x))
        else:
            comp_theta = np.asarray(comp_theta)
            if len(comp_theta) != len(x):
                raise ValueError(
                    "'comp_theta' must have the same length as 'x' and 'n'"
                )

        if np.any((comp_theta <= 0) | (comp_theta >= 1)):
            raise ValueError("Elements of 'comp_theta' must be in (0,1)")
    elif len(x) == 1:
        comp_theta = np.array([0.5])

    # If there's only one group, call bayes_binom_test instead
    if len(x) == 1:
        return bayes_binom_test(
            x[0], n[0], comp_theta[0], cred_mass, n_iter, progress_bar, random_seed
        )

    # Run MCMC
    mcmc_samples = prop_test(
        x, n, n_iter=n_iter, progress_bar=progress_bar, random_seed=random_seed
    )

    # Calculate statistics
    temp_comp_val = 0.5 if comp_theta is None else comp_theta
    stats = mcmc_stats(mcmc_samples, cred_mass=cred_mass, comp_val=temp_comp_val)

    # Calculate difference statistics
    theta_diff_samples = create_theta_diff_matrix(mcmc_samples)
    diff_stats = mcmc_stats(theta_diff_samples, cred_mass=cred_mass, comp_val=0)

    # Create result object
    result = {
        "x": x,
        "n": n,
        "comp_theta": comp_theta,
        "cred_mass": cred_mass,
        "mcmc_samples": mcmc_samples,
        "stats": pd.concat([stats, diff_stats]),
        "theta_diff_samples": theta_diff_samples,
    }

    return result


def bayes_binom_test(
    x: int,
    n: int,
    comp_theta: float = 0.5,
    cred_mass: float = 0.95,
    n_iter: int = 15000,
    progress_bar: bool = True,
    random_seed: int = None,
) -> dict[str, Any]:
    """
    Bayesian alternative to a binomial test.

    Parameters
    ----------
    x : int
        Number of successes.
    n : int
        Number of trials.
    comp_theta : float, optional
        Fixed relative frequency of success to compare with. Must be between 0 and 1.
    cred_mass : float, default=0.95
        The amount of probability mass that will be contained in reported credible intervals.
    n_iter : int, default=15000
        The number of iterations to run the MCMC sampling.
    progress_bar : bool, default=True
        Whether to display a progress bar during sampling.
    random_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        A dictionary containing the results of the analysis.
    """
    # Simple model for the binomial test
    with pm.Model() as model:
        # Prior
        theta = pm.Beta("theta", 1, 1)

        # Likelihood
        pm.Binomial("y", n=n, p=theta, observed=x)

        # Predicted values
        x_pred = pm.Binomial("x_pred", n=n, p=theta)

        # Sample
        trace = pm.sample(
            n_iter,
            random_seed=random_seed,
            progressbar=progress_bar,
            return_inferencedata=True,
        )

    # Convert to format similar to the main function
    mcmc_samples = {
        "theta": trace.posterior.theta.values.flatten(),
        "x_pred": trace.posterior.x_pred.values.flatten(),
    }

    # Calculate statistics
    stats = mcmc_stats(
        {"theta": mcmc_samples["theta"]}, cred_mass=cred_mass, comp_val=comp_theta
    )

    # Create result object
    result = {
        "x": x,
        "n": n,
        "comp_theta": comp_theta,
        "cred_mass": cred_mass,
        "mcmc_samples": mcmc_samples,
        "stats": stats,
    }

    return result


def prop_test(
    x: np.ndarray,
    n: np.ndarray,
    n_iter: int = 5000,
    progress_bar: bool = True,
    random_seed: int = None,
) -> dict[str, np.ndarray]:
    """
    Run the MCMC sampling for the proportion test model.

    Parameters
    ----------
    x : array-like
        Vector of counts of successes.
    n : array-like
        Vector of counts of trials.
    n_iter : int, default=5000
        The number of iterations to run the MCMC sampling.
    progress_bar : bool, default=True
        Whether to display a progress bar during sampling.
    random_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        A dictionary containing the MCMC samples.
    """
    with pm.Model() as model:
        # Priors
        theta = pm.Beta("theta", 1, 1, shape=len(x))

        # Likelihood
        pm.Binomial("y", n=n, p=theta, observed=x)

        # Predicted values
        x_pred = pm.Binomial("x_pred", n=n, p=theta)

        # Sample
        trace = pm.sample(
            n_iter,
            random_seed=random_seed,
            progressbar=progress_bar,
            return_inferencedata=True,
        )

    # Extract samples
    samples = {}
    for i in range(len(x)):
        samples[f"theta[{i + 1}]"] = trace.posterior["theta"].values[:, :, i].flatten()
        samples[f"x_pred[{i + 1}]"] = (
            trace.posterior["x_pred"].values[:, :, i].flatten()
        )

    return samples


def create_theta_diff_matrix(samples: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Create a matrix of pairwise differences between theta parameters.

    Parameters
    ----------
    samples : dict
        Dictionary of MCMC samples.

    Returns
    -------
    dict
        Dictionary of theta difference samples.
    """
    # Count the number of theta parameters
    n_groups = sum(1 for key in samples.keys() if key.startswith("theta["))

    # Generate all combinations
    theta_diffs = {}
    for i in range(1, n_groups):
        for j in range(i + 1, n_groups + 1):
            key = f"theta_diff[{i},{j}]"
            theta_diffs[key] = samples[f"theta[{i}]"] - samples[f"theta[{j}]"]

    return theta_diffs


def mcmc_stats(
    samples: dict[str, np.ndarray],
    cred_mass: float = 0.95,
    comp_val: Union[float, np.ndarray] = 0,
) -> pd.DataFrame:
    """
    Calculate various statistics from MCMC samples.

    Parameters
    ----------
    samples : dict
        Dictionary of MCMC samples.
    cred_mass : float, default=0.95
        The desired credible mass for intervals.
    comp_val : float or array-like, default=0
        Comparison value(s) for parameters.

    Returns
    -------
    DataFrame
        Statistics for each parameter.
    """
    stats = []

    # Make sure comp_val is an array
    if isinstance(comp_val, (int, float)):
        comp_val = np.repeat(comp_val, len(samples))

    # Calculate statistics for each parameter
    for i, (param, samp) in enumerate(samples.items()):
        # Basic statistics
        mean_val = np.mean(samp)
        median_val = np.median(samp)
        sd_val = np.std(samp)

        # HDI
        hdi = hdi_of_mcmc(samp, cred_mass)

        # Quantiles
        q025 = np.quantile(samp, 0.025)
        q25 = np.quantile(samp, 0.25)
        q75 = np.quantile(samp, 0.75)
        q975 = np.quantile(samp, 0.975)

        # Comparison probabilities
        if i < len(comp_val):
            pct_lt_comp = np.mean(samp < comp_val[i])
            pct_gt_comp = np.mean(samp > comp_val[i])
            comp_val_i = comp_val[i]
        else:
            # For theta_diff parameters, compare to 0
            pct_lt_comp = np.mean(samp < 0)
            pct_gt_comp = np.mean(samp > 0)
            comp_val_i = 0

        # Store statistics
        stats.append(
            {
                "parameter": param,
                "mean": mean_val,
                "median": median_val,
                "sd": sd_val,
                "HDI%": cred_mass * 100,
                "HDIlo": hdi[0],
                "HDIup": hdi[1],
                "comp": comp_val_i,
                "%<comp": pct_lt_comp,
                "%>comp": pct_gt_comp,
                "q2.5%": q025,
                "q25%": q25,
                "q75%": q75,
                "q97.5%": q975,
            }
        )

    return pd.DataFrame(stats)


def hdi_of_mcmc(sample_vec: np.ndarray, cred_mass: float = 0.95) -> tuple[float, float]:
    """
    Compute the Highest Density Interval (HDI) from MCMC samples.

    Parameters
    ----------
    sample_vec : array-like
        A vector of representative values from a probability distribution.
    cred_mass : float, default=0.95
        The desired credible mass.

    Returns
    -------
    tuple
        The lower and upper bounds of the HDI.
    """
    sorted_pts = np.sort(sample_vec)
    ci_idx_inc = int(np.floor(cred_mass * len(sorted_pts)))
    n_cis = len(sorted_pts) - ci_idx_inc

    ci_width = np.zeros(n_cis)
    for i in range(n_cis):
        ci_width[i] = sorted_pts[i + ci_idx_inc] - sorted_pts[i]

    min_idx = np.argmin(ci_width)
    hdi_min = sorted_pts[min_idx]
    hdi_max = sorted_pts[min_idx + ci_idx_inc]

    return (hdi_min, hdi_max)


def plot_bayes_prop_test(result: dict[str, Any], figsize: tuple[int, int] = (12, 10)):
    """
    Plot the results of a Bayesian proportion test.

    Parameters
    ----------
    result : dict
        The result dictionary from bayes_prop_test.
    figsize : tuple, default=(12, 10)
        Figure size.

    Returns
    -------
    None
        Displays the plot.
    """
    samples = result["mcmc_samples"]
    n_groups = len(result["x"])

    # Get theta samples
    theta_samples = {k: v for k, v in samples.items() if k.startswith("theta[")}

    # Get theta difference samples
    theta_diff_samples = result["theta_diff_samples"]

    # Create subplot grid
    fig = plt.figure(figsize=figsize)

    # Create layout grid
    total_plots = n_groups + len(theta_diff_samples)
    gs = plt.GridSpec(n_groups, n_groups)

    # Plot theta distributions
    for i in range(n_groups):
        ax = fig.add_subplot(gs[i, i])
        plot_posterior(
            theta_samples[f"theta[{i + 1}]"],
            ax=ax,
            cred_mass=result["cred_mass"],
            comp_val=(
                result["comp_theta"][i] if result["comp_theta"] is not None else None
            ),
            title=f"Rel. Freq. Group {i + 1}",
            x_label=f"θ{i + 1}",
            color="#5DE293",
        )

    # Plot theta differences
    for diff_name, diff_samples in theta_diff_samples.items():
        match = diff_name.split("[")[1].split("]")[0].split(",")
        i, j = int(match[0]), int(match[1])

        # Add subplot for θi - θj
        ax = fig.add_subplot(gs[i - 1, j - 1])
        plot_posterior(
            diff_samples,
            ax=ax,
            cred_mass=result["cred_mass"],
            comp_val=0,
            title=f"θ{i} - θ{j}",
            x_label=f"θ{i} - θ{j}",
            color="skyblue",
        )

        # Add subplot for θj - θi
        ax = fig.add_subplot(gs[j - 1, i - 1])
        plot_posterior(
            -diff_samples,
            ax=ax,
            cred_mass=result["cred_mass"],
            comp_val=0,
            title=f"θ{j} - θ{i}",
            x_label=f"θ{j} - θ{i}",
            color="skyblue",
        )

    plt.tight_layout()
    plt.show()


def plot_posterior(
    samples: np.ndarray,
    ax: plt.Axes = None,
    cred_mass: float = 0.95,
    comp_val: float = None,
    title: str = "",
    x_label: str = "Parameter",
    color: str = "skyblue",
    show_median: bool = True,
):
    """
    Plot a posterior distribution with HDI and optional comparison value.

    Parameters
    ----------
    samples : array-like
        MCMC samples of the parameter.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure is created.
    cred_mass : float, default=0.95
        The desired credible mass for HDI.
    comp_val : float, optional
        Comparison value to show on the plot.
    title : str, default=''
        Title for the plot.
    x_label : str, default='Parameter'
        Label for the x-axis.
    color : str, default='skyblue'
        Color for the histogram.
    show_median : bool, default=True
        Whether to show the median on the plot.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    # Calculate HDI
    hdi = hdi_of_mcmc(samples, cred_mass)

    # Plot histogram
    ax.hist(samples, bins=30, density=True, color=color, alpha=0.7)

    # Show median if requested
    if show_median:
        median_val = np.median(samples)
        ax.axvline(median_val, color="black", linestyle="-", linewidth=1.5)
        ax.text(
            median_val,
            ax.get_ylim()[1] * 0.9,
            f"median = {median_val:.2f}",
            horizontalalignment="center",
            verticalalignment="center",
        )

    # Show HDI
    ax.plot(
        [hdi[0], hdi[1]],
        [ax.get_ylim()[1] * 0.05, ax.get_ylim()[1] * 0.05],
        linewidth=4,
        color="black",
    )
    ax.text(
        (hdi[0] + hdi[1]) / 2,
        ax.get_ylim()[1] * 0.07,
        f"{cred_mass * 100:.0f}% HDI",
        horizontalalignment="center",
        verticalalignment="center",
    )
    ax.text(
        hdi[0],
        ax.get_ylim()[1] * 0.02,
        f"{hdi[0]:.2f}",
        horizontalalignment="center",
        verticalalignment="center",
    )
    ax.text(
        hdi[1],
        ax.get_ylim()[1] * 0.02,
        f"{hdi[1]:.2f}",
        horizontalalignment="center",
        verticalalignment="center",
    )

    # Show comparison value if provided
    if comp_val is not None:
        pct_lt_comp = np.mean(samples < comp_val)
        pct_gt_comp = np.mean(samples > comp_val)
        ax.axvline(comp_val, color="darkgreen", linestyle="dotted", linewidth=1.5)
        ax.text(
            comp_val,
            ax.get_ylim()[1] * 0.7,
            f"{pct_lt_comp * 100:.1f}% < {comp_val:.2f} < {pct_gt_comp * 100:.1f}%",
            horizontalalignment="center",
            verticalalignment="center",
            color="darkgreen",
        )

    # Labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel("Density")
    ax.set_title(title)

    return ax


def format_group_diffs(result, digits=2):
    """
    Format the pairwise group differences for printing.

    Parameters
    ----------
    result : dict
        Result dictionary from bayes_prop_test.
    digits : int, default=2
        Number of digits to round to.

    Returns
    -------
    pandas.DataFrame
        Formatted differences.
    """
    n_groups = len(result["x"])
    stats = result["stats"]
    diff_stats = stats.filter(regex="theta_diff")

    # Create matrices for medians and HDIs
    med_diff_mat = pd.DataFrame(np.zeros((n_groups, n_groups)), dtype=str)
    hdi_diff_mat = pd.DataFrame(np.zeros((n_groups, n_groups)), dtype=str)

    # Fill in matrices
    for diff_name in diff_stats.index:
        i, j = map(int, diff_name.split("[")[1].split("]")[0].split(","))
        med_diff_mat.iloc[i - 1, j - 1] = (
            f"{diff_stats.loc[diff_name, 'median']:.{digits}f}"
        )
        hdi_diff_mat.iloc[i - 1, j - 1] = (
            f"[{diff_stats.loc[diff_name, 'HDIlo']:.{digits}f}, {diff_stats.loc[diff_name, 'HDIup']:.{digits}f}]"
        )

    return med_diff_mat, hdi_diff_mat


# Example usage:
if __name__ == "__main__":
    # Example from a product attribution test
    f = np.array([10, 100])
    n = np.array([51, 51.4]) * f
    d = np.array([86, 86]) * f
    print(f"control n: {int(n[0]):,}, control d: {int(d[0]):,}")
    print(f"experiment n: {int(n[1]):,}, experiment d: {int(d[1]):,}")

    # Run the Bayesian proportion test
    fit = bayes_prop_test(n, d)

    # Plot results
    plot_bayes_prop_test(fit)
