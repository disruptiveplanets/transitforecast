"""Plotting functions."""
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
from scipy.stats import median_abs_deviation


def plot_map_soln(lc, map_soln):
    """
    Plot the maximum a posteriori solution for a transit model.

    Parameters
    ----------
    lc : `~lightkurve.LightCurve`
        A light curve object with the data.

    map_soln : dict
        A dictionary with the maximum a posteriori estimates of the variables.

    Returns
    -------
    fig : `~matplotlib.figure.Figure`
        A Figure object.

    axes : `~matplotlib.axes.Axes`
        An Axes object with plots of the MAP solution as function of time and
        orbital phase.
    """
    map_p = map_soln['period']
    map_t0 = map_soln['t0']

    fig, axes = plt.subplots(2)

    ax = axes[0]
    ax.plot(lc.time, lc.flux, 'k.', ms=3, mew=0, alpha=0.1)
    ax.plot(lc.time, map_soln['lc_model'], lw=1, color='C0')
    ax.set_xlabel('Time (TBJD)')
    ax.set_ylabel('Normalized Flux')

    ax = axes[1]
    det_flux = lc.flux
    map_phase = ((lc.time-map_t0) % map_p)/map_p
    map_phase[map_phase > 0.5] -= 1
    order = np.argsort(map_phase)
    blc = lc.fold(t0=map_t0, period=map_p).bin(100)
    ax.plot(map_phase, det_flux, 'k.', ms=3, mew=0, alpha=0.1)
    ax.plot(map_phase[order], map_soln['lc_model'][order], color='C0')
    ax.errorbar(
        blc.time, blc.flux, blc.flux_err,
        ls='', marker='o', ms=2, color='k', mfc='white'
        )
    ax.set_xlabel('Phase')
    ax.set_ylabel('Detrended Flux')

    return fig, axes


def plot_posterior_model(lc, trace):
    """
    Plot the posterior light curve model.

    Parameters
    ----------
    lc : `~lightkurve.LightCurve`
        A light curve object with the data.

    trace : `~pymc3.backends.base.MultiTrace`
        A ``MultiTrace`` object that contains the samples.

    Returns
    -------
    fig : `~matplotlib.figure.Figure`
        A Figure object.

    axes : `~matplotlib.axes.Axes`
        An Axes object with plots of the posterior model as functions of time
        and orbital phase.
    """
    varnames = [
        'm_star',
        'r_star',
        'f0',
        'u',
        'r',
        'b',
        'period',
        't0',
        'depth',
        'rearth',
        'a',
        'aRs',
        'incl',
        't14',
        'rho_star',
        'rho_star_sol',
    ]

    # Summary of posteriors
    func_dict = {
        'median': lambda x: np.percentile(x, 50),
        'upper': lambda x: np.percentile(x, 84)-np.percentile(x, 50),
        'lower': lambda x: np.percentile(x, 50)-np.percentile(x, 16),
    }

    summary = pm.summary(
        trace,
        varnames=varnames,
        hdi_prob=0.68,
        stat_funcs=func_dict,
        round_to=8,
    )

    lc_model_med = np.median(trace['lc_model'], axis=0)
    idx = np.random.choice(trace['lc_model'].shape[0], 100)
    lc_model_draws = trace['lc_model'][idx, :]

    post_t0 = summary['median']['t0']
    post_period = summary['median']['period']
    mad = median_abs_deviation(lc.flux, scale='normal')

    fig, axes = plt.subplots(2)

    ax = axes[0]
    ax.plot(lc.time, lc.flux, 'k.', ms=3, mew=0, alpha=0.5)
    ax.plot(lc.time, (lc_model_draws.T), color='C0', alpha=0.01)
    ax.plot(lc.time, lc_model_med, lw=1.5, color='white')
    ax.plot(lc.time, lc_model_med, lw=1, color='C0')
    ax.set_xlabel('Time (TBJD)')
    ax.set_ylabel('Normalized Flux')

    ax = axes[1]
    det_flux = lc.flux
    phase = ((lc.time-post_t0) % post_period)/post_period
    phase[phase > 0.5] -= 1
    order = np.argsort(phase)
    ax.plot(phase, det_flux, 'k.', ms=3, mew=0, alpha=0.5)
    ax.plot(
        phase[order], (lc_model_draws.T)[order],
        color='C0', alpha=0.01
    )
    ax.plot(
        phase[order], lc_model_med[order],
        color='white', lw=1.5
    )
    ax.plot(
        phase[order], lc_model_med[order],
        color='C0', lw=1
    )
    ax.set_xlim(-0.025, 0.025)
    ax.set_ylim(
        np.min(lc_model_med-3*mad),
        np.max(lc_model_med+3*mad)
    )
    ax.set_xlabel('Phase')
    ax.set_ylabel('Normalized Flux')

    return fig, axes
