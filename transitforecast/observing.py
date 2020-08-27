"""Observing forecasted events."""
import astroplan as ap
import astropy.table as astrotab
import numpy as np
from astropy import units
from astropy.time import Time
from scipy.signal import find_peaks
from scipy.stats import chi2

__all__ = [
    'transit_forecast',
    'relative_weights',
    'summarize_windows',
    'observable_windows'
]


def _lnlikelihood(obs, model, err):
    n = len(obs)
    lnlike = (
        - n/2 * np.log(2*np.pi)
        - np.sum(np.log(err))
        - np.sum((obs-model)**2 / (2*err**2))
    )

    return lnlike


def _bayesian_information_criterion(obs, model, err, nparam):
    n = len(obs)
    bic = np.log(n)*nparam - 2*_lnlikelihood(obs, model, err)

    return bic


def transit_forecast(trace):
    """
    Calculate the mean transit forecast.

    Parameters
    ----------
    trace : `~pymc3.backends.base.MultiTrace`
        The MCMC trace object.

    Returns
    -------
    forecast : ndarray
        The mean transit forecast for the scenario.
    """
    forecast = trace.forecast.mean(axis=0)

    return forecast


def relative_weights(lc, traces, nparam):
    """
    Calculate the relative weights for the scenarios.

    Parameters
    ----------
    lc : `~lightkurve.LightCurve`
        A light curve object with the data.

    traces : iterable
        A list of `~pymc3.backends.base.MultiTrace` objects.

    nparam : int
        The number of free parameters in the model.

    Returns
    -------
    weights : ndarray
        The relative weights for the scenarios.
    """
    # Calculate the median light curve model
    med_lc_models = [
        np.median(trace.lc_model, axis=0) for trace in traces
    ]

    # Calculate the BIC
    bics = [
        _bayesian_information_criterion(
            lc.flux, m, lc.flux_err, nparam
        ) for m in med_lc_models
    ]
    dbics = bics - np.min(bics)

    # Weight by the relative BIC values
    weights = np.exp(dbics/-2)/np.sum(np.exp(dbics/-2))

    return weights


def relative_weights_pvalues(lc, traces):
    """
    Calculate the relative weights for the scenarios.

    Parameters
    ----------
    lc : `~lightkurve.LightCurve`
        A light curve object with the data.

    traces : iterable
        A list of `~pymc3.backends.base.MultiTrace` objects.

    Returns
    -------
    weights : ndarray
        The relative weights for the scenarios.
    """
    # Calculate the median light curve model
    med_lc_models = [
        np.median(trace.lc_model, axis=0) for trace in traces
    ]

    # Calculate x2
    x2s = [
        np.sum(((lc.flux-model)/lc.flux_err)**2) for model in med_lc_models
    ]

    # Calculate the p-value
    ndata = traces[0].lc_model.shape[1]
    nparam = 9  # Need a better way to define/find this value
    dof = ndata - nparam
    pvalues = chi2.sf(x2s, dof)

    # Calculate relative weighted_transit_forecast
    weights = pvalues/pvalues.sum()  # Should this be pvalues.max() instead?

    return weights


def summarize_windows(trace, tforecast, tdist=None):
    """
    Summarize transit windows suggested by the MCMC sampling of one scenario.

    Parameters
    ----------
    trace : `~pymc3.backends.base.MultiTrace`
        The trace from the MCMC sampling.

    tforecast : `~numpy.array`
        The time array corresponding to the forecasted transit model.

    tdist : float, optional
        The time distance bewteen peaks in the same units as `tforecast.`
        Defaults to 1/2 the median of the posterior distribution of the period
        in the trace.

    Returns
    -------
    windows : `~astropy.table.Table`
        A table of the identified windows.
    """
    # Define some useful variables
    dt = np.median(np.diff(tforecast))

    # Identify peaks
    forecast = transit_forecast(trace)
    post_period = np.median(trace.period)
    if tdist is None:
        # Treat peaks within P as a single peak
        tdist = post_period
    distance = int((tdist)/dt)
    idx_peaks, _ = find_peaks(-forecast, distance=distance)
    tpeaks = tforecast[idx_peaks]

    # Summarize transit windows
    medians = np.empty(tpeaks.size)
    lowers = np.empty(tpeaks.size)
    uppers = np.empty(tpeaks.size)
    for i, tpeak in enumerate(tpeaks):
        idx = np.abs(tforecast-tpeak) < 0.5*tdist
        t_win = tforecast[idx]
        f_win = forecast[idx]
        medians[i] = _weighted_percentile(t_win, f_win, 50.)
        lowers[i] = t_win[np.nonzero(f_win)[0].min()]
        uppers[i] = t_win[np.nonzero(f_win)[0].max()]

    # Store results in a astropy.table.Table
    windows = astrotab.Table({
        'median': Time(medians, format='jd', scale='tdb'),
        'lower': Time(lowers, format='jd', scale='tdb'),
        'upper': Time(uppers, format='jd', scale='tdb')
    })

    return windows


def observable_windows(
    windows, tforecast, forecast, target, site, constraints, weight=1.
):
    """
    Determine which windows are observable, given constraints.

    Parameters
    ----------
    windows : `~astropy.table.Table`
        A table of the forecasted windows.

    tforecast : `~numpy.array`
        The time array for the forecasted transit models.

    forecast : `~numpy.array`
        The transit forecast.

    target : `~astroplan.FixedTarget`
        A target object.

    site : `~astroplan.Observer`
        A site object.

    constraints : iterable
        A list of `~astroplan.Constraint` objects.

    weight : float, optional
        Relative weight of the scenario. Defaults to 1 if not specified.

    Returns
    -------
    obs_windows : `~astropy.table.Table`
        A table of the observable windows.
    """
    # Iterate through windows, determining integrated forecasted signals,
    # start and end times, the duration of the observation, and
    # the observational efficiency metric.
    tbars = np.empty(len(windows))
    t_starts = np.empty(len(windows))
    t_ends = np.empty(len(windows))
    dts = np.empty(len(windows))
    Ms = np.empty(len(windows))
    for i, window in enumerate(windows):
        idx = np.logical_and(
            tforecast >= window['lower'].jd,
            tforecast <= window['upper'].jd
        )
        t_win = tforecast[idx]
        f_win = forecast[idx]
        obs = ap.is_event_observable(
            constraints,
            site,
            target,
            Time(t_win, format='jd')
        ).flatten()

        # Target is unobservable during window
        if not obs.sum():
            tbar = np.nan
            t_start = np.nan
            t_end = np.nan
            dt = np.nan
            M = np.nan
        # Target is observable during window
        else:
            tbar = np.trapz(f_win[obs], t_win[obs])
            t_start = t_win[obs][0]
            t_end = t_win[obs][-1]
            dt = t_win[obs].ptp()
            M = -weight*tbar/dt
        tbars[i] = tbar
        t_starts[i] = t_start
        t_ends[i] = t_end
        dts[i] = dt
        Ms[i] = M
    windows['int_signal'] = tbars
    windows['t_start'] = t_starts
    windows['t_end'] = t_ends
    windows['dt'] = dts
    windows['M'] = Ms

    # Select only observable windows
    obs_windows = windows[np.isfinite(windows['dt'])]

    # Convert some times to `astropy.time.Time` objects
    obs_windows['t_start'] = Time(obs_windows['t_start'], format='jd')
    obs_windows['t_end'] = Time(obs_windows['t_end'], format='jd')

    # Add units for dt
    obs_windows['dt'] = obs_windows['dt']*units.d

    # Add weights
    obs_windows['weight'] = weight

    # Reorder the columns
    cols = [
        'median', 'lower', 'upper', 't_start', 't_end', 'dt',
        'int_signal', 'weight', 'M'
    ]
    obs_windows = obs_windows[cols]

    return obs_windows


def _weighted_percentile(data, weights, percentile):
    """
    Calculate the weighted percentile of a data set.

    Parameters
    ----------
    data : iterable
        The data (or "x" values).

    weights : iterable
        The weights (or "y" values).

    percentile : float
        The percentile to calculate.

    Returns
    -------
    value : float
        The value corresponding to the percentile.
    """
    cumsum = np.cumsum(weights)
    percentiles = 100*(cumsum-0.5*weights)/cumsum[-1]
    value = np.interp(percentile, percentiles, data)
    return value
