"""Observing forecasted events."""
import astroplan as ap
import astropy.table as astrotab
import batman
import numpy as np
import pandas as pd
from astropy.time import Time
from scipy.signal import find_peaks

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


def transit_forecast(lc, traces, tforecast):
    """
    Calculate the mean transit forecast.

    Parameters
    ----------
    lc : `~lightkurve.LightCurve`
        A light curve object with the data.

    trace : iterable
        A list of `~pymc3.backends.base.MultiTrace` MCMC trace objects.

    tforecast : `~numpy.array`
        The times to calculate the forecast.

    Returns
    -------
    forecast : ndarray
        The transit forecast.
    """
    weights = relative_weights(lc, traces)
    texp = np.median(np.diff(tforecast))
    transit_signals = []
    # For each trace ...
    for trace in traces:
        pt_transit_signals = []
        # Calculate the forecast for each point in the trace
        for i, pt in enumerate(trace.points()):
            # `batman` is much faster than `exoplanet` in this case
            params = batman.TransitParams()
            params.t0 = pt['t0']
            params.per = pt['period']
            params.rp = pt['r']
            params.a = pt['aRs']
            params.inc = pt['incl']
            params.ecc = 0.
            params.w = 90.
            params.u = pt['u']
            params.limb_dark = 'quadratic'
            m = batman.TransitModel(params, tforecast, exp_time=texp)
            pt_transit_signal = m.light_curve(params) - 1
            pt_transit_signals.append(pt_transit_signal)
        # Take the mean of the point-by-point transit models
        transit_signal = np.array(pt_transit_signals).mean(axis=0)
        transit_signals.append(transit_signal)
    # And use the per-trace means, and their weights, to calculate the forecast
    forecast = (
        - weights[:, np.newaxis] * np.array(transit_signals)
    ).sum(axis=0)

    return forecast


def relative_weights(lc, traces, nparam=9):
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


def combine_forecasts(forecasts, weights=None):
    """
    Calculate the combined transit forecast.

    Parameters
    ----------
    forecasts : iterable
        A list of the transit forecasts.

    weights : iterable, optional.
        Relative weights for the forecasts. Defaults to uniform weight.

    Returns
    -------
    combined_forecast : `~numpy.array`
        The combined (weighted) forecast.
    """
    if weights is None:
        weights = np.ones(len(forecasts))
    combined_forecast = np.sum(weights[:, None]*forecasts, axis=0)

    return combined_forecast


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
    tforecast, forecast, target, site, constraints, max_obs_duration=np.inf
):
    """
    Identify observable follow-up windows.

    Parameters
    ----------
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

    max_obs_duration : float
        The maximum duration of an observation in days.

    Returns
    -------
    windows : `~pandas.DataFrame`
        A table of the observable windows.
    """
    observability = ap.is_event_observable(
        constraints, site, target, times=Time(tforecast + 2457000, format='jd')
    ).flatten()
    idx_observable = np.where(observability)[0]
    idx_window_list = np.split(
        idx_observable, np.where(np.diff(idx_observable) != 1)[0] + 1
    )
    dts = []
    t_starts = []
    t_ends = []
    t_maxs = []
    Ms = []
    for idx_window in idx_window_list:
        t_max = (tforecast[idx_window])[np.argmax(forecast[idx_window])]
        idx_window = _refine_window(
            tforecast, t_max, idx_window, max_obs_duration
        )
        t_start = (tforecast[idx_window]).min()
        t_end = (tforecast[idx_window]).max()
        dt = (tforecast[idx_window]).ptp()
        M = np.trapz(forecast[idx_window], tforecast[idx_window]) / dt

        dts.append(dt)
        t_starts.append(t_start)
        t_ends.append(t_end)
        t_maxs.append(t_max)
        Ms.append(M)

    windows = pd.DataFrame({
        't_start': t_starts,
        't_max': t_maxs,
        't_end': t_ends,
        'dt': dts,
        'M': Ms
    }).sort_values('M', ascending=False)

    return windows


def _refine_window(tforecast, t_max, idx_window, max_obs_duration):
    """
    Refine an observation window.
    """
    t_start = (tforecast[idx_window]).min()
    t_end = (tforecast[idx_window]).max()

    if (t_end - t_start) > max_obs_duration:
        t1 = t_max - max_obs_duration / 2.
        t2 = t_max + max_obs_duration / 2.
        buffer_1 = t1 - t_start
        buffer_2 = t_end - t2
        buffers = [buffer_1, buffer_2]
        # If t1 and t2 not t_win, find new start and end times
        if not np.min(buffers) > 0:
            # If t_max closer to t_start, use original start time
            # and adjust end time, unless would be outside the window
            if np.argmin(buffers) == 0:
                t1 = t_start
                t2 = np.min([t_end, t_start + max_obs_duration])
            # If t_max closer to t_end, use original end time
            # and adjust start time, unless would be outside the window
            elif np.argmin(buffers) == 1:
                t1 = np.max([t_start, t_end - max_obs_duration])
                t2 = t_end
        # Calculate new idx_window
        idx_window = np.where((tforecast >= t1) & (tforecast <= t2))[0]

    return idx_window


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
