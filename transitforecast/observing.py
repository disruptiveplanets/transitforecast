"""Observing forecasted events."""
import astroplan as ap
import astropy.units as units
import batman
import numpy as np
import pandas as pd
from astropy.time import Time

__all__ = [
    'relative_weights',
    'transit_forecast',
    'observable_windows'
]


def _median_light_curve(trace, time):
    texp = np.median(np.diff(time))
    pt_lc_models = []
    # Calculate the model for each point in the trace
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
        m = batman.TransitModel(params, time, exp_time=texp)
        pt_lc_model = m.light_curve(params) - 1 + pt['f0']
        pt_lc_models.append(pt_lc_model)
    # Take the median
    med_lc_model = np.median(pt_lc_models, axis=0)

    return med_lc_model


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
        _median_light_curve(trace, lc.time) for trace in traces
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


def observable_windows(
    tforecast, forecast, target, site, constraints, max_obs_duration=np.inf
):
    """
    Identify observable follow-up windows.

    Parameters
    ----------
    tforecast : `~astropy.time.Time`
        The time array for the transit forecast.

    forecast : `~numpy.array`
        The transit forecast.

    target : `~astroplan.FixedTarget`
        A target object.

    site : `~astroplan.Observer`
        A site object.

    constraints : iterable
        A list of `~astroplan.Constraint` objects.

    max_obs_duration : float or `~astropy.units.Quantity`, optional
        The maximum duration of an observation. Defaults to days if unit not
        specified.

    Returns
    -------
    windows : `~pandas.DataFrame`
        A table of the observable windows.
    """
    if type(max_obs_duration) is units.Quantity:
        max_obs_duration = max_obs_duration.to(units.day).value

    # For simplicity, just use BJD times
    times = tforecast.jd

    observability = ap.is_event_observable(
        constraints, site, target, times=tforecast
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
        t_max = (times[idx_window])[np.argmax(forecast[idx_window])]
        idx_window = _refine_window(
            times, t_max, idx_window, max_obs_duration
        )
        t_start = (times[idx_window]).min()
        t_end = (times[idx_window]).max()
        dt = (times[idx_window]).ptp()
        M = np.trapz(forecast[idx_window], times[idx_window]) / dt

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


def _refine_window(times, t_max, idx_window, max_obs_duration):
    """
    Refine an observation window.
    """
    t_start = (times[idx_window]).min()
    t_end = (times[idx_window]).max()

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
        idx_window = np.where((times >= t1) & (times <= t2))[0]

    return idx_window
