"""Observing forecasted events."""
import astroplan as ap
import astropy.table as at
import numpy as np
from astropy import units
from astropy.time import Time
from scipy.signal import find_peaks
from scipy.stats import chi2

__all__ = [
    'transit_forecast',
    'relative_weights',
    'transit_observability_metric',
    'transit_probability_metric',
    'summarize_windows',
    'observable_windows'
]


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
    forecast = trace.tmforecast.mean(axis=0)

    return forecast


def relative_weights(lc, traces):
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


def weighted_transit_forecast(traces):
    """
    Calculate the weighted transit forecast.

    Parameters
    ----------
    traces : iterable
        A list of `~pymc3.backends.base.MultiTrace` objects.

    Returns
    -------
    tbars : ndarray
        The weighted transit forecasts for the scenarios.
    """
    # Define weights for the scenarios
    ndata = traces[0].lc_model.shape[1]
    nparam = 9  # Need a better way to define/find this value
    dof = ndata - nparam
    pvalues = _get_pvalues(traces, dof)
    weights = _get_weights(pvalues)

    # Caculate the weighted transit forecasts
    tbars = []
    for i, trace in enumerate(traces):
        tbar = (trace.tmforecast*weights[i, np.newaxis].T).sum(axis=0)
        tbars.append(tbar)
    tbars = np.array(tbars)

    return tbars


def transit_observability_metric(tbars, time, dt):
    """
    Calculate the transit observability metric.

    Parameters
    ----------
    tbars : iterable
        The weighted transit signals.

    time : iterable
        The array of time values corresponding to tbar.

    dt : float
        The width of the time window in the same units as `time`.

    Returns
    -------
    Ms : ndarray
        The transit observability metrics for the transit signals.
    """
    npoints = int(dt/np.median(np.diff(time)))
    Ms = []
    for tbar in tbars:
        M = _moving_average(-tbar, npoints)
        Ms.append(M)
    Ms = np.array(Ms)

    return Ms


def transit_probability_metric(tbar, time, lower_bound, upper_bound):
    """
    Calculate the transit probability metric.

    Parameters
    ----------
    tbar : iterable
        The "weighted mean" transit signal.

    time : iterable
        The array of time values corresponding to tbar.

    lower_bound : float
        The lower bound for calculating the M.

    upper_bound : float
        The upper bound for calculating the M.

    Returns
    -------
    M : float
        The transit probability metric.
    """
    idx = np.logical_and(
        time >= lower_bound,
        time <= upper_bound
    )
    M = np.trapz(tbar[idx], time[idx])/(upper_bound-lower_bound)
    return M


def summarize_windows(trace, tforecast, tdistance=None):
    """
    Summarize all transit windows suggested by the MCMC sampling.

    Parameters
    ----------
    traces : iterable
        A list of `~pymc3.backends.base.MultiTrace` objects.

    tforecast : `~numpy.array`
        The time array corresponding to the forecasted transit models.

    tdistance : float
        The time distance bewteen peaks in the same units as `tforecast.`
        Defaults to 1/2 the median of the posterior distribution of the period
        in each `~pymc3.backends.base.MultiTrace`.

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
    if tdistance is None:
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
    windows = at.Table({
        'median': Time(medians, format='jd', scale='tdb'),
        'lower': Time(lowers, format='jd', scale='tdb'),
        'upper': Time(uppers, format='jd', scale='tdb')
    })

    return windows


def summarize_windows_v0(traces, tforecast, tdistance=None):
    """
    Summarize all transit windows suggested by the MCMC sampling.

    Parameters
    ----------
    traces : iterable
        A list of `~pymc3.backends.base.MultiTrace` objects.

    tforecast : `~numpy.array`
        The time array corresponding to the forecasted transit models.

    tdistance : float
        The time distance bewteen peaks in the same units as `tforecast.`
        Defaults to 1/2 the median of the posterior distribution of the period
        in each `~pymc3.backends.base.MultiTrace`.

    Returns
    -------
    windows : `~astropy.table.Table`
        A table of the identified windows.
    """
    # Define some useful variables
    dt = np.median(np.diff(tforecast))
    tdists = [tdistance]*len(traces)

    # Define weights for the scenarios
    ndata = traces[0].lc_model.shape[1]
    nparam = 9  # Need a better way to define/find this value
    dof = ndata - nparam
    pvalues = _get_pvalues(traces, dof)
    weights = _get_weights(pvalues)

    # Loop through the scenarios, summarizing transit windows
    windows_list = []
    for i, trace in enumerate(traces):
        tbar = _get_tbar(trace.tmforecast, weights[i, :])

        # Identify peaks
        post_period = np.median(trace.period)
        tdist = tdists[i]
        if tdist is None:
            # Treat peaks within P/2 as a single peak
            tdist = 0.5*post_period
            distance = int((tdist)/dt)
        idx_peaks, _ = find_peaks(tbar, distance=distance)
        tpeaks = tforecast[idx_peaks]

        # Identify the median and lower and upper bound of the distribution
        # surrounding each peak and it's corresponding M
        medians = np.empty(tpeaks.size)
        lowers = np.empty(tpeaks.size)
        uppers = np.empty(tpeaks.size)
        Ms = np.empty(tpeaks.size)
        for ii, tpeak in enumerate(tpeaks):
            idx = np.abs(tforecast-tpeak) < tdist
            t_win = tforecast[idx]
            tbar_win = tbar[idx]
            medians[ii] = _weighted_percentile(t_win, tbar_win, 50)
            lowers[ii] = _weighted_percentile(t_win, tbar_win, 2.5)
            uppers[ii] = _weighted_percentile(t_win, tbar_win, 97.5)
            Ms[ii] = transit_probability_metric(
                tbar, tforecast, lowers[ii], uppers[ii]
            )

        # Store results in a DataFrame
        windows = at.Table({
            'scenario': (i+1)*np.ones_like(tpeaks).astype('int'),
            'median': Time(medians, format='jd', scale='tdb'),
            'lower': Time(lowers, format='jd', scale='tdb'),
            'upper': Time(uppers, format='jd', scale='tdb'),
            'M': Ms
        })
        windows_list.append(windows)

    # Concatenate all results into a single DataFrame
    windows = at.vstack(windows_list)

    return windows


def observable_windows(target, site, constraints, windows):
    """
    Determine which windows are observable, given constraints.

    Parameters
    ----------
    target : `~astroplan.FixedTarget`
        A target.

    site : `~astroplan.Observer`
        A site.

    constraints : iterable
        A list of `~astroplan.Constraint` objects.

    windows : `~astropy.table.Table`
        A table of the potential windows.

    Returns
    -------
    obs_windows : `~astropy.table.Table`
        A table of the observable windows.
    """
    # Determine the observable fraction of the window
    fractions = []
    for window in windows:
        time_range = [window['lower'], window['upper']]
        obs_table = ap.observability_table(
            constraints,
            site,
            [target],
            time_range=time_range,
            time_grid_resolution=10*units.min
        )
        fractions.append(obs_table['fraction of time observable'][0])
    windows['fraction'] = fractions
    obs_windows = windows[windows['fraction'] > 0]

    # Determine start and end times of observations and
    # refine the observable fraction
    starts = []
    ends = []
    refined_fractions = []
    for window in obs_windows:
        time_range = [window['lower'], window['upper']]
        time_grid = ap.time_grid_from_range(
            time_range, time_resolution=1*units.min
        )
        observable = ap.is_event_observable(
            constraints,
            site,
            target,
            time_grid
        )[0]
        starts.append(time_grid[observable].min())
        ends.append(time_grid[observable].max())
        refined_fractions.append(observable.sum()/len(observable))

    obs_windows['fraction'] = refined_fractions
    obs_windows['start'] = starts
    obs_windows['end'] = ends

    # Calculate the airmass of the target at important times
    obs_windows['zstart'] = site.altaz(obs_windows['start'], target).secz
    obs_windows['zmedian'] = site.altaz(obs_windows['median'], target).secz
    obs_windows['zend'] = site.altaz(obs_windows['end'], target).secz

    # Drop some columns and sort the rest
    obs_windows.remove_columns(['lower', 'upper'])
    cols = [
        'scenario', 'M', 'fraction',
        'start', 'median', 'end',
        'zstart', 'zmedian', 'zend'
    ]
    obs_windows = obs_windows[cols]

    return obs_windows


def _get_pvalues(traces, dof):
    """
    Calculate the p values for models in a list of traces.

    Parameters
    ----------
    traces : iterable
        A list of `~pymc3.backends.base.MultiTrace` objects.

    dof : int
        The degrees of freedom for the model fit.

    Returns
    -------
    pvalues : `~numpy.array`
        An array with the p-values for the set of models.
    """
    x2s = np.empty((len(traces), len(traces[0].x2)))
    for i, trace in enumerate(traces):
        x2s[i, :] = trace.x2
    pvalues = chi2.sf(x2s, dof)
    return pvalues


def _get_weights(pvalues):
    """
    Calculate the normalized weights, given a set of p-values.

    Parameters
    ----------
    pvalues : `~numpy.array`
        An array of p-values.

    Returns
    -------
    weights : `~numpy.array`
        An array with the corresponding normalized weights.
    """
    weights = pvalues/pvalues.sum()
    return weights


def _get_tbar(tmforecast, weights):
    """
    Calculate Tbar, the inverse of the weighted-mean transit forecast.

    Parameters
    ----------
    tmforecast : `~numpy.array`
        An array of forecasted transit models.

    weights : `~numpy.array`
        An array with the corresponding normalized weights.

    Returns
    -------
    tbar : `~numpy.array`
        The inverse of the weighted-mean transit forecast.
    """
    tbar = -(weights[:, np.newaxis]*tmforecast).sum(axis=0)
    return tbar


def _moving_average(x, window):
    """
    Calculate the moving average within a window.

    Parameters
    ----------
    x : array_like
        The data (or "x" values).

    window : int
        The width of the window.

    Returns
    -------
    ma : `~numpy.array`
        The moving average of the data.
    """
    ma = np.convolve(x, np.ones(window), 'same') / window

    return ma


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
