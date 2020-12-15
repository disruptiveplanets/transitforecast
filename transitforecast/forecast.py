"""Forecasting transit events."""
import astroplan as ap
import exoplanet as xo
import joblib as jl
import multiprocessing as mp
import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
import transitleastsquares as tls
import warnings
import wotan
from astropy import units
from astropy.time import Time
from scipy.stats import median_abs_deviation

__all__ = [
    'build_model',
    'build_models',
    'get_map_soln',
    'get_map_solns',
    'get_map_solns_mp',
    'build_model_and_get_map_soln',  # Should remove this at some point
    'downselect',
    'flatten',
    'get_forecast_window',
    'get_priors_from_tic',
    'get_trial_ephemerides',
    'sample_from_model',
    'sample_from_models'
]


def build_model(
    lc, pri_t0, pri_p, pri_rprs, pri_m_star, pri_r_star, verbose=False
):
    """
    Build the transit light curve model.

    Parameters
    ----------
    lc : `~lightkurve.LightCurve`
        A light curve object with the data.

    pri_t0 : float
        Initial guess for mid-transit time.

    pri_p : float
        Initial guess for period.

    pri_rprs : float
        Initial guess for planet-to-star radius ratio.

    pri_m_star : ndarray
        Mean and standard deviation of the stellar mass estimate
        in solar masses.

    pri_r_star : ndarray
        Mean and standard deviation of the stellar radius estimate
        in solar radii.

    verbose : bool, optional
        Print details of optimization.

    Returns
    -------
    model : `~pymc3.model`
        A model object.
    """
    # Ignore warnings from theano, unless specified elsewise
    if not verbose:
        warnings.filterwarnings(
            action='ignore',
            category=FutureWarning,
            module='theano'
        )

    # Ensure right data type for theano
    dtype = np.dtype('float64')
    dts = [arr.dtype for arr in [lc.time, lc.flux, lc.flux_err]]
    if not all(dt is dtype for dt in dts):
        lc.time = np.array(lc.time, dtype=dtype)
        lc.flux = np.array(lc.flux, dtype=dtype)
        lc.flux_err = np.array(lc.flux_err, dtype=dtype)

    # Estimate flux uncertainties if not given
    idx_nan = np.isnan(lc.flux_err)
    if idx_nan.any():
        mad = median_abs_deviation(lc.flux, scale='normal')
        lc.flux_err[idx_nan] = mad

    # Define the model for the light curve
    with pm.Model() as model:
        # Stellar mass
        m_star = pm.Normal(
            'm_star',
            mu=pri_m_star[0],
            sigma=pri_m_star[1]
        )

        # Stellar radius
        r_star = pm.Normal(
            'r_star',
            mu=pri_r_star[0],
            sigma=pri_r_star[1]
        )

        # Quadratic limb-darkening parameters
        u = xo.distributions.QuadLimbDark(
            'u',
            testval=np.array([0.3, 0.2])
        )

        # Radius ratio
        r = pm.Uniform(
            'r',
            lower=0.,
            upper=1.,
            testval=pri_rprs
        )

        # Impact parameter
        b = xo.distributions.ImpactParameter(
            'b',
            ror=r,
        )

        # Period
        logperiod = pm.Uniform(
            'logperiod',
            lower=-2.3,  # 0.1 d
            upper=3.4,  # 30 d
            testval=np.log(pri_p)
        )
        period = pm.Deterministic('period', tt.exp(logperiod))

        # Mid-transit time
        t0 = pm.Uniform(
            't0',
            lower=lc.time.min(),
            upper=lc.time.max(),
            testval=pri_t0
        )

        # Keplerian orbit
        orbit = xo.orbits.KeplerianOrbit(
            m_star=m_star,
            r_star=r_star,
            period=period,
            t0=t0,
            b=b
        )

        # Model transit light curve
        light_curves = xo.LimbDarkLightCurve(
            u).get_light_curve(orbit=orbit, r=r*r_star, t=lc.time)
        transit_model = pm.math.sum(light_curves, axis=-1)

        # The baseline flux
        f0 = pm.Normal(
            'f0',
            mu=np.median(lc.flux),
            sigma=median_abs_deviation(lc.flux, scale='normal')
        )

        # The full model
        lc_model = transit_model + f0

        #######################
        # Track some parameters
        #######################

        # Track transit depth
        pm.Deterministic('depth', r**2)

        # Track planet radius (in Earth radii)
        pm.Deterministic(
            'rearth',
            r*r_star*(units.solRad/units.earthRad).si.scale
        )

        # Track semimajor axis (in AU)
        au_per_rsun = (units.solRad/units.AU).si.scale
        pm.Deterministic('a', orbit.a*au_per_rsun)

        # Track system scale
        pm.Deterministic('aRs', orbit.a/r_star)  # normalize by stellar radius

        # Track inclination
        pm.Deterministic('incl', np.rad2deg(orbit.incl))

        # Track transit duration
        # Seager and Mallen-Ornelas (2003) Eq. 3
        sini = np.sin(orbit.incl)
        t14 = (
            (period/np.pi) *
            np.arcsin((r_star/orbit.a*sini) * np.sqrt((1.+r)**2 - b**2))
        )*24.*60.  # min
        pm.Deterministic('t14', t14)

        # Track stellar density (in cgs units)
        pm.Deterministic('rho_star', orbit.rho_star)

        # Track stellar density (in units of solar density)
        rho_sol = (units.solMass/(4./3.*np.pi*units.solRad**3)).cgs.value
        pm.Deterministic('rho_star_sol', orbit.rho_star/rho_sol)

        # Track x2
        x2 = pm.math.sum(((lc.flux-lc_model)/lc.flux_err)**2)
        pm.Deterministic('x2', x2)

#         # Fit for variance
#         logs2 = pm.Normal('logs2', mu=np.log(np.var(lc.flux)), sigma=1)
#         sigma = pm.Deterministic('sigma', pm.math.sqrt(pm.math.exp(logs2)))

        # The likelihood function
        pm.Normal('obs', mu=lc_model, sigma=lc.flux_err, observed=lc.flux)

    # Reset warning filter
    warnings.resetwarnings()

    return model


def build_models(lc, ephem, pri_m_star, pri_r_star, verbose=False):
    """
    Build all transit light curve models.

    Parameters
    ----------
    lc : `~lightkurve.LightCurve`
        A light curve object with the data.

    ephem : pandas.DataFrame
        The trial ephemerides.

    pri_m_star : ndarray
        Mean and standard deviation of the stellar mass estimate
        in solar masses.

    pri_r_star : ndarray
        Mean and standard deviation of the stellar radius estimate
        in solar radii.

    verbose : bool, optional
        Print details of optimization.

    Returns
    -------
    models : list
        A list of `~pymc3.model` objects.
    """
    # This cannot be done in parallel due to clashes
    # with the theano.gof.compilelock.
    models = []
    for i, ep in ephem.iterrows():
        model = build_model(
            lc, ep.t0, ep.period, ep.rprs, pri_m_star, pri_r_star, verbose
        )
        models.append(model)

    return models


def get_map_soln(model, verbose=False, ignore_warnings=True):
    """
    Get the maximum a posteriori probability estimate of the parameters.

    Parameters
    ----------
    model : `~pymc3.model`
        The model object.

    verbose : bool, optional
        Print details of optimization.

    ignore_warnings : bool, optional
        Silence warnings.

    Returns
    -------
    map_soln : dict
        A dictionary with the maximum a posteriori estimates of the variables.
    """
    # Ignore warnings from theano, unless specified elsewise
    if ignore_warnings:
        warnings.filterwarnings(
            action='ignore',
            category=FutureWarning,
            module='theano'
        )

    with model:
        # Fit for the maximum a posteriori parameters
        map_soln = xo.optimize(
            start=model.test_point,
            verbose=verbose
        )
        map_soln = xo.optimize(
            start=map_soln,
            vars=[model.f0, model.period, model.t0, model.r],
            verbose=verbose
        )
        map_soln = xo.optimize(
            start=map_soln,
            vars=model.rho_star,
            verbose=verbose
        )
        map_soln = xo.optimize(
            start=map_soln,
            vars=model.t14,
            verbose=verbose
        )
        map_soln = xo.optimize(
            start=map_soln,
            verbose=verbose
        )

    # Reset warning filter
    warnings.resetwarnings()

    return map_soln


def get_map_solns(models, verbose=False, ignore_warnings=True):
    """
    Get the MAP estimates for a list of models.

    Parameters
    ----------
    models : list
        A list of `~pymc3.model` objects.

    verbose : bool, optional
        Print details of optimization.

    ignore_warnings : bool, optional
        Silence warnings.

    Returns
    -------
    map_solns : list
        A list of dictionaries with the MAP estimates for the models.
    """
    map_solns = [
        get_map_soln(model, verbose, ignore_warnings) for model in models
    ]

    return map_solns


def get_map_solns_mp(models, verbose=False, loud=False):
    """
    Get the maximum a posteriori probability estimate of the parameters.

    Parameters
    ----------
    models : list
        A list of `~pymc3.model` objects.

    verbose : bool, optional
        Print details of multiprocessing.

    loud : bool, optional
        Print details of optimization.

    Returns
    -------
    map_solns : list
        A list of dictionaries with the MAP estimates for the models.
    """

    n_cores = mp.cpu_count()
    backend = jl.parallel_backend('multiprocessing')
    with backend:
        map_solns = jl.Parallel(n_jobs=n_cores, verbose=verbose)(
            jl.delayed(get_map_soln)(model, loud) for model in models
        )

    return map_solns


def build_model_and_get_map_soln(
    lc, pri_t0, pri_p, pri_rprs, pri_m_star, pri_r_star, verbose=False
):
    """
    Build the transit light curve model.

    Parameters
    ----------
    lc : `~lightkurve.LightCurve`
        A light curve object with the data.

    pri_t0 : float
        Initial guess for mid-transit time.

    pri_p : float
        Initial guess for period.

    pri_rprs : float
        Initial guess for planet-to-star radius ratio.

    pri_m_star : ndarray
        Mean and standard deviation of the stellar mass estimate
        in solar masses.

    pri_r_star : ndarray
        Mean and standard deviation of the stellar radius estimate
        in solar radii.

    verbose : bool, optional
        Print details of optimization.

    Returns
    -------
    model : `~pymc3.model`
        A model object.

    map_soln : dict
        A dictionary with the maximum a posteriori estimates of the variables.
    """
    # Ignore warnings from theano, unless specified elsewise
    if not verbose:
        warnings.filterwarnings(
            action='ignore',
            category=FutureWarning,
            module='theano'
        )

    # Ensure right data type for theano
    dtype = np.dtype('float64')
    dts = [arr.dtype for arr in [lc.time, lc.flux, lc.flux_err]]
    if not all(dt is dtype for dt in dts):
        lc.time = np.array(lc.time, dtype=dtype)
        lc.flux = np.array(lc.flux, dtype=dtype)
        lc.flux_err = np.array(lc.flux_err, dtype=dtype)

    # Estimate flux uncertainties if not given
    idx_nan = np.isnan(lc.flux_err)
    if idx_nan.any():
        mad = median_abs_deviation(lc.flux, scale='normal')
        lc.flux_err[idx_nan] = mad

    # Define the model for the light curve
    with pm.Model() as model:
        # Stellar mass
        m_star = pm.Normal(
            'm_star',
            mu=pri_m_star[0],
            sigma=pri_m_star[1]
        )

        # Stellar radius
        r_star = pm.Normal(
            'r_star',
            mu=pri_r_star[0],
            sigma=pri_r_star[1]
        )

        # Quadratic limb-darkening parameters
        u = xo.distributions.QuadLimbDark(
            'u',
            testval=np.array([0.3, 0.2])
        )

        # Radius ratio
        r = pm.Uniform(
            'r',
            lower=0.,
            upper=1.,
            testval=pri_rprs
        )

        # Impact parameter
        b = xo.distributions.ImpactParameter(
            'b',
            ror=r,
        )

        # Period
        logperiod = pm.Uniform(
            'logperiod',
            lower=-2.3,  # 0.1 d
            upper=3.4,  # 30 d
            testval=np.log(pri_p)
        )
        period = pm.Deterministic('period', tt.exp(logperiod))

        # Mid-transit time
        t0 = pm.Uniform(
            't0',
            lower=lc.time.min(),
            upper=lc.time.max(),
            testval=pri_t0
        )

        # Keplerian orbit
        orbit = xo.orbits.KeplerianOrbit(
            m_star=m_star,
            r_star=r_star,
            period=period,
            t0=t0,
            b=b
        )

        # Model transit light curve
        light_curves = xo.LimbDarkLightCurve(
            u).get_light_curve(orbit=orbit, r=r*r_star, t=lc.time)
        transit_model = pm.math.sum(light_curves, axis=-1)

        # The baseline flux
        f0 = pm.Normal(
            'f0',
            mu=np.median(lc.flux),
            sigma=median_abs_deviation(lc.flux, scale='normal')
        )

        # The full model
        lc_model = transit_model + f0

        #######################
        # Track some parameters
        #######################

        # Track transit depth
        pm.Deterministic('depth', r**2)

        # Track planet radius (in Earth radii)
        pm.Deterministic(
            'rearth',
            r*r_star*(units.solRad/units.earthRad).si.scale
        )

        # Track semimajor axis (in AU)
        au_per_rsun = (units.solRad/units.AU).si.scale
        pm.Deterministic('a', orbit.a*au_per_rsun)

        # Track system scale
        pm.Deterministic('aRs', orbit.a/r_star)  # normalize by stellar radius

        # Track inclination
        pm.Deterministic('incl', np.rad2deg(orbit.incl))

        # Track transit duration
        # Seager and Mallen-Ornelas (2003) Eq. 3
        sini = np.sin(orbit.incl)
        t14 = (
            (period/np.pi) *
            np.arcsin((r_star/orbit.a*sini) * np.sqrt((1.+r)**2 - b**2))
        )*24.*60.  # min
        t14 = pm.Deterministic('t14', t14)

        # Track stellar density (in cgs units)
        rho_star = pm.Deterministic('rho_star', orbit.rho_star)

        # Track stellar density (in units of solar density)
        rho_sol = (units.solMass/(4./3.*np.pi*units.solRad**3)).cgs.value
        pm.Deterministic('rho_star_sol', orbit.rho_star/rho_sol)

        # Track x2
        x2 = pm.math.sum(((lc.flux-lc_model)/lc.flux_err)**2)
        x2 = pm.Deterministic('x2', x2)

#         # Fit for variance
#         logs2 = pm.Normal('logs2', mu=np.log(np.var(lc.flux)), sigma=1)
#         sigma = pm.Deterministic('sigma', pm.math.sqrt(pm.math.exp(logs2)))

        # The likelihood function
        pm.Normal('obs', mu=lc_model, sigma=lc.flux_err, observed=lc.flux)

        # Fit for the maximum a posteriori parameters
        map_soln = xo.optimize(
            start=model.test_point, verbose=verbose
        )
        map_soln = xo.optimize(
            start=map_soln, vars=[f0, period, t0, r], verbose=verbose
        )
        map_soln = xo.optimize(
            start=map_soln, vars=rho_star, verbose=verbose
        )
        map_soln = xo.optimize(
            start=map_soln, vars=t14, verbose=verbose
        )
        map_soln = xo.optimize(
            start=map_soln, verbose=verbose
        )

    # Reset warning filter
    warnings.resetwarnings()

    return model, map_soln


def downselect(
    models, map_solns, n_data, n_param, threshold=1e-6, return_idx=False
):
    """
    Downselect the models to test using the BMA weights.

    Parameters
    ----------
    models : list
        A list of `~pymc3.model` objects.

    map_solns : list
        A list of MAP estimates.

    n_data : int
        The number of data points.

    n_param : int
        The number of parameters in the model

    threshold : float, optional
        The threshold for ignoring low-weighted scenarios. Defaults to 1 ppm.

    return_idx : bool, optional
        Return the indices of the top models.

    Returns
    -------
    models_subset : list
        The subset of `~pymc3.model` objects.

    map_solns_subset : list
        The subset of dictionaries with the MAP estimates for the models.

    idx_subset : ndarray
        Indices of the top models. Only returned if ``return_idx`` is `True`.
    """
    # Calculate BICs and weights
    lnlikes = np.array([
        model.logp(map_soln) for model, map_soln in zip(models, map_solns)
    ])
    bics = np.log(n_data) * n_param - 2 * lnlikes
    dbics = bics - np.min(bics)
    weights = np.exp(dbics / -2) / np.sum(np.exp(dbics / -2))

    # Downselect
    idx_subset = np.where(np.array(weights) > threshold)[0]
    models_subset = [
        models[i] for i in idx_subset
    ]
    map_solns_subset = [
        map_solns[i] for i in idx_subset
    ]

    if return_idx:
        return models_subset, map_solns_subset, idx_subset

    else:
        return models_subset, map_solns_subset


def flatten(
    lc, t0=None, period=None, duration=1./24., window_length=3./24., **kwargs
):
    """
    Flatten a light curve using the `wotan` package.

    Parameters
    ----------
    lc : `~lightkurve.LightCurve`
        A light curve object with the data.

    t0 : float or iterable, optional
        Mid-transit time of transit signal(s) to mask.

    period : float or iterable, optional
        Period of transit signal(s) to mask.

    duration : float, optional
        Duration of transit signal to mask. Defaults to 1 hr.

    window_length : float, optional
        Length of the filter window for `wotan.flatten()`. Defaults to 3 hr.

    kwargs : dict, optional
        Any extra keyword arguments to pass to `wotan.flatten()`.

    Returns
    -------
    lc : `~lightkurve.LightCurve`
        A light curve object with the flattened light curve.

    trend : ndarray
        The removed flux trend. Only returned if ``return_trend`` is `True`.
    """
    # Mask transits if any ephemerides are given
    mask = np.zeros_like(lc.time, dtype=bool)
    if t0 is not None:
        t0s = np.array(t0).flatten()
        periods = np.array(period).flatten()
        for t0, period in zip(t0s, periods):
            mask += wotan.transit_mask(
                time=lc.time,
                T0=t0,
                period=period,
                duration=duration
            )

    # Flatten the light curve
    flux_flat = wotan.flatten(
        lc.time,
        lc.flux,
        window_length=window_length,
        mask=mask,
        **kwargs
    )

    # Return trend if return_trend=True
    return_trend = False
    if isinstance(flux_flat, tuple):
        return_trend = True
        flux_flat, trend = flux_flat

    lcflat = lc.copy()
    lcflat.flux = flux_flat

    if return_trend:
        return (lcflat, trend)
    else:
        return lcflat


def get_forecast_window(start=None, size=30*units.day, cadence=2*units.min):
    """
    Get an array of times in JD to forecast transits.

    Defaults to an array covering the next 30 days at 2-min cadence.

    Parameters
    ----------
    start : `~astropy.time.Time`, optional
        Start of the forecast window. `None` defaults to now.

    size : float or `~astropy.units.Quantity`, optional
        Size of the forecast window. Defaults to days if unit not specified.

    cadence : float or `~astropy.units.Quantity`, optional
        Cadence of the times in the forecast window. Defaults to 2-min if
        unit not specfied.

    Returns
    -------
    tforecast : `~astropy.time.Time`
        Array of times.
    """
    if start is None:
        start = Time.now()
    if type(size) is units.Quantity:
        size = size.to(units.day)
    else:
        size = size*units.day
    if type(cadence) is units.Quantity:
        cadence = cadence.to(units.day)
    else:
        cadence = cadence*units.day

    tforecast = ap.time_grid_from_range(
        [start, start+size], cadence
    )

    return tforecast


def get_priors_from_tic(tic_id):
    """
    Get stellar mass and radius priors from the TESS Input Catalog.

    Parameters
    ----------
    tic_id : int
        A TESS Input Catalog ID.

    Returns
    -------
    pri_m_star : ndarray
        Mean and standard deviation of the stellar mass estimate
        in solar masses.

    pri_r_star : ndarray
        Mean and standard deviation of the stellar radius estimate
        in solar radii.
    """
    tic_params = tls.catalog_info(TIC_ID=tic_id)
    ld, M_star, M_star_l, M_star_u, R_star, R_star_l, R_star_u = tic_params

    # Guard against bad values in the TIC
    if not np.isfinite(R_star):
        R_star = 1.
    if not np.isfinite(M_star):
        M_star = 1.
    if not np.isfinite(M_star_u):
        M_star_u = 0.1
    if not np.isfinite(M_star_l):
        M_star_l = 0.1
    if not np.isfinite(R_star_u):
        R_star_u = 0.1
    if not np.isfinite(R_star_l):
        R_star_l = 0.1

    # Use stellar parameters from TIC
    pri_m_star_mean = M_star
    pri_m_star_err = (M_star_u+M_star_l)/2.
    pri_m_star = np.append(pri_m_star_mean, pri_m_star_err)

    pri_r_star_mean = R_star
    pri_r_star_err = (R_star_u+R_star_l)/2.
    pri_r_star = np.append(pri_r_star_mean, pri_r_star_err)

    return pri_m_star, pri_r_star


def get_trial_ephemerides(
    t0, period=None, rprs=None,
    j_max=1, k_max=1, min_period=0., max_period=np.inf
):
    """
    Get the set of trial ephemerides.

    Parameters
    ----------
    t0 : iterable
        The mid-transit times of candidate signals.

    period : iterable, optional
        The periods of candidate signals. Calculated from `t0` if not provided.

    rprs : iterable, optional
        The planet-to-star radius ratios. Defaults to 0.1.

    j_max : int, optional
        The maximum j value for j:k aliased periods to calculate.

    k_max : int, optional
        The maximum k value for j:k aliased periods to calculate.

    min_period : float, optional
        The minimum period of interest.

    max_period : float, optional
        The maximum period of interest.

    Returns
    -------
    ephem : pandas.DataFrame
        The trial ephemerides.
    """
    # Use numpy array
    t0 = np.array(t0)

    # Make array of radius ratios, if not given
    if rprs is None:
        rprs = np.ones_like(t0) * 0.1

    # Calculate periods, if not given
    if period is None:
        t0_arr = np.array([])
        period_arr = np.array([])
        rprs_arr = np.array([])

        for i, t in enumerate(t0[:-1]):
            p = np.abs(t0[i + 1:] - t)
            t0_arr = np.append(t0_arr, t*np.ones_like(p))
            period_arr = np.append(period_arr, p)
            rprs_arr = np.append(rprs_arr, rprs[i]*np.ones_like(p))
        t0 = t0_arr
        period = period_arr
        rprs = rprs_arr

    # Ensure that the arrays are the right length
    array_lengths_good = all(len(arr) == len(t0) for arr in [t0, period, rprs])
    if not array_lengths_good:
        raise ValueError(
            't0, period, and rprs arrays must have the same length.'
        )

    # Calculate period ratios
    jj, kk = np.meshgrid(
        np.arange(j_max)+1,
        np.arange(k_max)+1
    )
    period_ratio_matrix = jj/kk
    period_ratios = np.unique(period_ratio_matrix.flatten())

    # Compile list of ephemerides
    period = np.array(period)
    rprs = np.array(rprs)
    t0s = (
        t0[:, np.newaxis] * np.ones_like(period_ratios)[np.newaxis, :]
    ).ravel()
    periods = (
        period[:, np.newaxis] * period_ratios[np.newaxis, :]
    ).ravel()
    rprss = (
        rprs[:, np.newaxis] * np.ones_like(period_ratios)[np.newaxis, :]
    ).ravel()

    # Truncate to periods within range
    idx_inrange = np.logical_and(
        periods >= min_period,
        periods <= max_period
    )
    t0s = t0s[idx_inrange]
    periods = periods[idx_inrange]
    rprss = rprss[idx_inrange]

    ephem = pd.DataFrame({
        't0': t0s,
        'period': periods,
        'rprs': rprss
    }).drop_duplicates()

    # Sort by t0 and then period
    ephem = ephem.sort_values(['t0', 'period']).reset_index(drop=True)

    return ephem


def sample_from_model(
    model, map_soln, tune=500, draws=200, chains=5, cores=None, step=None
):
    """
    Sample from the transit light curve model.

    Parameters
    ----------
    model : `~pymc3.model`
        A model object.

    map_soln : dict
        A dictionary with the maximum a posteriori estimates of the variables.

    tune : int, optional
        The number of iterations to tune.

    draws : int, optional
        The number of samples to draw.

    chains : int, optional
        The number of chains to sample.

    cores : int, optional
        The number of cores to run in parallel.

    step : function, optional
        A step function.

    Returns
    -------
    trace : `~pymc3.backends.base.MultiTrace`
        A ``MultiTrace`` object that contains the samples.
    """
    # Use 1 CPU thread per chain, unless specified otherwise
    if cores is None:
        cores = min(chains, mp.cpu_count())

    # Ignore FutureWarnings
    warnings.simplefilter('ignore', FutureWarning)

    with model:
        if step is None:
            step = xo.get_dense_nuts_step(target_accept=0.95)

        trace = pm.sample(
            tune=tune,
            draws=draws,
            start=map_soln,
            chains=chains,
            cores=cores,
            step=step
        )

    # Reset warnings
    warnings.resetwarnings()

    return trace


def sample_from_models(models, map_solns, **kwargs):
    """
    Sample from a list of models.

    Parameters
    ----------
    models : list
        A list of `~pymc3.model` objects.

    map_solns : list
        A list of dictionaries with the MAP estimates for the models.

    **kwargs
        Additional keyword arguments passed to ``sample_from_model``.

    Returns
    -------
    traces : list
        A list of ``MultiTrace`` objects that contain the samples.
    """
    traces = [
        sample_from_model(model, map_soln, **kwargs)
        for model, map_soln in zip(models, map_solns)
    ]

    return traces
