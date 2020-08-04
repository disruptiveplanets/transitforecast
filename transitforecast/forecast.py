"""Forecasting transit events."""
import exoplanet as xo
import multiprocessing
import numpy as np
import pymc3 as pm
import theano.tensor as tt
import transitleastsquares as tls
import warnings
from astropy import units
from astropy.time import Time
from scipy.stats import median_abs_deviation

__all__ = [
    'build_model',
    'get_forecast_window',
    'get_priors_from_tic',
    'sample_from_model'
]


def build_model(
    lc, pri_t0, pri_p, pri_rprs, pri_m_star, pri_r_star, tforecast,
    verbose=False
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

    tforecast : iterable
        The times for the forecast. Assumes same units as ``lc.time``.

    verbose : bool
        Print details of optimization.

    Returns
    -------
    model : `~pymc3.model`
        A model object.

    map_soln : dict
        A dictionary with the maximum a posteriori estimates of the variables.
    """
    # Ensure right data type for theano
    dtype = np.dtype('float64')
    dts = [arr.dtype for arr in [lc.time, lc.flux, lc.flux_err]]
    if not all(dt is dtype for dt in dts):
        lc.time = np.array(lc.time, dtype=dtype)
        lc.flux = np.array(lc.flux, dtype=dtype)
        lc.flux_err = np.array(lc.flux_err, dtype=dtype)

    # Define the model for the light curve
    with pm.Model() as model:
        # Stellar mass
        m_star = pm.Normal(
            'm_star',
            mu=pri_m_star[0],
            sd=pri_m_star[1]
        )

        # Stellar radius
        r_star = pm.Normal(
            'r_star',
            mu=pri_r_star[0],
            sd=pri_r_star[1]
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
        pm.Deterministic('light_curves', light_curves)
        transit_model = pm.math.sum(light_curves, axis=-1)
        transit_model = pm.Deterministic('transit_model', transit_model)

        # The baseline flux
        f0 = pm.Normal(
            'f0',
            mu=np.median(lc.flux),
            sd=median_abs_deviation(lc.flux, scale='normal')
        )

        # The full model
        lc_model = pm.Deterministic('lc_model', transit_model+f0)

        ########################
        # Forecast transits
        ########################

        texp = np.median(np.diff(tforecast))
        lcforecast = xo.LimbDarkLightCurve(
            u).get_light_curve(orbit=orbit, r=r*r_star, t=tforecast, texp=texp)
        forecast = pm.math.sum(lcforecast, axis=-1)
        forecast = pm.Deterministic('forecast', forecast)

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
#         logs2 = pm.Normal('logs2', mu=np.log(np.var(lc.flux)), sd=1)
#         sigma = pm.Deterministic('sigma', pm.math.sqrt(pm.math.exp(logs2)))

        # The likelihood function
        pm.Normal('obs', mu=lc_model, sd=lc.flux_err, observed=lc.flux)

        # Fit for the maximum a posteriori parameters
        with warnings.catch_warnings():
            if not verbose:
                warnings.simplefilter('ignore', FutureWarning)
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

    return model, map_soln


def get_forecast_window(size=30*units.day, cadence=2*units.min, start=None):
    """
    Get an array of times in JD to forecast transits.

    Defaults to an array covering the next 30 days at 2-min cadence.

    Parameters
    ----------
    size : float or `~astropy.units.Quantity`
        Size of the forecast window. Defaults to days if unit not specified.

    cadence : float or `~astropy.units.Quantity`
        Cadence of the times in the forecast window. Defaults to 2-min if
        unit not specfied.

    start : `~astropy.time.Time`
        Start of the forecast window. `None` defaults to now.

    Returns
    -------
    tforecast : `~numpy.ndarray`
        Array of times in JD.
    """
    if type(size) is units.Quantity:
        size = size.to(units.day)
    else:
        size = size*units.day
    if type(cadence) is units.Quantity:
        cadence = cadence.to(units.day)
    else:
        cadence = cadence*units.day
    if start is None:
        start = Time.now()

    forecast_times = start + np.arange(0, size.value, cadence.value)
    tforecast = forecast_times.jd

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
    # Use all CPU threads, unless specified otherwise
    if cores is None:
        cores = multiprocessing.cpu_count()

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
