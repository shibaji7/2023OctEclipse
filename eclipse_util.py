import ephem
import numpy as np


def intersection(r0,r1,d,n_s=100):
    A1 = np.zeros([n_s,n_s])
    A2 = np.zeros([n_s,n_s])
    I = np.zeros([n_s,n_s])
    x = np.linspace(-2.0*r0,2.0*r0,num=n_s)
    y = np.linspace(-2.0*r0,2.0*r0,num=n_s)
    xx, yy = np.meshgrid(x,y)
    A1[np.sqrt((xx+d)**2.0+yy**2.0) < r0] = 1.0
    n_sun = np.sum(A1)
    A2[np.sqrt(xx**2.0+yy**2.0) < r1] = 1.0
    S = A1+A2
    I[S>1] = 1.0
    eclipse = np.sum(I)/n_sun
    return eclipse

def occultation_function(time, lat, lon, alt):
    obs = ephem.Observer()
    t0 = ephem.date(
        (
            time.year,
            time.month,
            time.day,
            time.hour,
            time.minute,
            time.second,
        )
    )
    obs.lon, obs.lat = '%1.2f'%(lon), '%1.2f'%(lat) # ESR
    obs.elevation = alt
    obs.date = t0
    sun, moon = ephem.Sun(), ephem.Moon()
    # Output list
    results = []
    seps = []
    sun.compute(obs)
    moon.compute(obs)
    r_sun = (sun.size/2.0)/3600.0
    r_moon = (moon.size/2.0)/3600.0
    s = np.degrees(ephem.separation((sun.az, sun.alt), (moon.az, moon.alt)))
    percent_eclipse = 0.0
    if s < (r_moon + r_sun):
        percent_eclipse = 1.0 if s < 1e-3 else intersection(r_moon,r_sun,s,n_s=100)
    if np.degrees(sun.alt) <= r_sun:
        percent_eclipse = np.nan if np.degrees(sun.alt) <= -r_sun else (
            1.0 - ((np.degrees(sun.alt) + r_sun) / (2.0*r_sun))*(1.0-percent_eclipse)
        )
    return percent_eclipse

def helper_rti_get_eclipse(dates, lats, lons, alt=100):
    pe = np.nan * np.zeros((len(dates), len(lats)))
    for i, d in enumerate(dates):
        for j, lat, lon in zip(range(len(lats)), lats, lons):
            pe[i, j] = occultation_function(d, lat, lon, alt)
    return pe