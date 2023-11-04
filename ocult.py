import ephem
import numpy as np
import os
import datetime as dt
import xarray as X

class Eclipse(object):

    def __init__(
            self, event_name, start_time, end_time, 
            dtime_sec=60., 
            lats = np.linspace(0,90,num=181),
            lons = np.linspace(-180,180,num=361),
            alts = np.array([100]),
        ):
        self.event_name = event_name
        self.start_time = start_time
        self.end_time = end_time
        self.dtime_sec = dtime_sec
        self.lats = lats
        self.lons = lons
        self.alts = alts
        self.dir = f"tmp/{self.event_name}/"
        os.makedirs(self.dir, exist_ok=True)
        self.run_eclipse()
        return

    def intersection(self, r0,r1,d,n_s=100):
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

    def run_eclipse(self):
        # Location
        obs = ephem.Observer()
        n_t = int((self.end_time-self.start_time).total_seconds()/self.dtime_sec)
        n_alts = len(self.alts)
        n_lats = len(self.lats)
        n_lons = len(self.lons)
        
        self.p = np.zeros([n_t,n_alts,n_lats,n_lons])
        times = np.arange(n_t)*self.dtime_sec
        self.dts = []
        self.filename = self.dir + "occult_fun.nc"
        if not os.path.exists(self.filename):
            for ti, t in enumerate(times):
                print("Time %1.2f (s)"%(t))
                time = self.start_time + dt.timedelta(seconds=ti)
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
                for ai,alt in enumerate(self.alts):
                    for lai,lat in enumerate(self.lats):
                        for loi,lon in enumerate(self.lons):
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
                                if s < 1e-3:
                                    percent_eclipse=1.0
                                else:
                                    percent_eclipse=self.intersection(r_moon,r_sun,s,n_s=100)

                            if np.degrees(sun.alt) <= r_sun:
                                if np.degrees(sun.alt) <= -r_sun:
                                    percent_eclipse = np.nan
                                else:
                                    percent_eclipse = (
                                        1.0-(
                                            (np.degrees(sun.alt) + r_sun) / (2.0*r_sun)
                                        )*(1.0-percent_eclipse)
                                    )
                    
                            self.p[ti,ai,lai,loi] = percent_eclipse
                self.dts.append(t)
            o = {
                "coords": {
                    "t": {"dims": "t", "data": self.dts, "attrs": {"units": "s"}},
                    "latitude": {"dims": "latitude", "data": self.lats, "attrs": {"units": "deg"}},
                    "longitude": {"dims": "longitude", "data": self.lons, "attrs": {"units": "deg"}},
                    "altitude": {"dims": "altitude", "data": self.alts, "attrs": {"units": "km"}},
                },
                "attrs": {
                    "title": "Occultation function", 
                    "desc": self.event_name, "start": self.start_time.strftime("%Y-%m-%dT%H:%M"),
                    "end": self.end_time.strftime("%Y-%m-%dT%H:%M")
                },
                "dims": ["t", "altitude", "latitude", "longitude"],
                "data_vars": {
                    "of": {
                        "dims": ("t", "altitude", "latitude", "longitude"), 
                        "data": self.p
                    },
                },
            }
            self.ds = X.Dataset.from_dict(o)
            self.ds.to_netcdf(self.filename)
        else:
            self.ds = X.open_dataset(self.filename)
        return

    def fetch_of(self, date, lats=None, lons=None):
        ti = (date-self.start_time).total_seconds()/self.dtime_sec
        self.ds["data_vars"]["of"][ti,:,:,:]
        return

if __name__ == "__main__":
    e = Eclipse(
        "2023Oct",
        dt.datetime(2023,10,14,15),
        dt.datetime(2023,10,14,21),
    )