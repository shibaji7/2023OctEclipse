import pandas as pd
import numpy as np
from numpy import unravel_index
import os
import datetime as dt
import xarray as X
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Tahoma", "DejaVu Sans",
                                   "Lucida Grande", "Verdana"]
import matplotlib as mpl
mpl.rcParams.update({"xtick.labelsize": 12, "ytick.labelsize":12, "font.size":12})
import cartopy.crs as ccrs
from cartopy.feature.nightshade import Nightshade
import datetime as dt
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import eclipse_util as eutils

class Eclipse(object):

    def __init__(
            self, event_name, start_time, end_time, 
            dtime_sec=60., 
            lats = np.linspace(-90,90,num=181),
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
        self.speed_of_the_center()
        return

    def draw_images(
        self, date, alt=100, to=ccrs.Orthographic(-90, 30), 
        cb=False, figsize=(5,5), dpi=240, figname=None
    ):
        p = self.fetch_oclt(date, alt)
        plat, plon = self.find_max_loc_of(date, alt)
        fig = plt.figure(dpi=dpi, figsize=figsize)
        ax = plt.axes(projection=to)
        ax.set_global()
        ax.add_feature(Nightshade(date, alpha=0.3))
        ax.coastlines()
        p = np.ma.masked_invalid(p)
        im = ax.contourf(
            self.lons,
            self.lats,
            p,
            transform=ccrs.PlateCarree(),
            cmap="gray_r", alpha=0.6,
            levels=[0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
        )
        ax.scatter(
            plon, plat, 
            transform=ccrs.PlateCarree(),
            s=10, color="r", alpha=0.6
        )
        #if cb: _add_colorbar(fig, ax, im)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.3, 
            color="black", alpha=0.5, linestyle="--", draw_labels=True)
        gl.xlocator = mticker.FixedLocator(np.arange(-180,180,60))
        gl.ylocator = mticker.FixedLocator(np.arange(-90,90,30))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        figname = figname if figname else self.dir + f"{date.strftime('%H:%M')}.png"
        fig.savefig(figname, bbox_inches="tight")
        return ax, fig

    def run_eclipse(self):
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
                time = self.start_time + dt.timedelta(seconds=t)
                print("Time %1.2f (s)"%(t), time)
                for ai,alt in enumerate(self.alts):
                    for lai,lat in enumerate(self.lats):
                        for loi,lon in enumerate(self.lons):
                            self.p[ti,ai,lai,loi] = eutils.occultation_function(
                                time, lat, lon, alt
                            )
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

    def fetch_oclt(self, time, alt=100):
        if (time>=self.start_time) and (time<self.end_time):
            itime = int((time-self.start_time).total_seconds()/self.dtime_sec)
            ialt = self.alts.tolist().index(alt)
            p = self.ds.variables["of"][itime,ialt,:,:]
        else:
            p = np.nan*np.zeros((len(self.lats), len(self.lons)))
        return p

    def find_max_loc_of(self, time, alt=100):
        of = np.array(self.fetch_oclt(time, alt))
        of[of<0.1] = np.nan
        amax = unravel_index(np.nanargmax(of), of.shape)
        return (self.lats[amax[0]], self.lons[amax[1]])

    def speed_of_the_center(self, dates=None, alt=100):
        dates = dates if dates else [
            self.start_time + dt.timedelta(seconds=i * self.dtime_sec)
            for i in range(int((self.end_time-self.start_time).total_seconds()/self.dtime_sec))
        ]
        self.speed_details = dict(
            great_circle=[],
            latitude=[],
            longitude=[],
            speed=[],
            dates=dates,
        )
        for d in dates:
            lat, lon = self.find_max_loc_of(d, alt)
            self.speed_details["latitude"].append(lat)
            self.speed_details["longitude"].append(lon)
        self.speed_details["great_circle"].append(np.nan)
        self.speed_details["speed"].append(np.nan)
        for i in range(len(dates)-1):
            d = self.great_circle(
                self.speed_details["longitude"][i],
                self.speed_details["latitude"][i],
                self.speed_details["longitude"][i+1],
                self.speed_details["latitude"][i+1],
            )
            self.speed_details["great_circle"].append(d)
            self.speed_details["speed"].append(d/self.dtime_sec)
        self.speed_details = pd.DataFrame.from_dict(self.speed_details)
        return

    def great_circle(self, lon1, lat1, lon2, lat2):
        from math import radians, degrees, sin, cos, asin, acos, sqrt
        on1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        return 6371 * (
            acos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon1 - lon2))
        )

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
    
    # e.find_max_loc_of(dt.datetime(2023,10,14,17,45))
    # # e.find_max_loc_of(dt.datetime(2023,10,14,15,11))
    # # e.find_max_loc_of(dt.datetime(2023,10,14,15,12))
    # for t in range(240):
    #     e.draw_images(dt.datetime(2023,10,14,15)+dt.timedelta(minutes=t))