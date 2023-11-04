from radar import Radar
from rtiUtils import RTI
import datetime as dt


if __name__ == "__main__":
    rads = ["fhe", "fhw"]
    radars = {
        "fhe": Radar("fhe"),
        "fhw": Radar("fhw"),
    }
    dates = [dt.datetime(2023,10,14), dt.datetime(2023,10,15)]
    rti = RTI(dates, num_subplots=1)
    #rti.addParamPlot(radars["fhe"], 10, "2023 Oct 14/FHE/10", oclt=True)
    rti.addParamPlot(radars["fhw"], 10, "2023 Oct 14/FHW/10", xlabel="Time (UT)", oclt=True)
    rti.save("tmp/rti.png")