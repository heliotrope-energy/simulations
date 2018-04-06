#running PVLib simulation in steps

from pvlib.pvsystem import PVSystem, retrieve_sam
from pvlib.tracking import SingleAxisTracker
from pvlib.modelchain import ModelChain
from pvlib.forecast import GFS, NAM, NDFD, HRRR, RAP
import pandas as pd
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


#set up system
sandia_modules = retrieve_sam('sandiamod')
cec_inverters = retrieve_sam('cecinverter')
module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
inverter = cec_inverters['SMA_America__SC630CP_US_315V__CEC_2012_']

system = PVSystem(surface_tilt=20, surface_azimuth=200, module_parameters=module, inverter_parameters=inverter, modules_per_string=15, strings_per_inverter=300)

latitude, longitude, tz = 32.2, -110.9, 'US/Arizona'

start = pd.Timestamp(datetime.date.today() - pd.Timedelta(days=20), tz=tz)

end = start + pd.Timedelta(days=7)
# model = GFS()

model = NDFD()

data = model.get_processed_data(latitude, longitude, start, end)

mc = ModelChain(system, model.location)

for dat in data.index:
    mc.run_model(dat, weather=data)
    mc.system.surface_tilt = randint(20, 40)
    print(mc.system.surface_tilt)
    print(mc.aoi)
