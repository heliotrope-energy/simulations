#running PVLib forecasting functions

from pvlib.pvsystem import PVSystem, retrieve_sam
from pvlib.tracking import SingleAxisTracker
from pvlib.modelchain import ModelChain
from pvlib.forecast import GFS, NAM, NDFD, HRRR, RAP
import pandas as pd
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sandia_modules = retrieve_sam('sandiamod')

cec_inverters = retrieve_sam('cecinverter')

module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']

inverter = cec_inverters['SMA_America__SC630CP_US_315V__CEC_2012_']

# system = SingleAxisTracker(module_parameters=module,
#                             inverter_parameters=inverter,
#                             modules_per_string=15,
#                             strings_per_inverter=300)

system = PVSystem(surface_tilt=20, surface_azimuth=200, module_parameters=module, inverter_parameters=inverter, modules_per_string=15, strings_per_inverter=300)




latitude, longitude, tz = 32.2, -110.9, 'US/Arizona'

start = pd.Timestamp(datetime.date.today(), tz=tz)

end = start + pd.Timedelta(days=7)

irrad_vars = ['ghi', 'dni', 'dhi']

# Global Forecast System (GFS), defaults to 0.5 degree resolution
# 0.25 deg available
model = GFS()

#has raw data and processed data functions
#NOTE: contains wind components (east/west and north/south!!!!)
#NOTE: includes cloud cover modeling, by different cloud levels!
#NOTE: how to generate images? do we even need those?

data = model.get_processed_data(latitude, longitude, start, end)

mc = ModelChain(system, model.location)

print(data.index[0:2])

mc.run_model(data.index[0:2], weather=data)

mc.total_irrad.plot()

plt.ylabel('Plane of array irradiance ($W/m^2$)')

plt.legend(loc='best')

plt.savefig("../plots/poa_fixed_short.png")

plt.close()

mc.temps.plot();
plt.ylabel('Temperature (C)')
plt.savefig("../plots/temp_fixed_short.png")
plt.close()

mc.ac.fillna(0).plot();

plt.ylim(0, None);
plt.ylabel('AC Power (W)');
plt.savefig("../plots/ac_pwr_fixed_short.png")
