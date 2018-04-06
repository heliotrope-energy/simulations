#testing TMY data parsing + simulation

import pvlib
import pandas as pd
from pvlib.pvsystem import PVSystem, retrieve_sam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from random import randint
import numpy as np
loc = "/Users/edwardwilliams/Documents/research/heliotrope/simulations/data/722745TYA.CSV"


class FixedPolicyTracker:
    def __init__(self, angle, azimuth):
        self.name="Fixed at {}".format(angle)
        self.angle = angle
        self.azimuth = azimuth
    def get_angle(self, state):
        return self.angle
    def get_azimuth(self):
        return self.azimuth
    def update(self, state, reward):
        '''
        Doesn't do anything
        '''
        return None

class RandomTracker:
    def __init__(self, min, max, azimuth):
        self.name="Random from {} to {}".format(min, max)
        self.max = max
        self.min = min
        self.azimuth = azimuth
    def get_angle(self, state):
        return randint(self.min, self.max)
    def get_azimuth(self):
        return self.azimuth
    def update(self, state, reward):
        '''
        Doesn't do anything
        '''
        return None

class AstroTracker:
    def __init__(self, azimuth, fallback_angle=30):
        self.name="astronomical"
        self.azimuth = azimuth
        self.fallback_angle = fallback_angle
    def get_angle(self, state):
        angle_pos = pvlib.tracking.singleaxis(state['apparent_zenith'], state['azimuth'], backtrack=False)
        surface_tilt = angle_pos['tracker_theta']
        if np.isnan(surface_tilt[0]):
            #TODO: fix this
            surface_tilt = self.fallback_angle
        return surface_tilt
    def get_azimuth(self):
        return self.azimuth
    def update(self, state, reward):
        '''
        Doesn't do anything
        '''
        return None

tmy_data, meta = pvlib.tmy.readtmy3(filename=loc)

# surface_azimuth = 90 # pvlib uses 0=North, 90=East, 180=South, 270=West convention
albedo = 0.2 #assume fixed, changing in reality



# create pvlib Location object based on meta data
sand_point = pvlib.location.Location(meta['latitude'], meta['longitude'], tz='US/Arizona',
                                     altitude=meta['altitude'], name=meta['Name'].replace('"',''))



def run_sim_on_tracker(tracker, n_hours=500):
    '''
    Returns power
    '''
    total = pd.DataFrame()
    surface_azimuth = tracker.get_azimuth()
    for i in range(500):
        current_step_data = tmy_data[tmy_data.index[i]:tmy_data.index[i]]

        solpos = pvlib.solarposition.get_solarposition(current_step_data.index, sand_point.latitude, sand_point.longitude)

        surface_tilt = tracker.get_angle(solpos)

        # the extraradiation function returns a simple numpy array
        # instead of a nice pandas series. We will change this
        # in a future version
        dni_extra = pvlib.irradiance.extraradiation(current_step_data.index)
        dni_extra = pd.Series(dni_extra, index=current_step_data.index)

        # print(dni_extra)

        airmass = pvlib.atmosphere.relativeairmass(solpos['apparent_zenith'])

        # print(airmass)

        poa_sky_diffuse = pvlib.irradiance.haydavies(surface_tilt, surface_azimuth,
                                                     current_step_data['DHI'], current_step_data['DNI'], dni_extra,
                                                     solpos['apparent_zenith'], solpos['azimuth'])

    # print(poa_sky_diffuse)

        poa_ground_diffuse = pvlib.irradiance.grounddiffuse(surface_tilt, current_step_data['GHI'], albedo=albedo)

        aoi = pvlib.irradiance.aoi(surface_tilt, surface_azimuth, solpos['apparent_zenith'], solpos['azimuth'])

        poa_irrad = pvlib.irradiance.globalinplane(aoi, current_step_data['DNI'], poa_sky_diffuse, poa_ground_diffuse)


        pvtemps = pvlib.pvsystem.sapm_celltemp(poa_irrad['poa_global'], current_step_data['Wspd'], current_step_data['DryBulb'])

        sandia_modules = retrieve_sam('sandiamod')
        cec_inverters = retrieve_sam('cecinverter')
        module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
        inverter = cec_inverters['SMA_America__SC630CP_US_315V__CEC_2012_']

        effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(poa_irrad.poa_direct, poa_irrad.poa_diffuse, airmass, aoi, module)

        sapm_out = pvlib.pvsystem.sapm(effective_irradiance, pvtemps.temp_cell, module)

        tracker.update(solpos, sapm_out)

        total = total.append(sapm_out)
        total['p_cumulative_{}'.format(tracker.name)] = total.p_mp.cumsum()

    return total

fixed = FixedPolicyTracker(30, 90) #azimuth angle
rand = RandomTracker(10, 80, 90)
astro = AstroTracker(90)

trackers = [fixed, rand, astro]

totals = {tracker.name:run_sim_on_tracker(tracker) for tracker in trackers}

fig = plt.figure()
ax = fig.add_subplot(111)


for name, total in totals.items():
    total[['p_cumulative_{}'.format(name)]].plot(ax=ax)

plt.ylabel("Cumulative Energy (Wh)")
plt.savefig("../plots/tmy_combined.png")
