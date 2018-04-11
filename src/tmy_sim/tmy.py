#testing TMY data parsing + simulation

import pvlib
import pandas as pd
from pvlib.pvsystem import PVSystem, retrieve_sam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from random import randint
import numpy as np
import simple_rl as rl
from simple_rl.mdp.oomdp.OOMDPObjectClass import OOMDPObject
from simple_rl.mdp.oomdp.OOMDPStateClass import OOMDPState
from tqdm import trange
import time
from trackers import *
loc = "/Users/edwardwilliams/Documents/research/heliotrope/simulations/data/722745TYA.CSV"

def run_sim_on_tracker(tracker, tmy_data, sand_point, n_epochs=10, n_steps=500, energy_per_deg_per_mw = 0.01):
    '''
    Returns power, angle history
    energy_per_deg_per_mw = energy consumed in Kwh per mw per degree when moving
    source: ATI DuraTrack HZ3 Spec sheet
    '''
    print("running {}".format(tracker.name))

    #TODO: save previous state/reward
    for e in trange(n_epochs):
        #returning results from most recent epoch
        #TODO: avoid successive concats
        total = pd.DataFrame()
        angles = np.zeros((n_steps,))
        energy_consumed_move = np.zeros((n_steps,))
        temps = pd.DataFrame()
        radiation_rows = []
        surface_azimuth = tracker.get_azimuth()
        old_tilt = 0
        prev_reward = 0
        for i in trange(n_steps):
            current_step_data = tmy_data[tmy_data.index[i]:tmy_data.index[i]]

            solpos = pvlib.solarposition.get_solarposition(current_step_data.index, sand_point.latitude, sand_point.longitude)


            #time since epoch
            last_unix = time.mktime(tmy_data.index[i].timetuple())
            #time of day
            hour = tmy_data.index[i].hour

            sun_attributes = {'apparent_zenith': float(solpos['apparent_zenith']), 'azimuth': float(solpos['azimuth'])}
            env_attributes = {'GHI': float(current_step_data['GHI']), 'DNI':float(current_step_data['DNI']), 'DryBulb': float(current_step_data['DryBulb']), 'TotCld':float(current_step_data['TotCld']), 'OpqCld':float(current_step_data['OpqCld']), 'epoch':last_unix, 'hour':hour}

            angle_pos = pvlib.tracking.singleaxis(solpos['apparent_zenith'], solpos['azimuth'], backtrack=False)

            surface_tilt = angle_pos['tracker_theta']
            # print(surface_tilt)
            if np.isnan(surface_tilt[0]):
                #TODO: fix this wrt hour
                surface_tilt = tracker.fallback_angle if hour > 12 else -tracker.fallback_angle

            tracker_attributes = {'tracker_theta':surface_tilt, 'prev_angle':old_tilt}

            sun_obj = OOMDPObject(sun_attributes, name="sun")
            env_obj = OOMDPObject(env_attributes, name="env")
            tracker_obj = OOMDPObject(tracker_attributes, name="tracker")

            #passing list of objects for class!
            objects = {'sun':[sun_obj], 'env':[env_obj], 'tracker':[tracker_obj]}
            state = OOMDPState(objects)

            surface_tilt = float(tracker.get_angle(state, prev_reward))
            angles[i] = surface_tilt

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

            # radiation_timestep = pd.concat([dni_extra,poa_sky_diffuse,poa_ground_diffuse,poa_irrad], axis=1,verify_integrity=True)
            # radiation_timestep = pd.DataFrame({"dni extra {}".format(tracker.name):[dni_extra, "sky diffuse {}".format(tracker.name):poa_sky_diffuse, "ground diffuse {}".format(tracker.name):poa_ground_diffuse, "POA direct {}".format(tracker.name):poa_irrad['poa_direct']})

            #renaming series
            dni_extra.rename("dni extra {}".format(tracker.name), inplace=True)
            poa_sky_diffuse.rename("sky diffuse {}".format(tracker.name), inplace=True)
            poa_ground_diffuse.rename("ground diffuse {}".format(tracker.name), inplace=True)
            poa_irrad.poa_direct.rename("poa direct {}".format(tracker.name), inplace=True)
            rad_timestep = pd.concat([dni_extra, poa_sky_diffuse, poa_ground_diffuse, poa_irrad.poa_direct], axis=1)

            radiation_rows.append(rad_timestep)

            sandia_modules = retrieve_sam('sandiamod')
            cec_inverters = retrieve_sam('cecinverter')
            module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']

            cap = float(module['Isco']*module['Voco']/(10**6)) #convert to MW

            angle_delta = abs(old_tilt - surface_tilt)

            eng_consumed_move = cap*angle_delta*energy_per_deg_per_mw
            energy_consumed_move[i] = eng_consumed_move
            old_tilt = surface_tilt

            inverter = cec_inverters['SMA_America__SC630CP_US_315V__CEC_2012_']

            effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(poa_irrad.poa_direct, poa_irrad.poa_diffuse, airmass, aoi, module)

            sapm_out = pvlib.pvsystem.sapm(effective_irradiance, pvtemps.temp_cell, module)

            prev_reward = float(sapm_out['p_mp']) - eng_consumed_move*1000

            total = total.append(sapm_out)
            temps = temps.append(pvtemps)
            total['p cumulative {}'.format(tracker.name)] = total.p_mp.cumsum() - eng_consumed_move*1000 #kwh to wh


    #convert angles to df
    angles_series = pd.Series(angles, index=tmy_data.index[0:n_hours])
    energy_consumed = pd.Series(energy_consumed_move, index=tmy_data.index[0:n_hours])

    radiation = pd.concat(radiation_rows, axis=0)
    #rename
    temps.rename(columns={'temp_cell': 'cell temp {}'.format(tracker.name)}, inplace=True)

    angles_df = pd.DataFrame(angles_series, columns=['angle {}'.format(tracker.name)])
    energy_consumed_df = pd.DataFrame(energy_consumed, columns=['energy consumed {}'.format(tracker.name)])
    return total, angles_df, temps, energy_consumed_df, radiation

def generate_plots(results):

    fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(50,50))


    for name, res in results.items():
        res[0][['p cumulative {}'.format(name)]].plot(ax=ax)
        res[1][['angle {}'.format(name)]].plot(ax=ax2)
        res[2][['cell temp {}'.format(name)]].plot(ax=ax3)
        res[3][['energy consumed {}'.format(name)]].plot(ax=ax4)

    ax.set_ylabel("Cumulative Energy (Wh)")
    ax2.set_ylabel("angle (deg)")
    ax3.set_ylabel("cell temp (Deg C)")
    ax4.set_ylabel("energy consumed (kwh)")
    plt.savefig("../plots/tmy_results.png")
    plt.close()

    f, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2, figsize=(50,50))


    #plotting radiation breakdown
    for name, res in results.items():
        res[4][["dni extra {}".format(name)]].plot(ax=ax5)
        res[4][["sky diffuse {}".format(name)]].plot(ax=ax6)
        res[4][["ground diffuse {}".format(name)]].plot(ax=ax7)
        res[4][["poa direct {}".format(name)]].plot(ax=ax8)

    ax5.set_ylabel("Irradiance (W)")
    ax6.set_ylabel("Irradiance (W)")
    ax7.set_ylabel("Irradiance (W)")
    ax8.set_ylabel("Irradiance (W)")
    plt.savefig("../plots/tmy_rad_breakdown.png")

def run():
    #TODO: test with different fixed trackers
    fixed = FixedPolicyTracker(30, 90) #azimuth angle
    rand = RandomTracker(10, 80, 90)
    astro = AstroTracker(90)
    ucb = LinUCBTracker(90, context_size=11)
    sarsa = SARSATracker(90, 11)

    trackers = [astro, ucb, sarsa]

    tmy_data, meta = pvlib.tmy.readtmy3(filename=loc)

    #TODO: change this
    albedo = 0.2 #assume fixed, changing in reality

    # create pvlib Location object based on meta data
    sand_point = pvlib.location.Location(meta['latitude'], meta['longitude'], tz='US/Arizona',
                                         altitude=meta['altitude'], name=meta['Name'].replace('"',''))

    results = {tracker.name:run_sim_on_tracker(tracker, tmy_data, sand_point) for tracker in trackers}

    generate_plots(results)

if __name__=="__main__":
    run()