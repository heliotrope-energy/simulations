#testing TMY data parsing + simulation

import pvlib
import pandas as pd
from pvlib.pvsystem import PVSystem, retrieve_sam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import simple_rl as rl
from simple_rl.mdp.oomdp.OOMDPObjectClass import OOMDPObject
from simple_rl.mdp.oomdp.OOMDPStateClass import OOMDPState
from tqdm import trange
import time
from trackers import *
from energy_calcs import calculate_energy, energy_motion
import os

def tmy_step_to_OOMDP(current_step_data, tracker, solpos, old_tilt, albedo):
    '''
    Creates OOMDP state from TMY data.
    '''

    #time since epoch in sec
    last_unix = current_step_data.index.view('int64')
    #time of day
    hour = current_step_data.index.hour

    sun_attributes = {'apparent_zenith': float(solpos['apparent_zenith']), 'azimuth': float(solpos['azimuth'])}
    env_attributes = {'DHI': float(current_step_data['DHI']), 'GHI': float(current_step_data['GHI']), 'DNI':float(current_step_data['DNI']), 'Wspd':float(current_step_data['Wspd']), 'DryBulb': float(current_step_data['DryBulb']), 'TotCld':float(current_step_data['TotCld']), 'OpqCld':float(current_step_data['OpqCld']),
                        'albedo':albedo,  'datetime':last_unix, 'hour':hour}

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
    return OOMDPState(objects)

def run_sim_on_tracker(tracker, tmy_data, sand_point, albedo,  n_epochs=10, n_steps=500,):
    '''
    Returns power, angle history

    '''
    # print("running {} \n".format(tracker.name))
    sandia_modules = retrieve_sam('sandiamod')
    cec_inverters = retrieve_sam('cecinverter')
    module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
    cap = float(module['Isco']*module['Voco']/(10**6)) #convert to MW
    #TODO: save previous state/reward
    for e in range(n_epochs):
        #returning results from most recent epoch
        #TODO: avoid successive concats
        total = pd.DataFrame()
        angles = np.zeros((n_steps,))
        energy_consumed_move = np.zeros((n_steps,))
        temps = pd.DataFrame()
        radiation_rows = []
        ac_all = np.zeros((n_steps,))
        surface_azimuth = tracker.get_azimuth()
        old_tilt = 0
        prev_reward = 0
        for i in range(n_steps):
            current_step_data = tmy_data[tmy_data.index[i]:tmy_data.index[i]]

            solpos = pvlib.solarposition.get_solarposition(current_step_data.index, sand_point.latitude, sand_point.longitude)


            state = tmy_step_to_OOMDP(current_step_data, tracker, solpos, old_tilt, albedo)

            surface_tilt = float(tracker.get_angle(state, prev_reward))
            angles[i] = surface_tilt

            sapm_out, ac, rad_timestep, pvtemps = calculate_energy(surface_tilt, surface_azimuth, albedo, current_step_data['Wspd'], current_step_data['DryBulb'], current_step_data.index, solpos,  current_step_data['DHI'],  current_step_data['DNI'],  current_step_data['GHI'], tracker.name)

            radiation_rows.append(rad_timestep)

            eng_consumed_move = energy_motion(old_tilt, surface_tilt, cap)
            old_tilt = surface_tilt
            prev_reward = float(ac) - eng_consumed_move*1000
            energy_consumed_move[i] = eng_consumed_move

            ac_all[i] = ac - eng_consumed_move*1000 #kwh to wh
            temps = temps.append(pvtemps)


    #convert angles to df
    angles_series = pd.Series(angles, index=tmy_data.index[0:n_steps])
    energy_consumed = pd.Series(energy_consumed_move, index=tmy_data.index[0:n_steps])

    radiation = pd.concat(radiation_rows, axis=0)
    #rename
    temps.rename(columns={'temp_cell': 'cell temp {}'.format(tracker.name)}, inplace=True)

    angles_df = pd.DataFrame(angles_series, columns=['angle {}'.format(tracker.name)])
    energy_consumed_df = pd.DataFrame(energy_consumed, columns=['energy consumed {}'.format(tracker.name)])

    ac_total = pd.Series(ac_all, index=tmy_data.index[0:n_steps])
    sum = ac_total.sum()

    ac_df = pd.DataFrame(ac_total, columns=['ac_step'])
    ac_df['p cumulative {}'.format(tracker.name)] = ac_df.cumsum()


    return ac_df, angles_df, temps, energy_consumed_df, radiation, sum

def generate_plots(results, albedo, output_loc, tmy_id, steps, tmy_loc_name):

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
    plt.savefig("{}/{}_{}_tmy_results.png".format(output_loc, tmy_id, albedo))
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
    plt.savefig("{}/{}_{}_tmy_rad_breakdown.png".format(output_loc, tmy_id, albedo))


    #printing results values
    outstrings = [tmy_id, str(albedo), str(steps), tmy_loc_name]
    for name, res in results.items():
        outstrings.append("{} produced by {}".format(res[5], name))

    with open("{}/summary_{}_{}.txt".format(output_loc, tmy_id, albedo), 'w') as f:
        f.write("\n".join(outstrings))

def run(loc, albedo, output_loc, name, steps=1000):

    #TODO: test with different fixed trackers
    fixed = FixedPolicyTracker(30, 90) #azimuth angle
    rand = RandomTracker(-70, 90, 90)
    astro = AstroTracker(90)
    ucb = LinUCBTracker(90, context_size=13)
    sarsa = SARSATracker(90, 13)
    optimal = OptimalTracker(90)

    trackers = [astro, optimal]

    tmy_data, meta = pvlib.tmy.readtmy3(filename=loc)

    # create pvlib Location object based on meta data
    #TODO: add this to logs
    sand_point = pvlib.location.Location(meta['latitude'], meta['longitude'], tz='US/Arizona',
                                         altitude=meta['altitude'], name=meta['Name'].replace('"',''))

    if steps=="max":
        steps = len(tmy_data.index)

    results = {tracker.name:run_sim_on_tracker(tracker, tmy_data, sand_point, albedo, n_epochs=1, n_steps=steps) for tracker in trackers}

    generate_plots(results, albedo, output_loc, name, steps, meta['Name'])

if __name__=="__main__":
    loc = "/Users/edwardwilliams/Documents/research/heliotrope/simulations/data/722745TYA.CSV"
    run(loc, 0.2, "../../plots/all_tmy", '722745TYA', steps=100)
