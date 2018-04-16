#For every timestep in a TMY data file, determining the difference between
#the energy values in the astronomical and optimal angles.

from argparse import ArgumentParser
from multiprocessing import Pool
from sys import argv
import pvlib
from tqdm import trange
from energy_calcs import *
import numpy as np
import pandas as pd

sandia_modules = retrieve_sam('sandiamod')
module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']

def parse_args(args):
    parser = ArgumentParser()
    parser.add_argument("--tmy_file", help="location of tmy file")
    parser.add_argument("--albedo", help="albedo", type=float)
    parser.add_argument("--procs", help="number of processes", type=int)
    parser.add_argument("--n_steps", help="number of steps to run")
    parser.add_argument("--output", help="location of output dataframe")
    return parser.parse_args(args)


def calc_energy_parallel(args):
    return float(calculate_energy(*args)[0])

def calc_irrad_parallel(args):
    '''
    runs irradiance calculation.
    '''
    return calculate_irradiance(*args)

def opt_energy(pool, tmy_step_data, solpos, albedo, limits=(-52, 52), azimuth=90, step=1):
    '''
    Computes the optimal angle from the TMY step data
    '''

    configurations = np.arange(limits[0], limits[1] + step, step)

    args_list = []
    for config in configurations:
        args_list.append((config, azimuth, albedo, tmy_step_data['Wspd'], tmy_step_data['DryBulb'], tmy_step_data.index, solpos, tmy_step_data['DHI'], tmy_step_data['DNI'], tmy_step_data['GHI']))


    # results = pool.map(calc_energy_parallel, args_list)
    results = map(calc_irrad_parallel, args_list)

    #runs SAPM model

    effective_irradiance, temp_cell = zip(*results)

    sapm_results = pvlib.pvsystem.sapm(np.array(effective_irradiance), np.array(temp_cell), module)

    pwr = sapm_results['p_mp']

    opt_energy = max(pwr)
    opt_angle = configurations[np.argmax(pwr)]

    return opt_angle, opt_energy

def astro_energy(tmy_step_data, sand_point, solpos, albedo, azimuth=90, limits=(-52, 52), fallback_angle = 0):
    '''
    Gets the energy produced by an astronomical tracker at the provided TMY step.
    '''

    hour = tmy_step_data.index.hour

    angle_pos = pvlib.tracking.singleaxis(solpos['apparent_zenith'], solpos['azimuth'], backtrack=False)

    surface_tilt = float(angle_pos['tracker_theta'])
    # print(surface_tilt)
    if np.isnan(surface_tilt):
        surface_tilt = fallback_angle if hour > 12 else -fallback_angle


    #handling tracker limits! movement!
    if surface_tilt < limits[0]:
        surface_tilt = limits[0]
    elif surface_tilt > limits[1]:
        surface_tilt = limits[1]

    #simulating inaccuracy of controller
    #this reflects ATI tracker - other controllers claim to be higher precision
    # noise = np.random.normal(loc=0, scale=1.5)

    astro_angle = int(surface_tilt)

    #TODO: move this to irradiance calculation
    astro_energy = float(calculate_energy(astro_angle, azimuth, albedo, tmy_step_data['Wspd'], tmy_step_data['DryBulb'], tmy_step_data.index, solpos, tmy_step_data['DHI'], tmy_step_data['DNI'], tmy_step_data['GHI'], "", False)[0])

    return astro_angle, astro_energy

def generate_energy_records(tmy_data, sand_point, albedo, n_steps, n_procs=3):
    '''
    Generate the optimal energy and angle records for a TMY data set.
    '''

    p = Pool(n_procs)

    opt_engs = np.zeros((n_steps,))
    opt_angles = np.zeros((n_steps,))
    astro_angles = np.zeros((n_steps,))
    astro_engs = np.zeros((n_steps,))
    for i in trange(n_steps):
        current_step_data = tmy_data[tmy_data.index[i]:tmy_data.index[i]]

        solpos = pvlib.solarposition.get_solarposition(current_step_data.index, sand_point.latitude, sand_point.longitude)

        opt_angle, opt_eng = opt_energy(p, current_step_data, solpos, albedo)

        astro_angle, astro_eng = astro_energy(current_step_data, sand_point, solpos, albedo)

        opt_engs[i] = opt_eng
        opt_angles[i] = opt_angle
        astro_angles[i] = astro_angle
        astro_engs[i] = astro_eng

    #convert to DataFrame

    optimal_df = pd.DataFrame(data={"opt_eng":opt_engs, "opt_angle":opt_angles, "astro_angle":astro_angles, "astro_eng":astro_engs}, index=tmy_data.index[0:n_steps])

    return optimal_df


def run(args):

    print("loading data from {}".format(args.tmy_file))
    tmy_data, meta = pvlib.tmy.readtmy3(filename=args.tmy_file)

    sand_point = pvlib.location.Location(meta['latitude'], meta['longitude'], tz=meta['TZ'], altitude=meta['altitude'], name=meta['Name'].replace('"',''))

    if args.n_steps == "max":
        n_steps = len(tmy_data.index)
    else:
        n_steps = int(args.n_steps)

    # n_steps = 100

    # results = generate_energy_records(tmy_data, sand_point, 0.3, n_steps, 1)

    results = generate_energy_records(tmy_data, sand_point, args.albedo, n_steps, args.procs)

    print ("saving to {}".format(args.output))
    results.to_pickle(args.output)

if __name__=="__main__":
    run(parse_args(argv[1:]))
