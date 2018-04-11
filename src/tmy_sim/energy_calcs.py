#calculate energy returned as a function of data and tracker angle
import pvlib
import pandas as pd
from pvlib.pvsystem import PVSystem, retrieve_sam

sandia_modules = retrieve_sam('sandiamod')
sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')

def calculate_energy(surface_tilt, surface_azimuth, albedo, wspd, drybulb, current_index, solpos, dhi, dni, ghi, tracker_name, save_data = True, module = sandia_modules['Canadian_Solar_CS5P_220M___2009_'],inverter = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_'] ):
    dni_extra = pvlib.irradiance.extraradiation(current_index)
    dni_extra = pd.Series(dni_extra, index=current_index)

    # print(dni_extra)

    airmass = pvlib.atmosphere.relativeairmass(solpos['apparent_zenith'])

    # print(airmass)

    poa_sky_diffuse = pvlib.irradiance.haydavies(surface_tilt, surface_azimuth,
                                                 dhi, dni, dni_extra,
                                                 solpos['apparent_zenith'], solpos['azimuth'])
    # print(poa_sky_diffuse)

    poa_ground_diffuse = pvlib.irradiance.grounddiffuse(surface_tilt, ghi, albedo=albedo)


    aoi = pvlib.irradiance.aoi(surface_tilt, surface_azimuth, solpos['apparent_zenith'], solpos['azimuth'])

    poa_irrad = pvlib.irradiance.globalinplane(aoi, dni, poa_sky_diffuse, poa_ground_diffuse)

    pvtemps = pvlib.pvsystem.sapm_celltemp(poa_irrad['poa_global'], wspd, drybulb)


    rad_timestep = None
    #renaming series
    if save_data:
        dni_extra.rename("dni extra {}".format(tracker_name), inplace=True)
        poa_sky_diffuse.rename("sky diffuse {}".format(tracker_name), inplace=True)
        poa_ground_diffuse.rename("ground diffuse {}".format(tracker_name), inplace=True)
        poa_irrad.poa_direct.rename("poa direct {}".format(tracker_name), inplace=True)
        rad_timestep = pd.concat([dni_extra, poa_sky_diffuse, poa_ground_diffuse, poa_irrad.poa_direct], axis=1)



    effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(poa_irrad['poa_direct'], poa_irrad['poa_diffuse'], airmass, aoi, module)

    sapm_out = pvlib.pvsystem.sapm(effective_irradiance, pvtemps.temp_cell, module)

    ac =  pvlib.pvsystem.snlinverter(sapm_out['v_mp'], sapm_out['p_mp'], inverter)

    return sapm_out, ac, rad_timestep, pvtemps

def energy_motion(start, end, cap, energy_per_deg_per_mw = 0.01):
    '''
    energy_per_deg_per_mw = energy consumed in Kwh per mw per degree when moving
    source: ATI DuraTrack HZ3 Spec sheet
    '''
    angle_delta = abs(end - start)

    return cap*angle_delta*energy_per_deg_per_mw
