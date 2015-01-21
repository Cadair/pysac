# -*- coding: utf-8 -*-

import os
import numpy as np

import astropy.units as u

import pysac.mhs_atmosphere as atm
from pysac.mhs_atmosphere.parameters import scales, physical_constants
from pysac.mhs_atmosphere.parameters import models
    
#==============================================================================
# Define the Model
#==============================================================================
model = models.MFEModel()

#interpolate the hs 1D profiles from empirical data source[s]
empirical_data = atm.read_VAL3c_MTW(mu=physical_constants['mu'])

background_table = atm.interpolate_atmosphere(model, empirical_data)

#==============================================================================
#calculate 1d hydrostatic balance from empirical density profile
#==============================================================================
# the hs pressure balance is enhanced by pressure equivalent to the
# residual mean coronal magnetic pressure contribution once the magnetic
# field has been applied
magp_meanz = np.ones(len(model['Z'])) * u.one
magp_meanz *= model['pBplus']**2/(2*physical_constants['mu0'])

pressure_Z, rho_Z, Rgas_Z = atm.vertical_profile(model, background_table,
                                                 magp_meanz, physical_constants)

(Bx, By, Bz, pressure_m, rho_m,
 Fx, Fy, Btensx, Btensy) = atm.calculate_magnetic_field(model, physical_constants, scales)

#==============================================================================
# Construct 3D hs arrays and then add the mhs adjustments to obtain atmosphere
#==============================================================================
# select the 1D array spanning the local mpi process; the add/sub of dz to
# ensure all indices are used, but only once
indz = np.where(model['Z'] >= (model.z.min()-0.1*model['dz']).to(model['Z'].unit)) and \
       np.where(model['Z'] <= (model.z.max()+0.1*model['dz']).to(model['Z'].unit))
pressure_z, rho_z, Rgas_z = pressure_Z[indz], rho_Z[indz], Rgas_Z[indz]
# local proc 3D mhs arrays
pressure, rho = atm.mhs_3D_profile(model.z,
                                   pressure_z,
                                   rho_z,
                                   pressure_m,
                                   rho_m
                                  )
magp = (Bx**2 + By**2 + Bz**2)/(2.*physical_constants['mu0'])
print'max B corona = ',magp[:,:,-1].max().decompose()

energy = atm.get_internal_energy(pressure, magp, physical_constants)

#============================================================================
# Save data for SAC and plotting
#============================================================================
# set up data directory and file names
# may be worthwhile locating on /data if files are large
datadir = os.path.expanduser('~/mhs_atmosphere/'+ str(model) +'/')
filename = datadir + str(model) + model['suffix']
if not os.path.exists(datadir):
    os.makedirs(datadir)
sourcefile = datadir + str(model) + '_sources' + model['suffix']
auxfile = datadir + str(model) + '_aux' + model['suffix']

# save the variables for the initialisation of a SAC simulation
atm.save_SACvariables(model, filename, rho, Bx, By, Bz, energy, physical_constants)
# save the balancing forces as the background source terms for SAC simulation
atm.save_SACsources(model, sourcefile, Fx, Fy, physical_constants)
# save auxilliary variable and 1D profiles for plotting and analysis
Rgas = np.zeros(model.x.shape)
Rgas[:] = Rgas_z
temperature = pressure/rho/Rgas

if not model['l_hdonly']:
    inan = np.where(magp <=1e-7*pressure.min())
    magpbeta = magp
    magpbeta[inan] = 1e-7*pressure.min()  # low pressure floor to avoid NaN
    pbeta  = pressure/magpbeta
else:
    pbeta  = magp+1.0    #dummy to avoid NaN
    
alfven = np.sqrt(2.*physical_constants['mu0']*magp/rho)
cspeed = np.sqrt(physical_constants['gamma']*pressure/rho)
atm.save_auxilliary1D(model, auxfile, pressure_m, rho_m, temperature, pbeta,
                      alfven, cspeed, Btensx, Btensy, pressure_Z, rho_Z, 
                      Rgas_Z, physical_constants)
