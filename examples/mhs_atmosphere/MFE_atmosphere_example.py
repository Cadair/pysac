# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 13:55:17 2014

@author: sm1fg

This is the main module to construct a magnetohydrostatic solar atmosphere,
given a specified magnetic network of self-similar magnetic flux tubes and
save the output to gdf format.

To select an existing configuration change the import as model_pars, set Nxyz,
xyz_SI and any other special parameters, then execute mhs_atmopshere.

To add new configurations:
add the model options to set_options in parameters/options.py;
add options required in parameters/model_pars.py;
add alternative empirical data sets to hs_model/;
add alternativ table than interploate_atmosphere in hs_model/hs_atmosphere.py;
add option to get_flux_tubes in mhs_model/flux_tubes.py

If an alternative formulation of the flux tube is required add options to
construct_magnetic_field and construct_pairwise_field in
mhs_model/flux_tubes.py

Plotting options are included in plot/mhs_plot.py
"""

import os
import numpy as np

import astropy.units as u

import pysac.mhs_atmosphere as atm
from pysac.mhs_atmosphere.parameters import scales, physical_constants
from pysac.mhs_atmosphere.parameters import models
#==============================================================================
#check whether mpi is required and the number of procs = size
#==============================================================================
try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    l_mpi = True
    l_mpi = l_mpi and (size != 1)
except ImportError:
    l_mpi = False
    rank = 0
    size = 1
    
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

pressure_Z, rho_Z, Rgas_Z = atm.vertical_profile(model,
                                                 background_table,
                                                 magp_meanz,
                                                 physical_constants)

#==============================================================================
# load flux tube footpoint parameters
#==============================================================================
# axial location and value of Bz at each footpoint
xi, yi, Si = atm.get_flux_tubes(model)

#==============================================================================
# split domain into processes if mpi
#==============================================================================
cunit = model['zmax'].unit
x, y, z = np.mgrid[model['xmin'].to(cunit):model['xmax'].to(cunit):1j*model['domain_dimensions'][0],
                   model['ymin'].to(cunit):model['ymax'].to(cunit):1j*model['domain_dimensions'][1],
                   model['zmin'].to(cunit):model['zmax'].to(cunit):1j*model['domain_dimensions'][2]]

x = u.Quantity(x, unit=cunit)
y = u.Quantity(y, unit=cunit)
z = u.Quantity(z, unit=cunit)
#==============================================================================
# initialize zero arrays in which to add magnetic field and mhs adjustments
#==============================================================================
Bx   = u.Quantity(np.zeros(x.shape), unit=u.T)  # magnetic x-component
By   = u.Quantity(np.zeros(x.shape), unit=u.T)  # magnetic y-component
Bz   = u.Quantity(np.zeros(x.shape), unit=u.T)  # magnetic z-component
pressure_m = u.Quantity(np.zeros(x.shape), unit=u.Pa) # magneto-hydrostatic adjustment to pressure
rho_m = u.Quantity(np.zeros(x.shape), unit=u.kg/u.m**3)      # magneto-hydrostatic adjustment to density
# initialize zero arrays in which to add balancing forces and magnetic tension
Fx   = u.Quantity(np.zeros(x.shape), unit=u.N/u.m**3)  # balancing force x-component
Fy   = u.Quantity(np.zeros(x.shape), unit=u.N/u.m**3)  # balancing force y-component
# total tension force for comparison with residual balancing force
Btensx  = u.Quantity(np.zeros(x.shape), unit=u.N/u.m**3)
Btensy  = u.Quantity(np.zeros(x.shape), unit=u.N/u.m**3)
#==============================================================================
#calculate the magnetic field and pressure/density balancing expressions
#==============================================================================
for i in range(0,model['nftubes']):
    for j in range(i,model['nftubes']):
        if rank == 0:
            print'calculating ij-pair:',i,j
        if i == j:
            pressure_mi, rho_mi, Bxi, Byi ,Bzi, B2x, B2y =\
                atm.construct_magnetic_field(
                                             x, y, z,
                                             xi[i], yi[i], Si[i],
                                             model,
                                             physical_constants,
                                             scales
                                            )
            Bx, By, Bz = Bxi+Bx, Byi+By ,Bzi+Bz
            Btensx += B2x
            Btensy += B2y
            pressure_m += pressure_mi
            rho_m += rho_mi
        else:
            pressure_mi, rho_mi, Fxi, Fyi, B2x, B2y =\
                atm.construct_pairwise_field(
                                             x, y, z,
                                             xi[i], yi[i],
                                             xi[j], yi[j], Si[i], Si[j],
                                             model,
                                             physical_constants,
                                             scales
                                            )
            pressure_m += pressure_mi
            rho_m += rho_mi
            Fx   += Fxi
            Fy   += Fyi
            Btensx += B2x
            Btensy += B2y

# clear some memory
del pressure_mi, rho_mi, Bxi, Byi ,Bzi, B2x, B2y
#==============================================================================
# Construct 3D hs arrays and then add the mhs adjustments to obtain atmosphere
#==============================================================================
# select the 1D array spanning the local mpi process; the add/sub of dz to
# ensure all indices are used, but only once
indz = np.where(model['Z'] >= (z.min()-0.1*model['dz']).to(model['Z'].unit)) and \
       np.where(model['Z'] <= (z.max()+0.1*model['dz']).to(model['Z'].unit))
pressure_z, rho_z, Rgas_z = pressure_Z[indz], rho_Z[indz], Rgas_Z[indz]
# local proc 3D mhs arrays
pressure, rho = atm.mhs_3D_profile(z,
                                   pressure_z,
                                   rho_z,
                                   pressure_m,
                                   rho_m
                                  )
magp = (Bx**2 + By**2 + Bz**2)/(2.*physical_constants['mu0'])
if rank ==0:
    print'max B corona = ',magp[:,:,-1].max().decompose()
energy = atm.get_internal_energy(pressure,
                                                  magp,
                                                  physical_constants)
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
atm.save_SACvariables(
              model,
              filename,
              rho,
              Bx,
              By,
              Bz,
              energy,
              physical_constants)
# save the balancing forces as the background source terms for SAC simulation
atm.save_SACsources(model,
                    sourcefile,
                    Fx,
                    Fy,
                    physical_constants)
# save auxilliary variable and 1D profiles for plotting and analysis
Rgas = np.zeros(x.shape)
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
atm.save_auxilliary1D(
              model,
              auxfile,
              pressure_m,
              rho_m,
              temperature,
              pbeta,
              alfven,
              cspeed,
              Btensx,
              Btensy,
              pressure_Z,
              rho_Z,
              Rgas_Z,
              physical_constants
             )

