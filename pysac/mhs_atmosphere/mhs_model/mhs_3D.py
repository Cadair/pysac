# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 11:37:39 2014

@author: sm1fg

    Construct the magnetic network and generate the adjustments to the
    non-magnetic atmosphere for mhs equilibrium.

"""

import numpy as np
import astropy.units as u

from .flux_tubes import construct_magnetic_field, construct_pairwise_field

#==============================================================================
# The inner computational loop
#==============================================================================

def calculate_magnetic_field(model, physical_constants, scales):
    """
    Calculate the combined magnetic field and interaction forces of all flux tubes.
    
    Parameters
    ----------
    
    model : `~pysac.mhs_atmosphere.parameters.model.BaseModel`
        The model to construct
    
    physical_constants : dict
        Physical constants to use
    
    scales : dict
        characteristic scales
    
    Returns
    -------
    Bx, By, Bz, pressure_m, rho_m, Fx, Fy, Btensx, Btensy : `~astropy.units.Quantity`
        Resulting arrays
    """
    
    # initialize zero arrays in which to add magnetic field and mhs adjustments
    Bx   = u.Quantity(np.zeros(model.x.shape), unit=u.T)  # magnetic x-component
    By   = u.Quantity(np.zeros(model.x.shape), unit=u.T)  # magnetic y-component
    Bz   = u.Quantity(np.zeros(model.x.shape), unit=u.T)  # magnetic z-component
    pressure_m = u.Quantity(np.zeros(model.x.shape), unit=u.Pa) # magneto-hydrostatic adjustment to pressure
    rho_m = u.Quantity(np.zeros(model.x.shape), unit=u.kg/u.m**3)      # magneto-hydrostatic adjustment to density
    # initialize zero arrays in which to add balancing forces and magnetic tension
    Fx   = u.Quantity(np.zeros(model.x.shape), unit=u.N/u.m**3)  # balancing force x-component
    Fy   = u.Quantity(np.zeros(model.x.shape), unit=u.N/u.m**3)  # balancing force y-component
    # total tension force for comparison with residual balancing force
    Btensx  = u.Quantity(np.zeros(model.x.shape), unit=u.N/u.m**3)
    Btensy  = u.Quantity(np.zeros(model.x.shape), unit=u.N/u.m**3)
    
    
    #calculate the magnetic field and pressure/density balancing expressions
    for i in range(0,len(model.flux_tubes)):
        for j in range(i,len(model.flux_tubes)):
            print'calculating ij-pair:',i,j
            if i == j:
                pressure_mi, rho_mi, Bxi, Byi ,Bzi, B2x, B2y =\
                    construct_magnetic_field(model,
                                             i,
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
                    construct_pairwise_field(model,
                                             i,
                                             j,
                                             physical_constants,
                                             scales
                                            )
                pressure_m += pressure_mi
                rho_m += rho_mi
                Fx   += Fxi
                Fy   += Fyi
                Btensx += B2x
                Btensy += B2y
    
    return Bx, By, Bz, pressure_m, rho_m, Fx, Fy, Btensx, Btensy

#============================================================================
# Derive the hydrostatic profiles and include the magneto adjsutments
#============================================================================
def mhs_3D_profile(z,
                   pressure_z,
                   rho_z,
                   pressure_m,
                   rho_m
                  ):
    """Return the vertical profiles for thermal pressure and density in 3D"""
    #Make things 3D
    rho_0 = np.empty(z.shape) * u.g /u.m**3
    rho_0[:] = rho_z
    #hydrostatic vertical profile
    pressure_0 = np.empty(z.shape) * u.Pa
    pressure_0[:] = pressure_z
    #magnetohydrostatic adjusted full 3D profiles
    pressure =  pressure_m + pressure_0
    rho = rho_m+rho_0

    return pressure, rho

#============================================================================
# Calculate internal energy
#============================================================================
def get_internal_energy(pressure, magp, physical_constants):
    """ Convert pressures to internal energy -- this may need revision if an
    alternative equation of state is adopted.
    """
    energy = pressure/(physical_constants['gamma']-1) + magp
    return energy

