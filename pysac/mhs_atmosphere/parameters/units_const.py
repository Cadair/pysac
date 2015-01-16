# -*- coding: utf-8 -*-

import astropy.constants as asc
import astropy.units as u

__all__ = ['scales', 'physical_constants']

scales   = {'length':         1*u.Mm,
            'density':        1e-6*u.kg/u.m**3,
            'velocity':       1e3*u.m/u.s,
            'temperature':    1.0*u.K, 
            'magnetic':       1e-3*u.mT
            }

scales['energy density'] = scales['density'] * scales['velocity']**2
scales['time'] = scales['length'] / scales['velocity'] 
scales['mass'] = scales['density'] * scales['length']**3 
#D momentum/Dt force density balance 
scales['force density'] = scales['density'] * scales['velocity'] / scales['time'] 
                   
physical_constants = {'gamma':       5.0/3.0         , 
                      'mu':          0.602           , 
                      'mu0':         asc.mu0         , 
                      'boltzmann':   asc.k_B         ,
                      'proton_mass': asc.m_p         ,
                      'gravity':     -274.0*u.km/u.s/u.s
                     }
