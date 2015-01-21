# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 17:45:48 2014

@author: sm1fg
"""

import numpy as np

import astropy.utils
import astropy.units as u

class FluxTubes(object):
    def __init__(self, xi=None, yi=None, si=None):
        
        self._xi = []
        self._yi = []
        self._si = []
        
        self.add_fluxtube(xi, yi, si)

    @property
    def xi(self):
        return u.Quantity(self._xi).reshape((len(self._xi),1))
    
    @property
    def yi(self):
        return u.Quantity(self._yi).reshape((len(self._xi),1))
    
    @property
    def si(self):
        return u.Quantity(self._si).reshape((len(self._xi),1))
    
    def add_fluxtube(self, xi, yi, si):
        if any((xi, yi, si)) and not all ((xi, yi, si)) and len(xi) == len(yi) == len(si):
            raise ValueError("xi, yi, si must all be sepcified and the same length")
        
        
        if all(map(astropy.utils.isiterable, (xi, yi, si))):
            self._xi.extend(list(xi))
            self._yi.extend(list(yi))
            self._si.extend(list(si))
        else:
            self._xi.append(xi)
            self._yi.append(yi)
            self._si.append(si)
    
    def __repr__(self):
        
        return "Xi: {}, Yi: {}, Si: {}".format(self.xi.__repr__(), self.yi.__repr__(), self.si.__repr__())

    def __str__(self):
        
        return self.__repr__()
    
    def __getitem__(self, val):
        if not isinstance(val, int):
            raise TypeError("You can only slice FluxTubes with an integer")
        
        return self.xi[val], self.yi[val], self.si[val]

    def __len__(self):
        return len(self._xi)
        

class BaseModel(dict):
    """
    This is a class to hold the parameters for a MHS Flux Tube model
    """

    model_dict = {'photo_scale': None,
                  'chrom_scale': None,
                  'corona_scale': None,
                  'coratio': None,
                  'phratio': None,
                  'pixel': None,
                  'radial_scale': None,
                  'B_corona': None,
                  'pBplus': None}
    
    options_dict = {'l_hdonly': False,  # set mag field zero to check background
                    'l_ambB': False,  # include some ambient magnetic field b_z
                    'l_const': False,  # axial Alfven speed const  Z-depend (Spruit)
                    'l_sqrt': False,  # axial Alfven speed sqrt   Z-depend (Spruit)
                    'l_linear': False,  # axial Alfven speed linear Z-depend (Spruit)
                    'l_square': False,  # axial Alfven speed square Z-depend (Spruit)
                    'l_B0_expz': False,  # Z-depend of Bz(r=0) exponentials
                    'l_B0_quadz': False,  # Z-depend of Bz(r=0) polynomials + exponential 
                    'l_single': False,  # only one flux tube
                    'l_atmos_val3c_mtw': False,  # interpolate composite VAL3c+MTW atmosphere
                    'suffix': '.gdf'
                    }

    flux_tubes = FluxTubes()

    def __init__(self, *args, **kwargs):
        
        super(dict, self).__init__(self, *args, **kwargs)
        self.update(self.model_dict)
        self.update(self.options_dict)
        
        self.__dict__ = self
        
        self._x_grid = None
        self._y_grid = None
        self._z_grid = None
    
    def __getitem__(self, value):
        """
        Make it so the properties can be got at using dict syntax.
        """
        try:
            return dict.__getitem__(self, value)
        except KeyError:
            return self.__getattribute__(value)
    
    def __str__(self):
        return self.__class__.__name__


    def _get_grid(self):
        if self._x_grid:
            pass
        else:
            cunit = self['zmax'].unit
            x, y, z = np.mgrid[self['xmin'].to(cunit):self['xmax'].to(cunit):1j*self['domain_dimensions'][0],
                               self['ymin'].to(cunit):self['ymax'].to(cunit):1j*self['domain_dimensions'][1],
                               self['zmin'].to(cunit):self['zmax'].to(cunit):1j*self['domain_dimensions'][2]]
            
            self._x_grid = u.Quantity(x, unit=cunit)
            self._y_grid = u.Quantity(y, unit=cunit)
            self._z_grid = u.Quantity(z, unit=cunit)

    @property
    def xmin(self):
        return self.domain_left_edge[0]
        
    @property
    def ymin(self):
        return self.domain_left_edge[1]
        
    @property
    def zmin(self):
        return self.domain_left_edge[2]
        
    @property
    def xmax(self):
        return self.domain_right_edge[0]
        
    @property
    def ymax(self):
        return self.domain_right_edge[1]
        
    @property
    def zmax(self):
        return self.domain_right_edge[2]
    
    @property
    def Z(self):
        return np.linspace(self.domain_left_edge[2], self.domain_right_edge[2], 
                           self.domain_dimensions[2])

    @property
    def Zext(self):
        return np.linspace(self.Z.min()-4.*self.dz, self.Z.max()+4.*self.dz, self.domain_dimensions[2]+8)

    @property
    def dx(self):
        return (self.domain_right_edge[0]-self.domain_left_edge[0])/(self.domain_dimensions[0]-1)
        
    @property
    def dy(self):
        return (self.domain_right_edge[1]-self.domain_left_edge[1])/(self.domain_dimensions[1]-1)
        
    @property
    def dz(self):
        return (self.domain_right_edge[2]-self.domain_left_edge[2])/(self.domain_dimensions[2]-1)

    @property
    def x(self):
        self._get_grid()
        return self._x_grid   
        
    @property
    def y(self):
        self._get_grid()
        return self._y_grid  
        
    @property
    def z(self):
        self._get_grid()
        return self._z_grid
    
    @property
    def l_mpi(self):
        try:
            from mpi4py import MPI
        except ImportError:
            return False
        
        try:
            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            
            if size !=1:
                return True
            else:
                return False
        finally:
            return False
        


class HMIModel(BaseModel):
    
    model_dict = {'photo_scale': 0.6*u.Mm,
                 'chrom_scale': 0.31*u.Mm,
                 'corona_scale': 100*u.Mm,      #scale height for the corona
                 'coratio': 0.06*u.one,
                 'phratio': 0.15*u.one,
                 'pixel': 0.36562475*u.Mm,      #(HMI pixel)
                 'radial_scale': 0.044*u.Mm,
                 'B_corona': 0.*u.T,
                 'pBplus': 4.250e-4*u.T,
                 'l_B0_quadz': True,
                 'l_single': True,
                 'l_hmi': True,
                 'l_atmos_val3c_mtw': True}
    model_dict['chratio'] = 1*u.one - model_dict['coratio'] - model_dict['phratio']


class MFEModel(BaseModel):
    
    model_dict = {'photo_scale': 0.6*u.Mm,
                 'chrom_scale': 0.31*u.Mm,
                 'corona_scale': 100*u.Mm,  #scale height for the corona
                 'coratio': 0.075*u.one,
                 'phratio': 0.0*u.one,
                 'pixel': 0.36562475*u.Mm,  #(HMI pixel)
                 'radial_scale': 0.044*u.Mm,
                 'B_corona': 0.*u.T,
                 'pBplus': 4.250e-4*u.T,
                 'domain_dimensions':[128,128,128],
                 'domain_left_edge':u.Quantity([-1*u.Mm,-1*u.Mm,35*u.km]),
                 'domain_right_edge':u.Quantity([1*u.Mm,1*u.Mm,1.6*u.Mm]),
                 'l_single': True,
                 'l_mfe': True,
                 'l_B0_quadz': True,
                 'l_atmos_val3c_mtw': True}
    model_dict['chratio'] = 1*u.one - model_dict['coratio'] - model_dict['phratio']
    
    flux_tubes = FluxTubes(0.*u.Mm,  0.*u.Mm,  100*u.mT)

spruit = {'photo_scale': 1.5*u.Mm,
          'chrom_scale': 1.5*u.Mm,
          'corona_scale': 100*u.Mm,      #scale height for the corona
          'coratio': 0.0*u.one,
          'model': 'spruit',
          'phratio': 0.0*u.one,
          'pixel': 0.1*u.Mm,              #(HMI pixel)
          'radial_scale': 0.025*u.Mm,
          'nftubes': 1,
          'B_corona': 0.*u.T,
          'pBplus': 4.250e-4*u.T}
spruit['chratio'] = 1*u.one - spruit['coratio'] - spruit['phratio']
spruit['Nxyz'] = [64,64,256] # 3D grid
spruit['xyz']  = [-0.64*u.Mm,0.64*u.Mm,-0.64*u.Mm,0.64*u.Mm,0*u.km,5.12*u.Mm] #grid size

paper1 = {'photo_scale': 0.6*u.Mm,
          'chrom_scale': 0.42*u.Mm,
          'corona_scale': 175*u.Mm,         #scale height for the corona
          'coratio': 0.0225*u.one,
          'model': 'paper1',
          'phratio': 0.0*u.one,
          'pixel': 0.36562475*u.Mm,              #(HMI pixel)
          'radial_scale': 0.044*u.Mm,
          'nftubes': 1,
          'B_corona': 2.00875e-4*u.T,
          'pBplus': 4.250e-4*u.T}
paper1['chratio'] = 1*u.one - paper1['coratio'] - paper1['phratio']
paper1['Nxyz'] = [128,128,432] # 3D grid
paper1['xyz']  = [-1.27*u.Mm,1.27*u.Mm,-1.27*u.Mm,1.27*u.Mm,0.*u.km,8.62*u.Mm] #grid size

paper2a = {'photo_scale': 0.6*u.Mm,
           'chrom_scale': 0.42*u.Mm,
           'corona_scale': 175*u.Mm,         #scale height for the corona
           'coratio': 0.0225*u.one,
           'model': 'paper2a',
           'phratio': 0.0*u.one,
           'pixel': 0.36562475*u.Mm,              #(HMI pixel)
           'radial_scale': 0.044*u.Mm,
           'nftubes': 4,
           'B_corona': 2.00875e-4*u.T,
           'pBplus': 4.250e-4*u.T}
paper2a['chratio'] = 1*u.one - paper2a['coratio'] - paper2a['phratio']
paper2a['Nxyz'] = [160,80,432] # 3D grid
paper2a['xyz']  = [-1.59*u.Mm,1.59*u.Mm,-0.79*u.Mm,0.79*u.Mm,0.*u.km,8.62*u.Mm] #grid size

paper2b = {'photo_scale': 0.6*u.Mm,
           'chrom_scale': 0.42*u.Mm,
           'corona_scale': 175*u.Mm,         #scale height for the corona
           'coratio': 0.0225*u.one,
           'model': 'paper2b',
           'phratio': 0.0*u.one,
           'pixel': 0.36562475*u.Mm,              #(HMI pixel)
           'radial_scale': 0.044*u.Mm,
           'nftubes': 4,
           'B_corona': 2.00875e-4*u.T,
           'pBplus': 4.250e-4*u.T}
paper2b['chratio'] = 1*u.one - paper2b['coratio'] - paper2b['phratio']
paper2b['Nxyz'] = [48,48,140] # 3D grid
paper2b['xyz']  = [-0.47*u.Mm,0.47*u.Mm,-0.47*u.Mm,0.47*u.Mm,0*u.km,2.78*u.Mm] #grid size

paper2c = {'photo_scale': 0.6*u.Mm,
           'chrom_scale': 0.42*u.Mm,
           'corona_scale': 175*u.Mm,         #scale height for the corona
           'coratio': 0.0225*u.one,
           'model': 'paper2c',
           'phratio': 0.0*u.one,
           'pixel': 0.36562475*u.Mm,              #(HMI pixel)
           'radial_scale': 0.044*u.Mm,
           'nftubes': 15,
           'B_corona': 2.00875e-4*u.T,
           'pBplus': 4.250e-4*u.T}
paper2c['chratio'] = 1*u.one - paper2c['coratio'] - paper2c['phratio']
paper2c['Nxyz'] = [224,224,140] # 3D grid
paper2c['xyz']  = [-2.23*u.Mm,2.23*u.Mm,-2.23*u.Mm,2.23*u.Mm,0*u.km,2.78*u.Mm] #grid size

paper2d = {'photo_scale': 0.6*u.Mm,
           'chrom_scale': 0.42*u.Mm,
           'corona_scale': 175*u.Mm,         #scale height for the corona
           'coratio': 0.0225*u.one,
           'model': 'paper2d',
           'phratio': 0.0*u.one,
           'pixel': 0.36562475*u.Mm,              #(HMI pixel)
           'radial_scale': 0.044*u.Mm,
           'nftubes': 18,
           'B_corona': 2.00875e-4*u.T,
           'pBplus': 4.250e-4*u.T}
paper2d['chratio'] = 1*u.one - paper2d['coratio'] - paper2d['phratio']
paper2d['Nxyz'] = [224,224,140] # 3D grid
paper2d['xyz']  = [-2.23*u.Mm,2.23*u.Mm,-0.79*u.Mm,0.79*u.Mm,0*u.km,2.78*u.Mm] #grid size

