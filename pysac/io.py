# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 12:53:52 2011

@author: Stuart Mumford

FortranFile stolen from: Neil Martinsen-Burrell [via Enthought Mailing list]
"""

import struct

class FortranFile(file):
    """File with methods for dealing with fortran unformatted data files"""

    def __init__(self,fname, mode='r', buf=0):
         """Open the file for writing, defaults to big endian."""
         file.__init__(self, fname, mode, buf)
         self.setEndian('>')

    def setEndian(self,c):
        """Set endian to big (c='>') or little (c='<') or native (c='@')
        
        :Parameters:
          `c` : string
        The endian-ness to use when reading from this file.
        """
        if c == '<' or c == '>' or c =='@' or c == '=':
            self.ENDIAN = c
        else:
            raise ValueError('Cannot set endian-ness')

    def readString(self):
        """Read in a string with error checking"""
        l = struct.unpack(self.ENDIAN+'i',self.read(4))[0]
        str = self.read(l)
        if  struct.unpack(self.ENDIAN+'i',self.read(4))[0] != l:
            raise IOError('Error reading string from data file')
        return str

    def writeString(self,s):  
        """Write a string
         
        :Parameters:
          - `s`: the string to write
        """
        self.write(struct.pack(self.ENDIAN+'i',len(s)))
        self.write(s)
        self.write(struct.pack(self.ENDIAN+'i',len(s)))

    def readReals(self, prec='d'):
        """Read in an array of reals (given precision) with error checking"""
    
        if prec not in ['d','f']:
             raise ValueError('Not an appropriate precision')

        l = struct.unpack(self.ENDIAN+'i',self.read(4))[0]
        data_str = self.read(l)
        len_real = struct.calcsize(prec)
        if l % len_real != 0:
            raise IOError('Error reading array of reals from data file')
        num = l/len_real
        reals = struct.unpack(self.ENDIAN+str(num)+prec,data_str)
        if struct.unpack(self.ENDIAN+'i',self.read(4))[0] != l:
            raise IOError('Error reading array of reals from data file')
        return list(reals)

    def writeReals(self, reals, prec='d'):
        """Write an array of floats in given precision

         :Parameters:
           `reals` : array
             Data to write
           `prec` : string
             Character code for the precision to use in writing
         """
        if prec not in ['d','f']: raise ValueError('Not an appropriate precision')
        self.write(struct.pack(self.ENDIAN+'i',len(reals)))
        for r in reals:
            self.write(struct.pack(self.ENDIAN+prec,r))
        self.write(struct.pack(self.ENDIAN+'i',len(reals)))

    def readInts(self):
        """Read in an array of integers with error checking"""
        l = struct.unpack('i',self.read(4))[0]
        data_str = self.read(l)
        len_int = struct.calcsize('i')
        if l % len_int != 0:
            raise IOError('Error reading array of integers from data file')
        num = l/len_int
        ints = struct.unpack(str(num)+'i',data_str)
        if struct.unpack(self.ENDIAN+'i',self.read(4))[0] != l:
            raise IOError('Error reading array of integers from data file')
        return list(ints)

    def readRecord(self):
         """Read a single fortran record"""
         l = struct.unpack(self.ENDIAN+'i',self.read(4))[0]
         data_str = self.read(l)
         # check length
         if len(data_str) != l:
             raise IOError('Didn''t read enough data')
         check = self.read(4)
         if len(check) != 4:
             raise IOError('Didn''t read enough data')
         if struct.unpack(self.ENDIAN+'i',check)[0] != l:
             raise IOError('Error reading record from data file')
         return data_str

class VACfile(FortranFile): #TODO: Make this not a subclass of FortranFile
    def __init__(self,fname,mode='r',buf=0):
        """Based on FortranFile has been modified to read VAC / SAC output files.
       
        Reads a iteration into the following structure:
           file.header: -Dictionary containging
                        -filehead: string at begging of file
                        -params: Iteration Parameters, it, t, ndim, neqpar, nw
                        -nx: Size of cordinate array [list]
                        -eqpar: eqpar_ parameters [list]
                        -varnames: list containg varible names for dimensions, nw and eqpar?
            file.w : w array from file which is [params,[nx]] in size
            file.w_: dict containing the {varname:index} pairs for the w array
            file.x : x array from file which is [ndim,[nx]] in size
        """
        #Do FORTRAN read init, set Endian for VAC/SAC files
        FortranFile.__init__(self,fname,mode,buf)
        self.setEndian('<')
        self.readstep()
        self.recordsize = self.tell()
        from os import stat
        self.num_records = stat(fname).st_size / self.recordsize
        
        #Find out first and last time values        
        self.t_start = self.header['params'][1]
        self.readrecord(self.num_records)
        self.t_end = self.header['params'][1]
        self.readrecord(1)
        
        print "File is %i Records Long"%self.num_records
            
    def readrecord(self,i):
        self.seek(int(i-1) * self.recordsize)
        self.readstep(i)
    
    def readParams(self,prec='d'):
        """Reads the Params line which is a mix of Ints and Reals"""
        #Check that prec is spec'd proper        
        if prec not in ['d','f']: raise ValueError('Not an appropriate precision')
        #read in line
        data_str = self.readRecord()
        pars = struct.unpack(self.ENDIAN+'idiii',data_str)
        return list(pars)
    
    def readstep(self,i=0):
        """reads one time step of data"""
        from numpy import reshape, zeros, transpose
        
        self.header = {}
        self.header['filehead'] = self.readRecord()
        self.header['params'] = self.readParams()
        #params is: it, t, ndim, neqpar, nw
        self.header['it'] = self.header['params'][0]
        self.header['t'] = self.header['params'][1]
        self.header['ndim'] = self.header['params'][2]
        self.header['neqpar'] = self.header['params'][3]
        self.header['nw'] = self.header['params'][4]
        self.header['nx'] = self.readInts()
        self.header['eqpar'] = self.readReals()
        self.header['varnames'] = self.readRecord().split()

        self.x = self.readReals()
#        s = self.header['nx'] + [self.header['ndim']]
        s = [self.header['params'][2]] + self.header['nx']
        #s = [self.header['params'][2]] + self.header['nx']
        self.x = reshape(self.x,s,order='C') ## - Don't know! Array was wrong 
        #self.E = self.readReals()
        #self.E = reshape(self.E,s)
        #shape when using F order, makes me wonder!
        
        self.w = zeros([self.header['params'][-1]]+self.header['nx'],order='C')
        for i in xrange(0,self.header['params'][-1]):
            self.w[i] = reshape(self.readReals(), self.header['nx'], order='C')
        self.w_ = {}
        ndim = self.header['params'][2]
        nw = self.header['params'][-1]
        #find h in varnames (to prevent x y h bug in 3D file)
        index = next((i for i in xrange(len(self.header['varnames'])) if not(self.header['varnames'][i] in ["x","y","z"])),ndim)
        for i,name in enumerate(self.header['varnames'][index:nw+index]):
            self.w_.update({name:i})

#==============================================================================
#         SAC calculated varible methods
#==============================================================================

class SACdata():
    """ A class containing conservative varible definitions and sac only data 
    update methods. Designed to be subclassed, so SACfile and SAChdf5 can share
    common methods without horrible file inheratance """
    def update_w_sac(self):
        self.ndim = self.header['ndim']
        self.w_sac = {}
        if self.ndim == 2:
            self.w_sac.update({'rho':self.w[self.w_["h" ]] + self.w[self.w_["rhob"]]})
            self.w_sac.update({'v1':self.w[self.w_["m1"]] / self.w_sac['rho']})
            self.w_sac.update({'v2':self.w[self.w_["m2"]] / self.w_sac['rho']})
            self.w_sac.update({'e':self.w[self.w_["e"]]+self.w[self.w_["eb"]]})
            self.w_sac.update({'b1':self.w[self.w_["b1"]] + self.w[self.w_["bg1"]]})
            self.w_sac.update({'b2':self.w[self.w_["b2"]] + self.w[self.w_["bg2"]]})
        if self.ndim == 3:
            self.w_sac.update({'rho':self.w[self.w_["h" ]] + self.w[self.w_["rhob"]]})
            self.w_sac.update({'v1':self.w[self.w_["m1"]] / self.w_sac['rho']})
            self.w_sac.update({'v2':self.w[self.w_["m2"]] / self.w_sac['rho']})
            self.w_sac.update({'v3':self.w[self.w_["m3"]] / self.w_sac['rho']})
            self.w_sac.update({'e':self.w[self.w_["e"]] + self.w[self.w_["eb"]]})
            self.w_sac.update({'b1':self.w[self.w_["b1"]] + self.w[self.w_["bg1"]]})
            self.w_sac.update({'b2':self.w[self.w_["b2"]] + self.w[self.w_["bg2"]]})
            self.w_sac.update({'b3':self.w[self.w_["b2"]] + self.w[self.w_["bg3"]]})
    
    def convert_B(self):
        from numpy import sqrt
        mu = 1.25663706e-6
        if self.ndim == 2:
            self.w_sac['b1'] *= sqrt(mu)
            self.w_sac['b2'] *= sqrt(mu)
        if self.ndim == 3:
            self.w_sac['b1'] *= sqrt(mu)
            self.w_sac['b2'] *= sqrt(mu)
            self.w_sac['b3'] *= sqrt(mu)
    
    def get_thermalp(self,beta=False):
        """Calculate Thermal pressure from varibles """
        if self.ndim == 3:
            #raise NotImplementedError("This Dosen't work for 3D yet, go fix")
            g1 = (self.header['eqpar'][0]-1)
            kp = (self.w_sac['rho'] * (self.w_sac['v1']**2 + self.w_sac['v2']**2 + self.w_sac['v3']**2))/2.
            mp = (self.w_sac['b1']**2 + self.w_sac['b2']**2 + self.w_sac['b3']**2) / 2.
            p = g1 * (self.w_sac['e'] - kp - mp)
            #p = (\gamma -1) ( e - \rho v^2/2 - B^2/2)
        else:   
            g1 = (self.header['eqpar'][0]-1)
            kp = (self.w_sac['rho'] * (self.w_sac['v1']**2 + self.w_sac['v2']**2))/2.
            mp = (self.w_sac['b1']**2 + self.w_sac['b2']**2) / 2.
            p = g1 * (self.w_sac['e'] - kp - mp)
        
        if beta:
            return p, mp
        else:
            return p
    
    def get_bgp(self):
        print "WARNING: Background Pressure will not work if inital conditions are not V=0"
        if self.ndim == 3:
            #raise NotImplementedError("This Dosen't work for 3D yet, go fix")
            g1 = (self.header['eqpar'][0]-1)
            kp = 0.0#(self.w[self.w_["rhob"]] * (self.w_sac['v1']**2 + self.w_sac['v2']**2 + self.w_sac['v3']**2))/2.
            mp = (self.w[self.w_["bg1"]]**2 + self.w[self.w_["bg2"]]**2 + self.w[self.w_["bg3"]]**2) / 2.
            p = g1 * (self.w[self.w_["eb"]] - kp - mp)
            #p = (\gamma -1) ( e - \rho v^2/2 - B^2/2)
        else:   
            g1 = (self.header['eqpar'][0]-1)
            kp = 0.0#(self.w[self.w_["rhob"]] * (self.w_sac['v1']**2 + self.w_sac['v2']**2))/2.
            mp = (self.w[self.w_["bg1"]]**2 + self.w[self.w_["bg2"]]**2) / 2.
            p = g1 * (self.w[self.w_["eb"]] - kp - mp)
        return p
    
    def get_total_p(self):
        if self.ndim == 3:
           gamma = self.header['eqpar'][0]
           
           vtot2 = (self.w_sac['v1']**2 + self.w_sac['v2']**2 + self.w_sac['v3']**2)
           therm = self.w[self.w_["e"]] - (self.w_sac["rho"] * vtot2) / 2.
           
           Bpert = self.w[self.w_['b1']] + self.w[self.w_['b2']] + self.w[self.w_['b3']]
           Bpert2 = self.w[self.w_['b1']]**2 + self.w[self.w_['b2']]**2 + self.w[self.w_['b3']]**2
           Bback = self.w[self.w_['bg1']] + self.w[self.w_['bg2']] + self.w[self.w_['bg3']]
           mag = Bback * Bpert + (Bpert2 / 2.)
           
           return (gamma - 1) * therm - (gamma - 2) * mag
        else:
            raise NotImplementedError("This Dosen't work for 2D yet, go fix")
    
    def get_temp(self,p=None):
        if not(p):
            p = self.get_thermalp()
        T = (p * 1.2) / (8.3e3 * self.w_sac['rho'])
        return T
    
    def get_bgtemp(self):
        print "WARNING: Background Temprature will not work if inital conditions are not V=0"
        if self.ndim == 3:
            kp = 0.0#(self.w[self.w_["rhob"]] * (self.w_sac['v1']**2 + self.w_sac['v2']**2 + self.w_sac['v3']**2))/2.
            mp = (self.w[self.w_["bg1"]]**2 + self.w[self.w_["bg2"]]**2 + self.w[self.w_["bg3"]]**2) / 2.
            T = self.w[self.w_["eb"]] - kp - mp
        else:
            kp = 0.0#(self.w[self.w_["rhob"]] * (self.w_sac['v1']**2 + self.w_sac['v2']**2))/2.
            mp = (self.w[self.w_["bg1"]]**2 + self.w[self.w_["bg2"]]**2) / 2.
            T = self.w[self.w_["eb"]] - kp - mp
        return T
    
    def get_va(self):
        from numpy import sqrt#, abs
        return (sqrt(self.w_sac['b1']**2 + self.w_sac['b2']**2 + self.w_sac['b3']**2)/sqrt(self.w_sac['rho']))
        #return (abs(self.w_sac['b1']) + abs(self.w_sac['b2']) + abs(self.w_sac['b3'])) / sqrt(self.w_sac['rho'])
    
    def get_cs(self,p=None):
        from numpy import sqrt
        if not p:
            p = self.get_thermalp()
        g1 = self.header['eqpar'][0]
        return sqrt((g1 * p) / self.w_sac['rho'])

#==============================================================================
#       END SAC calculated varible methods
#==============================================================================


class SACfile(VACfile,SACdata):
    """Specification of VACFile for the reading of SAC files, to calculate the
       total from the background and pertuabation parts.
       Also defines:
           w_sac: Dict, containing conservative varibles as sum of
                   background and pertubation parts.
       """
    def __init__(self,fname,mode='r',buf=0):
        VACfile.__init__(self,fname,mode,buf)
    
    def readstep(self,i=0):
        VACfile.readstep(self,i)
        """Need a good way of determining the number of varibles that are 
        background or pertuabation"""
        
        #split up into background / pertubation varibles
        self.update_w_sac()


#==============================================================================
# HDF5 File classes
#==============================================================================

class VAChdf5():
    import h5py
    from numpy import array
    def __init__(self,filename):
        """Based on FortranFile has been modified to read VAC / SAC HDF5 files.
       
        Reads a iteration into the following structure:
           file.header: -Dictionary containging
                        -filehead: string at begging of file
                        -params: Iteration Parameters, it, t, ndim, neqpar, nw
                        -nx: Size of cordinate array [list]
                        -eqpar: eqpar_ parameters [list]
                        -varnames: list containg varible names for dimensions, nw and eqpar?
            file.w : w array from file which is [params,[nx]] in size
            file.w_: dict containing the {varname:index} pairs for the w array
            file.x : x array from file which is [ndim,[nx]] in size
        
        Also creates HDF5 specific attributes:
            file.sac_group - Holds the x and time_group attributes.
            file.time_group - Holds the series of w arrays.
        
        Largely the HDF5 file is designed so the functionality mimics the VAC
        binary file, i.e. all the vars are still in the W array etc.
        """
        self.h5file = self.h5py.File(filename,'r')
        #Open top level group
        if not("SACdata" in self.h5file.keys()):
            print """Are you sure this is a proper SAC HDF5 file?
                Opening first group."""
            self.sac_group = self.h5file[self.h5file.keys()[0]]
        else:
            self.sac_group = self.h5file["SACdata"]
            
        self.x = self.sac_group['x']
        
        self.header = dict(self.sac_group.attrs)
        self.header['neqpar'] = int(self.header['neqpar'])
        self.header['ndim'] = int(self.header['ndim'])
        try:
            self.header['filehead'] = self.h5file.attrs['filehead'][0]
        except:
            pass
        
        self.time_group = self.sac_group['wseries']
        self.header.update(dict(self.time_group.attrs))
        self.header['varnames'] = self.header['varnames'][0].split()
        self.readrecord(0)
        self.t_start = self.header['t']
        self.t_end = self.time_group.items()[-1][1].attrs['t'][0]
        self.num_records = len(self.time_group.items())
    
    def readrecord(self,i):
        wstepname = self.time_group.keys()[i]
        self.header['it'] = int(self.time_group[wstepname].attrs['it'])
        self.header['t'] = float(self.time_group[wstepname].attrs['t'])
        #to maintain backwards compatibility with VACfile
        self.header['params'] = [self.header['it'], self.header['t'], 
                                self.header['ndim'], self.header['neqpar'], 
                                self.header['nw']]
        
        self.w = self.time_group[wstepname]
        self.w_ = {}
        index = next((i for i in xrange(len(self.header['varnames'])) if not(self.header['varnames'][i] in ["x","y","z"])),self.header['ndim'])
        for i,name in enumerate(self.header['varnames'][index:self.header['nw']+index]):
            self.w_.update({name:i})

    def readstep(self,i=0):
        raise Warning("""readstep() isn't implemented for the HDF5
        file, it will read first record unless i is specified""")
        self.readrecord(i)

class SAChdf5(VAChdf5,SACdata):
    """Specification of VAChdf5 for the reading of SACHDF5 files, to calculate the
       total from the background and pertuabation parts.
       Also defines:
           w_sac: Dict, containing conservative varibles as sum of
                   background and pertubation parts.
       """
    def __init__(self,fname,mode='r',buf=0):
        VAChdf5.__init__(self,fname)
    
    def readrecord(self,i=0):
        VAChdf5.readrecord(self,i)
        """Need a good way of determining the number of varibles that are 
        background or pertuabation"""

        self.update_w_sac()
        