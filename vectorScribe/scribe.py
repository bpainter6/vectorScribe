# -*- coding: utf-8 -*-
"""
Created on Mon May  5 19:14:28 2025

@author: 17066
"""

import numpy as np
import pandas as pd
from copy import deepcopy

import vectorScribe.checkerrors as err
from warnings import warn

class Scribe():
    """Object that conveniently stores data in a "vector" structure
    and various "array" structures. Also provides methods that allow users to 
    specify data using one structure and then convert it to another 
    structure"""
    
    def __init__(self, template, vec_key, iterables=None, names=None, 
                 src_keys=None, fill=0, dtype=float, shape=tuple(), 
                 mode="mean"):
        
        """Initialize the scribe
        
        Parameters
        ----------
        template - dict of nd iterable of str
            Provides the structure for all arrays associated with the scribe.
            
            The array corresponding to ``vec_key`` specifies all the region ids
            that will be used in the Scribe. Each element of the array 
            corresponding to ``vec_key`` must be unique.
            
            Every other array provides various ways to map the data 
            corresponding to ``vec_key``. Elements in every other array do NOT
            need to be unique.
        vec_key - str
            indicates which of the string arrays contained in ``template``
            correspond to the "vector" structure.
        iterables - list of list of str
            Provides the index names for multiindexing the data which is stored
            in a pandas DataFrame
        names - list of str
            Provides the names of each "layer" of the multiindexing
        src_key - str
            Indicates which of the arrays will contain the source data to 
            populate the other arrays. This will be the default for each index,
            and it can be changed later.
        fill - int/float/numpy nan/numpy inf
            Specifies the number that will be used to regions that do not 
            contain any data. This will be the default for each index, and it 
            can be changed later.
        dtype - str/object name
            Specifies whether data is int or float type. This will be the 
            default for each index, and it can be changed later.
        shape - tuple of ints
            Provides the shape of the data beyound axis 0
        mode - str
            Specifies whether data is summed or average when moving from an 
            array format to the vector format. This will be the default for 
            each index, and it can be changed later.
        
        """
        
        # Default value for multiindexing
        if iterables is None:
            iterables = [["dat"]]
        
        # Preallocate the dataframe
        arr_keys = list(template.keys())
        index = pd.MultiIndex.from_product(iterables, names=names)
        
        # Force None into each element of the DataFrame
        nr = len(index)
        nc = len(arr_keys)+1
        data = [[None]*nc]*nr
        
        self.df = pd.DataFrame(data, index=index, 
                               columns=(arr_keys + ["settings"]), dtype=object)
        
        # Insert copies of the default settings
        settings = (src_keys,fill,dtype,shape,mode)
        self.df.loc[:,"settings"] = [deepcopy(settings) for _ in range(nr)]
        
        # Handle idn_vec first
        idn_vec = err._to_array(template[vec_key],vec_key)
        err._iterable_dims(  idn_vec,vec_key,[1])
        err._iterable_str(   idn_vec,vec_key)
        err._iterable_unique(idn_vec,vec_key)
        
        ni = len(idn_vec)
        idn_tmp = {vec_key:idn_vec}
        msk_tmp = {vec_key:np.full(ni,True)}
        idx_tmp = {vec_key:np.linspace(0,ni-1,ni,dtype=int)}
        
        # Convert each array in the region group to a numpy array
        for arr_key,idn_arr in template.items():
            
            # For the vector data
            if arr_key==vec_key:
                continue
            
            # Check that reg_arr is an array of str
            err._iterable_str(idn_arr,arr_key)
            
            # Check that reg_arr can be converted to a numpy array. And if so,
            # convert it
            idn_arr = err._to_array(idn_arr,arr_key)
            
            # Identify locations in the array where data is undefined by the 
            # vector
            msk_arr = np.logical_not(np.isin(idn_arr, idn_vec))
            
            # Convert to index array (accounting for lexographic order)
            lex_ord = np.argsort(idn_vec)
            idn_vec_lex = idn_vec[lex_ord]
            idx_arr_lex = np.searchsorted(idn_vec_lex, idn_arr)
            idx_arr_lex[msk_arr] = 0
            idx_arr = lex_ord[idx_arr_lex]
            
            idn_tmp[arr_key] = idn_arr
            msk_tmp[arr_key] = msk_arr
            idx_tmp[arr_key] = idx_arr
        
        self.ni       = ni
        self.vec_key  = vec_key
        self.arr_keys = arr_keys
        self.idn_tmp  = idn_tmp
        self.msk_tmp  = msk_tmp
        self.idx_tmp  = idx_tmp
    
    def vec_to_arr(self, dat_vec, arr_key, fill,  dtype):
        """Function for transcribing data stored in dat_vec to an arr format. 
        This mapping is based on the user-defined template and a given arr_key.
        
        Parameters
        ----------
        dat_vec - 1d iterable of int/float
            data in vec format
        arr_key - str 
            indicates which array the vector is transcribed into
        """
        
        # Skip undefined data vectors
        if dat_vec is None:
            return 
        
        # Extract useful parameters
        vec_key = self.vec_key
        idx_vec = self.idx_tmp[vec_key]
        msk_arr = self.msk_tmp[arr_key]
        idx_arr = self.idx_tmp[arr_key]
        
        # Check that the data is in the proper vector format
        dat_vec = err._vec_consistency(dat_vec,vec_key,idx_vec)
        
        # Transcribe the data
        dat_arr = np.take(dat_vec, idx_arr, axis=0)
        dat_arr[msk_arr] = fill
        
        # Enforce dtype of the array
        return dat_arr.astype(dtype)
    
    def arr_to_vec(self, dat_arr, arr_key):
        """Function for mapping data stored in an array format to vector format. 
        This mapping is based on the user-defined template
        
        Parameters
        ----------
        dat_arr - numpy array
            Contains the source data to be mapped to the vector
        arr_key - str
            Identifies which array in the group is providing the source data
        
        """
        
        # Extract useful parameters
        ni      = self.ni
        msk_arr = self.msk_tmp[arr_key]
        idx_arr = self.idx_tmp[arr_key]
        
        # Flatten data
        msk_flt = np.logical_not(msk_arr.flatten())
        idx_flt = idx_arr.flatten()
        flt_len = len(idx_flt)
        dat_flt_shp = (flt_len,) + np.shape(dat_arr)[idx_arr.ndim:]
        dat_flt = dat_arr.reshape(dat_flt_shp)
        
        # Get the shape of the vectorized data
        dat_vec_shp = (ni,) + np.shape(dat_arr)[idx_arr.ndim:]
        
        # Mask out nonexistent data
        idx_flt = idx_flt[msk_flt]
        dat_flt = dat_flt[msk_flt]
        
        # Tally data
        sum_vec = np.zeros(dat_vec_shp)
        cnt_vec = np.zeros(dat_vec_shp,dtype=int) 
        one_flt = np.ones_like(dat_flt,dtype=int)
        np.add.at(sum_vec,idx_flt,dat_flt)
        np.add.at(cnt_vec,idx_flt,one_flt)
        
        return sum_vec,cnt_vec
    
    def transcribe_row(self,row):
        """Covert data that was specified by the user into the various 
        formats along a given row of the DataFrame"""
        
        ni       = self.ni
        vec_key  = self.vec_key
        arr_keys = self.arr_keys
        idx_tmp  = self.idx_tmp
        try:
            src_keys, fill, dtype, shape, mode = row["settings"]
        except:
            pass
        
        # Vector is source by default. Reflect this selection in settings
        if src_keys is None:
            src_keys = [vec_key]
        
        # Ensure that src_keys are iterable
        try:
            iter(src_keys)
        except:
            src_keys = [src_keys]
        
        src_bool = [False]*len(arr_keys) # Whether each array is a source array
        spc_bool = [True]*len(arr_keys) # Whether each array has been specified
                                        # by the user
        
        for i,arr_key in enumerate(arr_keys):
            if arr_key in src_keys:
                src_bool[i] = True
            if row[arr_key] is None:
                spc_bool[i] = False
            
            if src_bool[i] and not spc_bool[i]:
                warn("No Usable Data found for {}. It will not be used as a \
                     source".format(arr_key))
        
        # Remove any src_keys that don't have corresponding data
        arr_bool = np.logical_and(src_bool,spc_bool)
        src_keys = np.array(arr_keys)[arr_bool]
        
        # Skip transcription if there is no usable source data
        if len(src_keys)==0:
            warn("No usable data found! Skipping transcription!")
            return
        
        # Preallocate the tally for filling the data vector
        if vec_key in src_keys:
            sum_vec = row[vec_key]
            cnt_vec = np.ones_like(sum_vec)
        else:
            # Get the expected shape of the data vector
            arr_key = src_keys[0]
            dat_arr = row[arr_key]
            idx_arr = idx_tmp[arr_key]
            dat_vec_shp = (ni,) + np.shape(dat_arr)[idx_arr.ndim:]
            
            sum_vec = np.zeros(dat_vec_shp,dtype=float)
            cnt_vec = np.zeros_like(sum_vec)
        
        # Now fill the vector from the other sources
        for arr_key in src_keys:
            
            # Skip the data vector. It is being written to
            if arr_key==vec_key:
                continue
            
            # The array data
            dat_arr = row[arr_key]
            
            # Get the counts and summed values from the array
            sum_tmp,cnt_tmp = self.arr_to_vec(dat_arr,arr_key)
            
            # Sum the tabulated data
            sum_vec += sum_tmp
            cnt_vec += cnt_tmp
        
        # Insert fill data for the vector and return
        if mode=="mean":
            nbl_vec = cnt_vec!=0
            avg_vec = np.full_like(sum_vec,fill)
            avg_vec[nbl_vec] = sum_vec[nbl_vec]/cnt_vec[nbl_vec]
            row[vec_key] = avg_vec.astype(dtype)
        elif mode=="sum":
            zbl_vec = cnt_vec==0
            sum_vec[zbl_vec] = fill
            row[vec_key] = sum_vec.astype(dtype)
        else:
            # Raise error
            pass
        
        # Now save the shape of the vector data beyond axis 0
        shape = np.shape(row[vec_key])[:-1]
        
        # Now take the data in the vector format and convert to the remaining
        # array formats
        for arr_key in self.arr_keys:
            
            # Skip vector data or any other source data
            if arr_key in src_keys or arr_key==vec_key:
                continue
            
            row[arr_key] = self.vec_to_arr(row[vec_key],arr_key,fill,dtype)
        
        # Save updated settings
        row["settings"] = (src_keys,fill,dtype,shape,mode)
    
    def transcribe(self):
        """Covert data that was specified by the user into the various 
        formats across all rows of the data frame"""
        
        # Apply unit_transcribe to each row in the data frame
        self.df.apply(lambda row: self.transcribe_row(row), axis=1)