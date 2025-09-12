# -*- coding: utf-8 -*-
"""
Created on Mon May  5 19:14:28 2025

@author: 17066
"""

import numpy as np
import pandas as pd
from copy import deepcopy

import vectorScribe.checkerrors as err

class Scribe():
    """Object that conveniently stores data in a "vector" structure
    and various "array" structures. Also provides methods that allow users to 
    specify data using one structure and then convert it to another 
    structure"""
    
    def __init__(self, template, vec_key, iterables=[], names=[], src_key=None, 
                 fill=0, dtype=float, mode="mean"):
        
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
        mode - str
            Specifies whether data is summed or average when moving from an 
            array format to the vector format. This will be the default for 
            each index, and it can be changed later.
        
        """
        
        # Preallocate the dataframe
        arr_keys = list(template.keys())
        index = pd.MultiIndex.from_product(iterables, names=names)
        
        # Force None into each element of the DataFrame
        nr = len(index)
        nc = 3
        data = [[None]*nc]*nr
        
        self.df = pd.DataFrame(data, index=index, 
                               columns=(arr_keys + ["settings"]), dtype=object)
        
        # Insert copies of the default settings
        settings = (src_key,fill,dtype,mode)
        self.df.loc[:,"settings"] = [deepcopy(settings) for _ in range(nr)]
        
        # Handle idn_vec first
        idn_vec = err._to_array(template[vec_key],vec_key)
        err._iterable_dims(  idn_vec,vec_key,[1])
        err._iterable_str(   idn_vec,vec_key)
        err._iterable_unique(idn_vec,vec_key)
        
        ni = len(idn_vec)
        idn_tmp = {vec_key:idn_vec}
        msk_tmp = {vec_key:np.full(ni,True)}
        idx_tmp = {vec_key:np.linspace(0,ni-1,ni)}
        
        # Convert each array in the region group to a numpy array
        for arr_key,idn_arr in template.items():
            
            # For the vector data
            if arr_key==vec_key:
                continue
            
            # Check that reg_arr is an array of str
            err._iterable_str(idn_arr,arr_key)
            
            # Check that reg_arr can be converted to a numpy array. And if so,
            # convert it
            idn_tmp[arr_key] = err._to_array(idn_arr,arr_key)
            
            # Specify the rdx array
            msk_arr = np.logical_not(np.isin(idn_arr, idn_vec))
            idx_arr = np.searchsorted(idn_vec, idn_arr)
            idx_arr[msk_arr] = 0
            
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
        idn_vec = self.idn_tmp[vec_key]
        msk_arr = self.msk_tmp[arr_key]
        idx_arr = self.idx_tmp[arr_key]
        
        # Check that the data is in the proper vector format
        err._vec_consistency(dat_vec,vec_key,idn_vec)
        
        # Transcribe the data
        dat_arr = np.take(dat_vec, idx_arr, axis=0)
        dat_arr[msk_arr] = fill
        
        # Enforce dtype of the array
        return dat_arr.astype(dtype)
    
    def arr_to_vec(self, dat_arr, arr_key, fill, dtype, mode):
        """Function for mapping data stored in an array format to vector format. 
        This mapping is based on the user-defined template
        
        Parameters
        ----------
        dat_arr - numpy array
            Contains the source data to be mapped to the vector
        arr_key - str
            Identifies which array in the group is providing the source data
        mode - str
            Indicates whehter data is summed or averaged when transcribing to 
            vec format
        
        """
        
        # Extract useful parameters
        ni      = self.ni
        msk_arr = self.msk_tmp[arr_key]
        idx_arr = self.idx_tmp[arr_key]
        
        # Flatten data
        msk_flt = np.logical_not(msk_arr.flatten())
        idx_flt = idx_arr.flatten()
        flt_len = len(idx_flt)
        if np.size(dat_arr)==flt_len:
            dat_flt = dat_arr.flatten()
        else:
            dat_flt = dat_arr.reshape(flt_len,-1)
        
        # Get the shape of the vectorized data
        dat_vec_shp = (ni,) + np.shape(dat_arr)[idx_arr.ndim:]
        
        # Mask out nonexistent data
        idx_flt = idx_flt[msk_flt]
        dat_flt = dat_flt[msk_flt]
        
        # Allocate data
        sum_vec = np.zeros(dat_vec_shp)
        cnt_vec = np.zeros(dat_vec_shp,dtype=int) 
        one_flt = np.ones_like(dat_flt,dtype=int)
        np.add.at(sum_vec,idx_flt,dat_flt)
        np.add.at(cnt_vec,idx_flt,one_flt)
        
        # Insert fill data and return the result
        if mode=="mean":
            nbl_vec = cnt_vec!=0
            avg_vec = np.full_like(sum_vec,fill)
            avg_vec[nbl_vec] = sum_vec[nbl_vec]/cnt_vec[nbl_vec]
            return avg_vec.astype(dtype)
        elif mode=="sum":
            zbl_vec = cnt_vec==0
            sum_vec[zbl_vec] = fill
            return sum_vec.astype(dtype)
        else:
            # Raise error
            pass
    
    def transcribe_row(self,row):
        """Covert data that was specified by the user into the various 
        formats along a given row of the DataFrame"""
        
        vec_key = self.vec_key
        src_key, fill, dtype, mode = row["settings"]
        
        # Vector is source by default. Reflect this selection in settings
        if src_key is None:
            src_key = vec_key
            row["settings"] = (src_key,fill,dtype,mode)
        
        # Skip if source data is undefined
        if row[src_key] is None:
            return 
        
        # First convert the source in array format to a vector format
        if src_key!=vec_key:
            row[vec_key] = self.arr_to_vec(row[src_key],src_key,fill,dtype,mode)
        
        # Then take the data in the vector format and conver to all other array
        # Formats
        elif src_key==vec_key:
            for arr_key in self.arr_keys:
                if arr_key in [vec_key,src_key]:
                    continue
                row[arr_key] = self.vec_to_arr(row[src_key],arr_key,fill,dtype)
    
    def transcribe(self):
        """Covert data that was specified by the user into the various 
        formats across all rows of the data frame"""
        
        # Apply unit_transcribe to each row in the data frame
        self.df.apply(lambda row: self.transcribe_row(row), axis=1)