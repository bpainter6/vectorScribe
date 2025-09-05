# -*- coding: utf-8 -*-
"""
Created on Mon May  5 19:14:28 2025

@author: 17066
"""

from copy import deepcopy
import numpy as np

from vectorScribe.functions.scribes import vec_to_grp, grp_to_vec
import vectorScribe.functions.checkerrors as err

class Template():
    """Object that provides useful attributes for allowing scriges to quickly 
    translating between data vectors and data groups"""
    
    def __init__(self, id_vec, id_grp, fill=0, dtype=float):
        """Check whether univ_vec and univ_grp are appropriately defined and
        calculate the mapping between univ_vec and univ_grp. 
        
        Parameters
        ----------
        id_vec - 1d iterable of str
            Specifies all the ids that will be used in the template. Each 
            element of univ_vec must be unique.
        id_grp - dict of Nd iterable of str
            specifies how the universes in univ_vec are grouped for prediction 
            models. Each array in univ_grp must be able to be converted to a 
            numpy array. Elements in each array do not need to be unique.
        
        """
        
        # Identify all the potential dimensions of arrays in the group. Also,
        # convert arrays in the universe group to numpy arrays
        arr_dims = []
        for arr_key,id_arr in id_grp.items():
            arr_dim = len(np.shape(id_arr))
            if arr_dim not in arr_dims:
                arr_dims.append(arr_dim)
            
            # Check that univ_arr is an array of str
            err._iterable_str(id_arr,"id_arr")
            
            # Check that univ_arr can be converted to a numpy array. And if so,
            # convert it
            id_grp[arr_key] = err._to_array(id_arr,"id_arr")
        
        # Make sure if all elements of univ_vec are unique
        err._iterable_dims(  id_vec,"id_vec",[1])
        err._iterable_str(   id_vec,"id_vec")
        err._iterable_unique(id_vec,"id_vec")
        
        # Tracks the number of occurences of each universe in the group
        # count_vec = np.zeros_like(id_vec,dtype=float)
        
        # Create the univ_map and fill the count vector
        id_map = {}
        for ndx,Id in enumerate(id_vec):
            id_map[Id] = {}
            for arr_key,id_arr in id_grp.items():
                mask = np.isin(id_arr,Id)
                if np.any(mask):
                    id_map[Id][arr_key] = np.where(mask)
        
        # Create the null vector and group
        null_vec = np.full_like(id_vec,fill,dtype=dtype)
        null_grp = {}
        for arr_key,id_arr in id_grp.items():
            null_grp[arr_key] = np.full_like(id_arr,fill,dtype=dtype)
        
        self.id_vec    = id_vec
        self.id_grp    = id_grp
        self.arr_dims  = arr_dims
        self.id_map    = id_map
        self.null_vec  = null_vec
        self.null_grp  = null_grp

class Quick_scribe():
    """Object that conveniently stores data in a "vector" structure
    and "group" structure. Also provides methods that allow users to specify
    data using one structure and then convert it to another structure"""
    
    def __init__(self):
        """Initialize the scribe"""
        pass
    
    def set_template(self, template):
        """Set the template for the scribe object"""
        
        # Check that template is a Template object
        err._is_type(template,"template",[Template])
        
        self.template = template
    
    def set_vec(self,data_vec):
        """Specify data at a specific state in vector form
        
        Parameters
        ----------
        kwargs - keyword args
            keyword arguments specifing the input data.
            keyword - data_key
            argument - data_vec
        
        """
        
        id_vec = self.template.id_vec
        
        # Check that data_vec is consistent with univ_vec
        data_vec = err._vec_consistency(data_vec,"data_vec",id_vec)
        
        self.vec = data_vec
    
    def set_grp(self,data_grp):
        """Specify data at a specific state in group form
        
        Parameters
        ----------
        kwargs - keyword args
            keyword arguments specifing the input data.
            keyword - data_id
            argument - data_grp
        
        """
        
        id_grp = self.template.id_grp
        
        # Check that data_grp is consitent with univ_grp
        data_grp = err._grp_consistency(data_grp,"data_key",id_grp)
        
        self.grp = data_grp
    
    def vec_to_grp(self):
        """Covert data that was specified in vector form into data that is 
        specified in group form"""
        
        # Extract useful attributes
        template = self.template
        vec = self.vec
        
        # Store the updated data groups
        self.grp = vec_to_grp(template,vec)
    
    def grp_to_vec(self,arr_keys=None,mode="mean"):
        """Covert data that was specified in group form into data that is 
        specified in vector form"""
        
        # Extract useful attributes
        template = self.template
        grp = self.grp
        
        self.vec = grp_to_vec(template,grp,arr_keys,mode)

class Data_scribe():
    """Object that conveniently stores data in a "vector" structure
    and "group" structure. Also provides methods that allow users to specify
    data using one structure and then convert it to another structure"""
    
    def __init__(self):
        """Initialize the scribe"""
        pass
    
    def set_template(self, template):
        """Set the template for the scribe object"""
        
        # Check that template is a Template object
        err._is_type(template,"template",[Template])
        
        self.template = template
    
    def set_keys(self, data_keys):
        """Specify settings pertaining to data keys"""
        
        # Make sure that state_ids is a 1d iterable
        err._iterable_dims(data_keys,"data_keys",[1])
        
        # Extract useful parameters
        template = self.template
        id_grp   = template.id_grp
        
        vecs = {}
        grps = {}
        
        for data_key in data_keys:
            vecs[data_key] = None
            grps[data_key] = {}
            
            for arr_key,id_arr in id_grp.items():
                grps[data_key][arr_key] = None
        
        self.data_keys = data_keys
        self.vecs      = vecs
        self.grps      = grps
    
    def add_keys(self, data_keys):
        """Specify additional keys in the object"""
        
        # Make sure that state_ids is a 1d iterable
        err._iterable_dims(data_keys,"data_keys",[1])
        
        # Extract useful parameters
        template = self.template
        vecs     = self.vecs
        grps     = self.grps
        id_grp   = template.id_grp
        
        for data_key in data_keys:
            vecs[data_key] = None
            grps[data_key] = {}
            
            for arr_key,id_arr in id_grp.items():
                grps[data_key][arr_key] = None
        
        self.data_keys = data_keys
        self.vecs      = vecs
        self.grps      = grps
    
    def set_vecs(self,**kwargs):
        """Specify data at a specific state in vector form
        
        Parameters
        ----------
        kwargs - keyword args
            keyword arguments specifing the input data.
            keyword - data_key
            argument - data_vec
        
        """
        
        id_vec = self.template.id_vec
        
        # Unpack kwargs and insert data
        for data_key,data_vec in kwargs.items():
            # Check that data_vec is consistent with univ_vec
            data_vec = err._vec_consistency(data_vec,data_key,id_vec)
            
            # Ensure that data_id is in the data_ids of Data_scribe
            if data_key not in self.data_keys:
                raise ValueError("{} is not one of the following valid ids: {}\
                                 ".format(data_key,self.data_keys))
            
            self.vecs[data_key] = data_vec
    
    def set_grps(self,**kwargs):
        """Specify data at a specific state in group form
        
        Parameters
        ----------
        kwargs - keyword args
            keyword arguments specifing the input data.
            keyword - data_id
            argument - data_grp
        
        """
        
        id_grp = self.template.id_grp
        
        # Unpack kwargs and insert data
        for data_key,data_grp in kwargs.items():
            # Check that data_grp is consitent with univ_grp
            data_grp = err._grp_consistency(data_grp,data_key,id_grp)
            
            # Ensure that data_id is in the data_ids of Data_scribe
            if data_key not in self.data_keys:
                raise ValueError("{} is not one of the following valid ids: {}\
                                 ".format(data_key,self.data_keys))
            
            self.grps[data_key] = data_grp
    
    def vecs_to_grps(self):
        """Covert data that was specified in vector form into data that is 
        specified in group form"""
        
        # Extract useful attributes
        template = self.template
        grps  = self.grps
        vecs  = self.vecs
        
        # Perform the conversion for each data_id
        for data_key,vec in vecs.items():
            # Calculate the new data_grp
            grps[data_key] = vec_to_grp(template,vec)
        
        # Store the updated data groups
        self.grps = grps
    
    def grps_to_vecs(self,arr_keys=None,mode="mean"):
        """Covert data that was specified in group form into data that is 
        specified in vector form"""
        
        # Extract useful attributes
        template = self.template
        grps  = self.grps
        vecs  = self.vecs
        
        # Perform the conversion for each state_id and each data_id
        for data_key,grp in grps.items():
            # Store the result
            vecs[data_key] = grp_to_vec(template,grp,arr_keys,mode)
        
        # Store the updated data vectors
        self.vecs = vecs

class Multi_data_scribe(dict):
    """Object that stores scribe objects by states and can perform 
    operations on each state"""
    
    def __init__(self,state_keys):
        """Preallocate a data scribe for each state"""
        
        # Make sure that state_ids is a 1d iterable
        err._iterable_dims(state_keys,"state_keys",[1])
        
        for state_key in state_keys:
            self[state_key] = Data_scribe()
    
    def set_tenplate(self,template):
        """set the template for each state"""
        
        # Error checking is performed in each Data_scribe
        for state_key in self:
            self[state_key].set_template(template)
    
    def set_keys(self,data_keys):
        """set ids for each state"""
        
        # Make sure that data_ids is a 1d iterable
        err._iterable_dims(data_keys,"data_keys",[1])
        
        for state_key in self:
            self[state_key].set_ids(data_keys)
    
    def vecs_to_grps(self):
        """Perform the vecs_to_grps operation for each state"""
        
        for state_key in self:
            self[state_key].vecs_to_grps()
    
    def grps_to_vecs(self):
        """Perform the grps_to_vecs operation for each state"""
        
        for state_key in self:
            self[state_key].grps_to_vecs()
    
    def add_states(self,state_keys):
        """Add states to the Multi_scribe"""
        
        # Make sure that state_ids is a 1d iterable
        err._iterable_dims(state_keys,"state_keys",[1])
        
        for state_key in state_keys:
            self[state_key] = Data_scribe()
