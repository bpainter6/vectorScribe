# -*- coding: utf-8 -*-
"""
Created on Mon May  5 19:16:47 2025

@author: 17066
"""

import numpy as np

##############################################################################
###     TYPE ERRROR CHECKS      ##############################################
##############################################################################

def _is_type(var,var_name,types):
    """Raises error if the type of ``var`` is not in ``types``"""
    if type(var) not in types:
        raise TypeError("{} must be one of the following types: {}. Instead, \
                        got: {}".format(var_name,types,var))

def _to_array(var,var_name):
    """Function for converting an arbitrary nested iterable into a numpy 
    array. Also raises error if a given variable can not be converted to an 
    array"""
    if isinstance(var,np.ndarray):
        return var
    else:
        try:
            return np.array(var)
        except:
            raise TypeError("{} cannot be converted to a numpy array. Instead,\
                            got: {}".format(var_name,var))

def _iterable_unique(var,var_name):
    """Raises error if all the elements of ``var`` are not unique."""
    arr = _to_array(var,var_name)
    if np.size(arr) != np.size(np.unique(arr)):
        raise TypeError("{} must contain all unique elments. Instead, got: {}"\
                        .format(var_name,var))

def _iterable_dims(var,var_name,dims):
    """Raises error if the number of dimensions of ``var`` is not in ``dims``
    """
    arr = _to_array(var,var_name)
    if len(np.shape(arr)) not in dims:
        raise TypeError("{} must be an iterable with one of the following \
                        dimensions: {}. Instead, got: {}"\
                        .format(var_name,dims,var))

def _iterable_types(var,var_name,types,empty_allowed=False):
    """Raises error if the type of the elements in ``var`` is not in ``types``
    """
    arr = _to_array(var,var_name)
    if empty_allowed and arr.size==0:
        return
    if arr.dtype not in types:
        raise TypeError("{} must only contain elements that are one of the \
                        following types: {}. Instead, got: {}"\
                        .format(var_name,types,var))

def _iterable_str(var,var_name,empty_allowed=False):
    """Raises error if the elements in ``var`` are not strings"""
    arr = _to_array(var,var_name)
    if empty_allowed and arr.size==0:
        return
    if arr.dtype.type is not np.str_:
        raise TypeError("{} must only contain elements that are strings. \
                        Instead, got: {}".format(var_name,var))

def _iterable_shape(var,var_name,shape):
    """Raises error if the shape of ``var`` does not match ``shape``"""
    arr = _to_array(var,var_name)
    if np.shape(arr) != shape:
        raise TypeError("{} must have the following shape: {}. Instead, got: \
                        {}".format(var_name,shape,var))

def _vec_consistency(vec,vec_name,univ_vec):
    """Raises error if ``vec`` is not consistent with ``univ_vec``"""
    univ_shape = np.shape(univ_vec)
    
    # Ensure that vector has proper shape
    _iterable_shape(vec,vec_name,univ_shape)
    
    # Convert vector to numpy array
    vec = _to_array(vec,vec_name)
    
    return vec

def _grp_consistency(grp,grp_name,univ_grp):
    """Raises error if ``grp`` is not consistent with a ``univ_grp``"""
    
    # Perform error checking of each array in the group
    for arr_id,data_arr in grp.items():
        univ_arr = univ_grp[arr_id]
        arr_shape = np.shape(univ_arr)
        
        # Construct a representative array name
        arr_name = grp_name+"-"+arr_id
        
        # Ensure that the data array has proper shape
        _iterable_shape(data_arr,arr_name,arr_shape)
        
        # Convert the data array to numpy array
        data_arr = _to_array(data_arr,arr_name)
        
        grp[arr_id] = data_arr
    
    return grp

##############################################################################
###     VALUE ERRROR CHECKS      #############################################
##############################################################################

def _is_value(var,var_name,vals):
    """Raises error if the value of ``var`` is not in ``vals``"""
    if var not in vals:
        raise ValueError("{} must be one of the following values: {}. Instead \
                         got: {}".format(var_name,vals,var))

def _is_all_zero_iterable(var,var_name):
    """Raises error if any of the elements in ``var`` are not zero"""
    arr = _to_array(var)
    if not (arr == 0.0).all():
        raise ValueError("{} must only contain numbers that are equal to zero.\
                         Instead, got: {}".format(var_name,var))

def _is_positive_iterable(var,var_name):
    """Raises error if any of the elements in ``var`` are not positive"""
    arr = _to_array(var)
    if not (arr >= 0.0).all():
        raise ValueError("{} must only contain numbers that are positive.\
                         Instead, got: {}".format(var_name,var))

def _is_strictly_positive_iterable(var,var_name):
    """Raises error if any of the elements in ``var`` are not strictly positive
    """
    arr = _to_array(var)
    if not (arr > 0.0).all():
        raise ValueError("{} must only contain numbers that are strictly \
                         positive. Instead, got: {}".format(var_name,var))

def _is_real_iterable(var,var_name):
    """Raises error if any of the elements in ``var`` are not real (complex)"""
    arr = _to_array(var)
    if arr.dtype is complex:
        raise ValueError("{} must only contain real numbers. Instead, got:\
                         {}".format(var_name,var))

def _iterable_value(var,var_name,values):
    """Raises error if any of the elements in ``var`` is not in ``values``"""
    arr = _to_array(var,var_name)
    if not np.all( np.isin(arr,values) ):
        raise ValueError("At least one of the elements in {} is not one of the\
                         following values: {}. Instead, got: {}"\
                         .format(var_name,values,var))