# -*- coding: utf-8 -*-
"""
Created on Mon May  5 19:15:46 2025

@author: 17066
"""

import numpy as np
from copy import deepcopy
    
def vec_to_grp(template,vec):
    """Function for mapping data stored in data_vec to data_group. This mapping 
    is based on the user-defined template"""
    
    # Extract useful parameters
    id_vec  = template.id_vec
    id_map  = template.id_map
    null_grp  = template.null_grp
    
    # Copy the null group and fill it
    grp = deepcopy(null_grp)
    
    # univ_map contains all the arrays and positions in those arrays associated
    # with a given universe
    for ndx,Id in enumerate(id_vec):
        for arr_key,grp_indices in id_map[Id].items():
            grp[arr_key][grp_indices] = vec[ndx]
    
    return grp

def grp_to_vec(template,grp,arr_keys,mode="mean"):
    """Function for mapping data stored in data_grou to data_vec. This mapping 
    is based on the user-defined template"""
    
    # Extract useful parameters
    id_vec  = template.id_vec
    id_map  = template.id_map
    # count_vec = template.count_vec
    null_vec  = template.null_vec
    
    # Copy the null vector and fill it
    vec = deepcopy(null_vec)
    count_vec = deepcopy(null_vec)
    
    # univ_map contains all the arrays and positions in those arrays associated
    # with a given universe
    for ndx,Id in enumerate(id_vec):
        for arr_key,grp_indices in id_map[Id].items():
            if (arr_keys is None) or (arr_key in arr_keys):
                vec[ndx] += np.sum(grp[arr_key][grp_indices])
                count_vec[ndx] += np.size(grp[arr_key][grp_indices])
    
    # Idenitfy locations where data was not specified
    zmask = count_vec==0
    nmask = np.logical_not(zmask)
    
    if mode=="mean":
        # Convert sum to average and return the result
        vec[nmask] = vec[nmask]/count_vec[nmask]
    
    vec[zmask] = np.nan
    
    return vec
