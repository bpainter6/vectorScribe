# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 12:59:12 2025

@author: 17066
"""

import numpy as np
from vectorScribe import Scribe

iterables = [ ["0","1","2"], ["rab","nsf","trp","chi"] ]
names = ["state", "xs_type"]

template = {"rvc" : ["a","b","c","d"],
            "arr" : [["a","d"],["b","a"],["a","e"]]}

scr = Scribe(template,"rvc",iterables,names)
scr.df.loc[("0","rab"),"arr"] = np.random.uniform(0,1,[3,2,2])
scr.df.loc[("0","rab"),"settings"] = ("arr",0,float,"mean")
scr.transcribe()