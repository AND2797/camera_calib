# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 14:55:33 2020

@author: Aditya
"""

from stereo_calibrate import camera_calibrate as cc
camera_model1 = cc(img_path_1 = '../left', dims = (9,6), img_size = (1920, 1080), drawn_path = '../save')
camera_model2 = cc(img_path_1 = '../right', dims = (9,6), img_size = (1920, 1080))
stereo_params = cc.stereo_calib(camera_model1, camera_model2)