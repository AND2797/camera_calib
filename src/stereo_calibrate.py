import numpy as np
import cv2
import glob
from tqdm import tqdm
import os
#https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
class camera_calibrate:
    def __init__(self, img_size, img_path_1 = None, img_path_2 = None, dims= None, drawn_path = None):
        self.img_size = img_size
        self.img_path_1 = img_path_1 #left
        self.img_path_2 = img_path_2 #right
        self.dims = dims
        self.drawn_path = drawn_path
        
    def calib(self, refine = 0):
        # import pdb; pdb.set_trace()
        if (refine > 1 or refine < 0):
            raise ValueError('Refine can only be 1 or 0')
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((self.dims[0]*self.dims[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:self.dims[0], 0:self.dims[1]].T.reshape(-1,2)
        obj_points = []
        img_points = []
        
        #use image path - 1
        if self.img_path_1 == None: #assumes same directory
            images = glob.glob("*.png")
        else:
            images = glob.glob(os.path.join(self.img_path_1,'*.png'))
        
        if images == []:
            raise ValueError('No images found in specified directory.')
        corners_found = 0
        # print(f"{len(images)} images found.\n")
        for fname in tqdm(range(len(images)), position = 0, leave = True, total = len(images)):
            img = cv2.imread(images[fname])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            ret, corners = cv2.findChessboardCorners(gray, (self.dims[0], self.dims[1]), None)
            
            if ret == True:
                obj_points.append(objp)
                
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                img_points.append(corners)
                
                cv2.drawChessboardCorners(img, (self.dims[0], self.dims[1]),
                                               corners2, ret)
                    
                cv2.imshow('img', img)
                cv2.waitKey(500)
                corners_found += 1
        
        cv2.destroyAllWindows()
        print(f"{corners_found} / {len(images)} images calibrated.")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, self.img_size, None, None)
        raw = (img_points, obj_points, ret, mtx, dist, rvecs, tvecs)
        
        if (refine):
            new_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, self.img_size)
            refined = (img_points, obj_points, ret, mtx, new_mtx, dist, rvecs, tvecs)
            return refined
        return raw
        
  
    @classmethod
    def stereo_calib(cls, camera_1, camera_2):
        # import pdb;pdb.set_trace()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1, 1e-5)
        flags = cv2.CALIB_FIX_INTRINSIC
        params1 = camera_1.calib()
        params2 = camera_2.calib()
        objpoints = params1[1]
        imgpoints_1 = params1[0]
        imgpoints_2 = params2[0]
        mtx_1 = params1[3]
        dist_1 = params1[4]
        mtx_2 = params2[3]
        dist_2 = params2[4]
        ret, mtx_1, dist_1, mtx_2, dist_2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_1, 
                                                                            imgpoints_2, mtx_1, dist_1, 
                                                                            mtx_2, dist_2, camera_1.img_size, 
                                                                            criteria = criteria, flags = flags)
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(mtx_1, dist_1, mtx_2, dist_2, camera_1.img_size, R, T)

        return (R, T, E, F, R1, R2, P1, P2, Q)


            
    

     
    

    
