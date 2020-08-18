import numpy as np
import cv2
import glob
from tqdm import tqdm
import os
''' for finer implementation details:  https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html'''
class camera_calibrate:
    def __init__(self, img_size, img_path = None, dims= None, drawn_path = None, common_id = None):
        '''
        img_size: specify image resolution in pixels (width, height)
        img_path_1: specify image path (absolute or relative) for camera 1
        dims: specify dimensions of checkerboard (width, height)
        drawn_path: path to save images
        common_id: common suffix for saved images
        '''
        self.img_size = img_size
        self.img_path = img_path
        self.dims = dims
        self.drawn_path = drawn_path
        self.common_id = common_id
        
    def calib(self, display = 1, criteria = None):
        '''
        display: 1(default, show calibrated images during exec), 0(don't show images)
        
        for more details: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
        returns: 
        ret: Overall RMS Re-projection error.
        img_points: corresponding corner coordinates in the images
        obj_points: calibration pattern coordinate space
        mtx: camera matrix
        dist: distortion coefficients
        rvecs: output vector of rotation vectors
        tvecs: output vector of translation vectors
        '''
        
        if (display > 1 or display < 0):
            raise ValueError('display can only be 1 or 0')
        
        if criteria == None:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((self.dims[0]*self.dims[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:self.dims[0], 0:self.dims[1]].T.reshape(-1,2)
        obj_points = []
        img_points = []
        
        #use image path - 1
        if self.img_path == None: #assumes same directory
            images = glob.glob("*.png")
        else:
            images = glob.glob(os.path.join(self.img_path,'*.png'))
        
        if images == []:
            raise ValueError('No images found in specified directory.')
        corners_found = 0
        # print(f"{len(images)} images found.\n")
        for fname in tqdm(range(len(images)), position = 0, leave = True, total = len(images)):
            im_name = images[fname]
            img = cv2.imread(im_name)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            ret, corners = cv2.findChessboardCorners(gray, (self.dims[0], self.dims[1]), None)
            
            if ret == True:
                obj_points.append(objp)
                
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                img_points.append(corners)
                
                cv2.drawChessboardCorners(img, (self.dims[0], self.dims[1]),
                                               corners2, ret)
                if (display):    
                    cv2.imshow('img', img)
                if self.drawn_path != None:
                    # import pdb; pdb.set_trace()
                    cv2.imwrite(self.drawn_path + f'/detected_{fname}_{self.common_id}.png', img)
                cv2.waitKey(500)
                corners_found += 1
        
        cv2.destroyAllWindows()
        print(f"{corners_found} / {len(images)} images calibrated.")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, self.img_size, None, None)
        
        return {'img_points':img_points, 'obj_points':obj_points, 'mtx':mtx, 'dist':dist, 'rvecs':rvecs, 'tvecs':tvecs}
        
  
    @classmethod
    def stereo_calib(cls, camera_1, camera_2, criteria = None, flags = None, alpha = -1, newImageSize = None):
        '''
        camera_1: object of type camera_calibrate
        camera_2: object of type camera_calibrate
        criteria: Termination criteria for the iterative optimization algorithm.
        
        for alpha, newImageSize, flags, view: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
        
        returns:
        R: Rotation matrix camera_2 w.r.t camera_1
        T: Translation vector camera_2 w.r.t camera_1
        E: Essential matrix
        F: Fundamental matrix
        R1: Output 3x3 rectification transform (rotation matrix) for the first camera. 
        R2: Output 3x3 rectification transform (rotation matrix) for the second camera.
        P1: Output 3x4 projection matrix in the new (rectified) coordinate systems for the first camera.
        P2: Output 3x4 projection matrix in the new (rectified) coordinate systems for the second camera.
        Q: Output 4 \times 4 disparity-to-depth mapping matrix.
        '''
        
        if camera_1.img_size != camera_2.img_size:
            raise ValueError(f'camera_1 image size {camera_1.img_size} does not match camera_2 image size {camera_2.img_size}')
        if criteria == None:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1, 1e-5)
        if flags == None:
            flags = cv2.CALIB_FIX_INTRINSIC
        params1 = camera_1.calib()
        params2 = camera_2.calib()
        objpoints = params1['obj_points']
        imgpoints_1 = params1['img_points']
        imgpoints_2 = params2['img_points']
        mtx_1 = params1['mtx']
        dist_1 = params1['dist']
        mtx_2 = params2['mtx']
        dist_2 = params2['dist']
        ret, mtx_1, dist_1, mtx_2, dist_2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_1, 
                                                                            imgpoints_2, mtx_1, dist_1, 
                                                                            mtx_2, dist_2, camera_1.img_size, 
                                                                            criteria = criteria, flags = flags)
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(mtx_1, dist_1, mtx_2, dist_2, camera_1.img_size, R, T, alpha = alpha, newImageSize = newImageSize)

        return {'R':R, 'T':T, 'E':E, 'F':E, 'R1':R1, 'R2':R2, 'P1':P1, 'P2':P2, 
                'Q':Q, 'roi1': validPixROI1, 'roi2': validPixROI2, 'ret':ret, 'mtx_1':mtx_1, 'dist_1':dist_1, 'mtx_2':mtx_2, 'dist_2':dist_2}


            
    

     
    

    
