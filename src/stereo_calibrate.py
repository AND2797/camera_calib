import numpy as np
import cv2
import glob
from tqdm import tqdm
# termination criteria
#https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
class camera_calibrate:
    def __init__(self, img_size, dims, drawn_path = None):
        self.dims = dims
        self.img_size = img_size
        self.drawn_path = drawn_path
        
    def calibrate(self, refine = 0):
        # import pdb; pdb.set_trace()
        if (refine > 1 or refine < 0):
            raise ValueError('Refine can only be 1 or 0')
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((self.dims[0]*self.dims[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:self.dims[0], 0:self.dims[1]].T.reshape(-1,2)
        obj_points = []
        img_points = []
        images = glob.glob("*.png")
        
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
        raw = (ret, mtx, dist, rvecs, tvecs)
        
        if (refine):
            new_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, self.img_size)
            refined = (ret, mtx, new_mtx, dist, rvecs, tvecs)
            return refined
        return raw
        
  
    
    
    def stereo_rectify(self, cmtx1, dist1, cmtx2, dist2, R, T):
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cmtx1, dist1, cmtx2, dist2, (1920,1080), R, T)
        return (R1, R2, P1, P2, Q)
    

            
            
        
if __name__ == '__main__':
    camera_model = camera_calibrate(dims = (9,6), img_size = (1920, 1080))
    params = camera_model.calibrate()
    '''
    to do : stereo calibration
    robustness checks
    '''

    
