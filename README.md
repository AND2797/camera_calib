# Camera Calibration w/ Python + OpenCV
A wrapper around the main functionalities offered by OpenCV for camera calibration for cleaner and maintainable calibration routines.

## Dependencies
```
numpy (1.17.4 preferred)
opencv (3.4.2 preferred)
tqdm
```
## Installation 
```
pip install camcalib
```

## Instructions
### Import 
```
from cam_calib import camera_calibrate
```
### Single Camera Calibration

Instantiate an object of type `camera_calibrate` by passing in relevant arguments to the constructor. (Example below uses some place holder arguments)

```
camera_1 = camera_calibrate(img_path = './path', dims = (w, h), img_size = (w_i, h_i),...)
```

Use the `calib` method on the object for single camera calibration

```
params = camera_1.calib()
```
### Stereo Camera Calibration
Instantiate two objects of the type `camera_calibrate` by passing in relevant arguments to the constructor. (Example below uses some place holder arguments)

```
camera_1 = camera_calibrate(img_path = '../left_path', dims = (w, h), img_size = (w_i, h_i),...)
camera_2 = camera_calibrate(img_path = '../right_path', dims = (w, h), img_size = (w_i, h_i),...)
```
Call the class method `stereo_calib` method on the class `camera_calibrate` by passing the two objects as arguments. 

```
stereo_params = camera_calibrate.stereo_calib(camera_1, camera_2) 
```

### TO DO:
- [ ] Write tests
- [ ] Update docs with detailed information on inputs, return values etc.
- [ ] Misc. checks
