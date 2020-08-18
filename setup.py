from setuptools import setup

def readme():
    with open('README.md') as f:
        README = f.read()
    return README


setup(
    name="camcalib",
    version="1.0.5",
    description="A wrapper around the main functionalities offered by OpenCV for camera calibration for cleaner and maintainable calibration routines.",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/AND2797/camera_calib",
    author="Aditya Narayan Das",
    author_email="aditya.das2797@gmail.com",
    license="MIT",
    packages=["camcalib"],
    include_package_data=True,
    install_requires=["numpy","tqdm","opencv-python","opencv-contrib-python"],
)
