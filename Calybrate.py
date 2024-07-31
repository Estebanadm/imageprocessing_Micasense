import cv2 #openCV
import exiftool
import os, glob
import numpy as np
import pyzbar.pyzbar as pyzbar
import matplotlib.pyplot as plt
import numpy
import skimage
from skimage.transform import warp,matrix_transform,resize,FundamentalMatrixTransform,estimate_transform,ProjectiveTransform
import mapboxgl
import time
from IPython import get_ipython
from pathlib import Path
from IPython.display import display
from skimage.transform import ProjectiveTransform
import matplotlib.pyplot as plt
import micasense.imageset as imageset
from micasense.image import Image
from micasense.panel import Panel
import micasense.imageutils as imageutils
import micasense.utils as msutils
import micasense.plotutils as plotutils
import micasense.metadata as metadata
import micasense.capture as capture



ipython = get_ipython()
plt.rcParams["figure.facecolor"] = "w"
panelNames = None
imagePath = Path("./data/ALTUM-PT")



def testEnv():
    print()
    print("Testing the required libraries...")
    print()
    print("Successfully imported all required libraries.")

    if os.name == 'nt':
        if os.environ.get('exiftoolpath') is None:
            print("Set the `exiftoolpath` environment variable as described above")
        else:
            if not os.path.isfile(os.environ.get('exiftoolpath')):
                print("The provided exiftoolpath isn't a file, check the settings")

    try:
        with exiftool.ExifTool(os.environ.get('exiftoolpath')) as exift:
            print('Successfully executed exiftool.')
            print()
    except Exception as e:
        print("Exiftool isn't working. Double check that you've followed the instructions above.")
        print("The execption text below may help to find the source of the problem:")
        print()
        print(e)
        import ipdb; ipdb.set_trace()

def testPanelDetection():
    imageName = glob.glob(os.path.join(imagePath,'IMG_0000_1.tif'))[0]
    print('Testing Panel Detection...')
    print()
    img = Image(imageName)
    img.plot_raw(figsize=(8.73,8.73));
    panel = Panel(img)  
    if not panel.panel_detected():
        raise IOError("Panel Not Detected! Check your installation of pyzbar")
    else:
        print('Panel Succesfully Detected!')
        print()
        panel.plot(figsize=(8,8));

def testing():
    print("Welcome to the Calybrate Image Processing Module.")
    testEnv()
    testPanelDetection()
    print("Testing completed.\n")

def multispectralBandHistogram(thecapture, im_aligned, panchroCam, sharpened_stack):
    theColors = {'Blue': 'blue', 'Green': 'green', 'Red': 'red', \
             'Red edge': 'maroon', 'NIR': 'purple', 'Panchro': 'yellow', 'PanchroB': 'orange',\
            'Red edge-740': 'salmon', 'Red Edge': 'maroon', 'Blue-444': 'aqua', \
             'Green-531': 'lime', 'Red-650': 'lightcoral', 'Red edge-705':'brown'}

    eo_count = len(thecapture.eo_indices())
    multispec_min = np.min(np.percentile(im_aligned[:,:,1:eo_count].flatten(),0.01))
    multispec_max = np.max(np.percentile(im_aligned[:,:,1:eo_count].flatten(), 99.99))

    theRange = (multispec_min,multispec_max)

    fig, axis = plt.subplots(1, 1, figsize=(10,4))
    for x,y in zip(thecapture.eo_indices(),thecapture.eo_band_names()):
        axis.hist(im_aligned[:,:,x].ravel(), bins=512, range=theRange, \
                histtype="step", label=y, color=theColors[y], linewidth=1.5)
    plt.title("Multispectral histogram (radiance)")
    axis.legend()
    plt.show()

    if panchroCam:
        eo_count = len(thecapture.eo_indices())
        multispec_min = np.min(np.percentile(sharpened_stack[:,:,1:eo_count].flatten(),0.01))
        multispec_max = np.max(np.percentile(sharpened_stack[:,:,1:eo_count].flatten(), 99.99))

        theRange = (multispec_min,multispec_max)

        fig, axis = plt.subplots(1, 1, figsize=(10,4))
        for x,y in zip(thecapture.eo_indices(),thecapture.eo_band_names()):
            axis.hist(sharpened_stack[:,:,x].ravel(), bins=512, range=theRange, \
                    histtype="step", label=y, color=theColors[y], linewidth=1.5)
        plt.title("Pan-sharpened multispectral histogram (radiance)")
        axis.legend()
        plt.show()


def Calybrate(panelImageName, currImageName, showMultispectralBandHistogram):
    print("Calybration process started.\n")
    if '__IPYTHON__' in globals():
        ipython.magic('load_ext autoreload')
        ipython.magic('autoreload 2')
        ipython.magic('matplotlib inline')
    # these will return lists of image paths as strings 
    panelNames = list(imagePath.glob(panelImageName))
    panelNames = [x.as_posix() for x in panelNames]

    imageNames = list(imagePath.glob(currImageName))
    imageNames = [x.as_posix() for x in imageNames]

    if panelNames is not None:
        panelCap = capture.Capture.from_filelist(panelNames)
    else:
        panelCap = None

    thecapture = capture.Capture.from_filelist(imageNames)

    if len(thecapture.camera_serials) > 1:
        cam_serial = "_".join(thecapture.camera_serials)
        print(cam_serial)
    else:
        cam_serial = thecapture.camera_serial
        
    # print("Camera model:",cam_model)
    # print("Bit depth:", thecapture.bits_per_pixel)
    # print("Camera serial number:", cam_serial)
    # print("Capture ID:",thecapture.uuid)

    # get camera model for future use 
    cam_model = thecapture.camera_model

    # determine if this sensor has a panchromatic band 
    if cam_model == 'RedEdge-P' or cam_model == 'Altum-PT':
        panchroCam = True
    else:
        panchroCam = False
        panSharpen = False 

    if panelCap is not None:
        if panelCap.panel_albedo() is not None:
            panel_reflectance_by_band = panelCap.panel_albedo()
        else:
            panel_reflectance_by_band = [0.49]*len(thecapture.eo_band_names()) #RedEdge band_index order
        panel_irradiance = panelCap.panel_irradiance(panel_reflectance_by_band)  
        irradiance_list = panelCap.panel_irradiance(panel_reflectance_by_band) + [0] # add to account for uncalibrated LWIR band, if applicable
        img_type = "reflectance"
        thecapture.plot_undistorted_reflectance(panel_irradiance)
    else:
        if thecapture.dls_present():
            img_type='reflectance'
            irradiance_list = thecapture.dls_irradiance() + [0]
            thecapture.plot_undistorted_reflectance(thecapture.dls_irradiance())
        else:
            img_type = "radiance"
            thecapture.plot_undistorted_radiance() 
            irradiance_list = None
    if panchroCam:
        warp_matrices_filename = cam_serial + "_warp_matrices_SIFT.npy"
    else:
        warp_matrices_filename = cam_serial + "_warp_matrices_opencv.npy"

    if Path('./' + warp_matrices_filename).is_file():
        print("Found existing warp matrices for camera", cam_serial)
        load_warp_matrices = np.load(warp_matrices_filename, allow_pickle=True)
        loaded_warp_matrices = []
        for matrix in load_warp_matrices: 
            if panchroCam:
                transform = ProjectiveTransform(matrix=matrix.astype('float64'))
                loaded_warp_matrices.append(transform)
            else:
                loaded_warp_matrices.append(matrix.astype('float32'))
        print("Warp matrices successfully loaded.")

        if panchroCam:
            warp_matrices_SIFT = loaded_warp_matrices
        else:
            warp_matrices = loaded_warp_matrices
    else:
        print("No existing warp matrices found. Creating a new one...")
        warp_matrices_SIFT = False
        warp_matrices = False

    if not panchroCam:
        cropped_dimensions, edges = imageutils.find_crop_bounds(thecapture, warp_matrices, warp_mode=warp_mode, reference_band=match_index)
        print("Cropped dimensions:",cropped_dimensions)
        im_aligned = thecapture.create_aligned_capture(warp_matrices=warp_matrices, motion_type=warp_mode, img_type=img_type)

    if panchroCam: 
        st = time.time()
        if not warp_matrices_SIFT :
            print("Generating new warp matrices...")
            warp_matrices_SIFT = thecapture.SIFT_align_capture(min_matches = 10)
            
        sharpened_stack, upsampled = thecapture.radiometric_pan_sharpened_aligned_capture(warp_matrices=warp_matrices_SIFT, irradiance_list=irradiance_list, img_type=img_type)
        
    # we can also use the Rig Relatives from the image metadata to do a quick, rudimentary alignment 
    #     warp_matrices0=thecapture.get_warp_matrices(ref_index=5)
    #     sharpened_stack,upsampled = radiometric_pan_sharpen(thecapture,warp_matrices=warp_matrices0)

        print("Pansharpened shape:", sharpened_stack.shape)
        print("Upsampled shape:", upsampled.shape)
        # re-assign to im_aligned to match rest of code 
        im_aligned = upsampled
        et = time.time()
        elapsed_time = et - st
        print('Alignment and pan-sharpening time:', int(elapsed_time), 'seconds')

    if (showMultispectralBandHistogram):
        multispectralBandHistogram(thecapture, im_aligned, panchroCam, sharpened_stack)
 



def main(): 
    panelImageName = 'IMG_0000_*.tif'
    currImageName = 'IMG_0001_*.tif'
    showMultispectralBandHistogram = True
    # testing()
    Calybrate(panelImageName, currImageName, showMultispectralBandHistogram)
    

    
  
  
# Using the special variable  
# __name__ 
if __name__=="__main__": 
    main() 