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
import matplotlib.cm as cm
from PIL import ImageEnhance
from PIL import Image as pilImage
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
figsize=(30,23) # use this size for full-image-resolution display
# figsize=(16,13)   # use this size for export-sized display

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
    plt.savefig('Results/Calibration/Multispectral_histogram.png')

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
        plt.savefig('Results/Calibration/pan_sharpened_Multispectral_histogram.png')

def saveVisualizationOfAlignetImages(thecapture, im_aligned, panchroCam, sharpened_stack):
    

    rgb_band_indices = [thecapture.band_names_lower().index('red'),
                        thecapture.band_names_lower().index('green'),
                        thecapture.band_names_lower().index('blue')]
    cir_band_indices = [thecapture.band_names_lower().index('nir'),
                        thecapture.band_names_lower().index('red'),
                        thecapture.band_names_lower().index('green')]

    # Create normalized stacks for viewing
    im_display = np.zeros((im_aligned.shape[0],im_aligned.shape[1],im_aligned.shape[2]), dtype=np.float32)
    im_min = np.percentile(im_aligned[:,:,rgb_band_indices].flatten(), 0.5)  # modify these percentiles to adjust contrast
    im_max = np.percentile(im_aligned[:,:,rgb_band_indices].flatten(), 99.5)  # for many images, 0.5 and 99.5 are good values

    if panchroCam:
        im_display_sharp = np.zeros((sharpened_stack.shape[0],sharpened_stack.shape[1],sharpened_stack.shape[2]), dtype=np.float32 )
        im_min_sharp = np.percentile(sharpened_stack[:,:,rgb_band_indices].flatten(), 0.5)  # modify these percentiles to adjust contrast
        im_max_sharp = np.percentile(sharpened_stack[:,:,rgb_band_indices].flatten(), 99.5)  # for many images, 0.5 and 99.5 are good values


    # for rgb true color, we use the same min and max scaling across the 3 bands to 
    # maintain the "white balance" of the calibrated image
    for i in rgb_band_indices:
        im_display[:,:,i] =  imageutils.normalize(im_aligned[:,:,i], im_min, im_max)
        if panchroCam: 
            im_display_sharp[:,:,i] = imageutils.normalize(sharpened_stack[:,:,i], im_min_sharp, im_max_sharp)

    rgb = im_display[:,:,rgb_band_indices]

    if panchroCam:
        rgb_sharp = im_display_sharp[:,:,rgb_band_indices]

    nir_band = thecapture.band_names_lower().index('nir')
    red_band = thecapture.band_names_lower().index('red')

    ndvi = (im_aligned[:,:,nir_band] - im_aligned[:,:,red_band]) / (im_aligned[:,:,nir_band] + im_aligned[:,:,red_band])

    # for cir false color imagery, we normalize the NIR,R,G bands within themselves, which provides
    # the classical CIR rendering where plants are red and soil takes on a blue tint
    for i in cir_band_indices:
        im_display[:,:,i] =  imageutils.normalize(im_aligned[:,:,i])

    cir = im_display[:,:,cir_band_indices]
    if panchroCam:
        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)
    ax1.set_title("Red-Green-Blue Composite")
    ax1.imshow(rgb)
    if panchroCam:
        ax2.set_title("Red-Green-Blue Composite (pan-sharpened)")
        ax2.imshow(rgb_sharp)

    fig, (ax3,ax4) = plt.subplots(1, 2, figsize=figsize)
    ax3.set_title("NDVI")
    ax3.imshow(ndvi)
    ax4.set_title("Color Infrared (CIR) Composite")
    ax4.imshow(cir)
    plt.savefig('Results/AlignedVisualization/AlignedImages.png')

    return rgb_sharp

def imageEnhancement(thecapture, panchroCam, rgb_sharp):
    if panchroCam:
        rgb = rgb_sharp
    # Create an enhanced version of the RGB render using an unsharp mask
    gaussian_rgb = cv2.GaussianBlur(rgb, (9,9), 10.0)
    gaussian_rgb[gaussian_rgb<0] = 0
    gaussian_rgb[gaussian_rgb>1] = 1
    unsharp_rgb = cv2.addWeighted(rgb, 1.5, gaussian_rgb, -0.5, 0)
    unsharp_rgb[unsharp_rgb<0] = 0
    unsharp_rgb[unsharp_rgb>1] = 1

    # Apply a gamma correction to make the render appear closer to what our eyes would see
    gamma = 1.4
    gamma_corr_rgb = unsharp_rgb**(1.0/gamma)
    fig = plt.figure(figsize=figsize)
    cv2.imwrite('Results/EnhancedImages/'+thecapture.uuid+'-enhanced.png', cv2.cvtColor(gamma_corr_rgb*255, cv2.COLOR_RGB2BGR))
    # plt.imshow(gamma_corr_rgb, aspect='equal')
    # plt.axis('off')
    # plt.savefig('Results/EnhancedImages/'+thecapture.uuid+'-enhanced2.png')
    return gamma_corr_rgb

def stackExport(thecapture, panchroCam):
    #set output name to unique capture ID, e.g. FWoNSvgDNBX63Xv378qs
    outputName = thecapture.uuid

    st = time.time()
    if panchroCam:
        # in this example, we can export both a pan-sharpened stack and an upsampled stack
        # so you can compare them in GIS. In practice, you would typically only output the pansharpened stack 
        print(outputName+"-pansharpened.tif")
        thecapture.save_capture_as_stack("Results/Pan Sharpened Stacks/"+outputName+"-pansharpened.tif", sort_by_wavelength=True, pansharpen=True)
        thecapture.save_capture_as_stack("Results/Pan Sharpened Stacks/"+outputName+"-upsampled.tif", sort_by_wavelength=True, pansharpen=False)
    else:
        thecapture.save_capture_as_stack(outputName+"-noPanels.tif", sort_by_wavelength=True)

    et = time.time()
    elapsed_time = et - st
    print("Time to save stacks:", int(elapsed_time), "seconds.")

def save_array_as_png_pil(array, output_path):
    # Normalize the array to the range [0, 255]
    array_normalized = np.clip(array, 0, 1)  # Ensure values are between 0 and 1
    array_normalized = (array_normalized * 255).astype(np.uint8)  # Scale to 0-255

    # Convert to image
    image = pilImage.fromarray(array_normalized)
    image.save(output_path)
    
def save_overlay_as_png(imgbase_path, imgcolor_path, output_filename, alpha=1.0, colormap='viridis'):
    """ Overlay an image with another image using a colormap and save as PNG """
    
    # Open the base image
    imgbase = pilImage.open(imgbase_path).convert("RGBA")
    
    # Open the overlay image and convert it to grayscale
    imgcolor = pilImage.open(imgcolor_path).convert("L")
    
    # Resize overlay to match base image if necessary
    if imgbase.size != imgcolor.size:
        imgcolor = imgcolor.resize(imgbase.size, pilImage.ANTIALIAS)

    # Convert the grayscale image to a numpy array
    imgcolor_np = np.array(imgcolor)
    
    # Normalize the image to the range [0, 1]
    imgcolor_np = imgcolor_np / 255.0
    
    # Apply the colormap
    cmap = cm.get_cmap(colormap)
    imgcolor_colormap = cmap(imgcolor_np)
    
    # Convert the colormap image to RGBA format and then to a PIL image
    imgcolor_colormap = (imgcolor_colormap[:, :, :3] * 255).astype(np.uint8)
    imgcolor_colormap = pilImage.fromarray(imgcolor_colormap, mode="RGB").convert("RGBA")
    
    # Adjust the alpha of the overlay image
    alpha_channel = ImageEnhance.Brightness(pilImage.fromarray((imgcolor_np * 255).astype(np.uint8), mode="L")).enhance(alpha)
    imgcolor_colormap.putalpha(alpha_channel)
    
    # Overlay the images
    combined = pilImage.alpha_composite(imgbase, imgcolor_colormap)
    
    # Save the combined image
    combined.save(output_filename, format='PNG')

def NDVIEnhancement(thecapture, im_aligned , panchroCam, sharpened_stack,img_type,gamma_corr_rgb):
    figsize=(30,23)
    nir_band = thecapture.band_names_lower().index('nir')
    red_band = thecapture.band_names_lower().index('red')

    thelayer = im_aligned
    if panchroCam:
        thelayer = sharpened_stack
    np.seterr(divide='ignore', invalid='ignore') # ignore divide by zero errors in the index calculation

    # Compute Normalized Difference Vegetation Index (NDVI) from the NIR(3) and RED (2) bands
    ndvi = (thelayer[:,:,nir_band] - thelayer[:,:,red_band]) / (thelayer[:,:,nir_band] + thelayer[:,:,red_band])
    print("Image type:",img_type)

    # remove shadowed areas (mask pixels with NIR reflectance < 20%))
    # this does not seem to work on panchro stacks 
    if img_type == 'reflectance':
        ndvi = np.ma.masked_where(thelayer[:,:,nir_band] < 0.20, ndvi) 
    elif img_type == 'radiance':
        lower_pct_radiance = np.percentile(thelayer[:,:,nir_band],  10.0)
        ndvi = np.ma.masked_where(thelayer[:,:,nir_band] < lower_pct_radiance, ndvi) 

    min_display_ndvi = 0.45 # further mask soil by removing low-ndvi values
    #min_display_ndvi = np.percentile(ndvi.flatten(),  5.0)  # modify with these percentilse to adjust contrast
    max_display_ndvi = np.percentile(ndvi.flatten(), 99.5)  # for many images, 0.5 and 99.5 are good values
    masked_ndvi = np.ma.masked_where(ndvi < min_display_ndvi, ndvi)

    # #reduce the figure size to account for colorbar
    figsize=np.asarray(figsize) - np.array([3,2])

    #plot NDVI over an RGB basemap, with a colorbar showing the NDVI scale
    fig, axis = plotutils.plot_overlay_withcolorbar(gamma_corr_rgb, 
                                        masked_ndvi, 
                                        figsize = (14,7), 
                                        title = 'NDVI filtered to only plants over RGB base layer',
                                        vmin = min_display_ndvi,
                                        vmax = max_display_ndvi,
                                        show=False)
    fig.savefig("Results/Indexes/NDVI/Fig/"+thecapture.uuid+'_ndvi_over_rgb.png')
    save_array_as_png_pil(masked_ndvi, "Results/Indexes/NDVI/Mask/"+ thecapture.uuid+'_NDVIMask.png')
    cv2.imwrite('Results/Indexes/NDVI/Img/'+thecapture.uuid+'-enhanced.png', cv2.cvtColor(gamma_corr_rgb*255, cv2.COLOR_RGB2BGR))
    save_overlay_as_png('Results/EnhancedImages/'+thecapture.uuid+'-enhanced.png',
                        'Results/Indexes/NDVI/Mask/'+ thecapture.uuid+'_NDVIMask.png',
                        'Results/Indexes/NDVI/Overlay/'+thecapture.uuid+'_NDVIOverlay.png')
    



def NDREComputation(thecapture,gamma_corr_rgb,thelayer,nir_band,rededge_band, min_display_ndvi,ndvi):
    # Compute Normalized Difference Red Edge Index from the NIR(3) and RedEdge(4) bands
    rededge_band = thecapture.band_names_lower().index('red edge')
    ndre = (thelayer[:,:,nir_band] - thelayer[:,:,rededge_band]) / (thelayer[:,:,nir_band] + thelayer[:,:,rededge_band])

    # Mask areas with shadows and low NDVI to remove soil
    masked_ndre = np.ma.masked_where(ndvi < min_display_ndvi, ndre)

    # Compute a histogram
    ndre_hist_min = np.min(np.percentile(masked_ndre,0.5))
    ndre_hist_max = np.max(np.percentile(masked_ndre,99.5))
    fig, axis = plt.subplots(1, 1, figsize=(10,4))
    axis.hist(masked_ndre.ravel(), bins=512, range=(ndre_hist_min, ndre_hist_max))
    plt.title("NDRE Histogram (filtered to only plants)")
    plt.show()

    min_display_ndre = np.percentile(masked_ndre, 5)
    max_display_ndre = np.percentile(masked_ndre, 99.5)

    BasePath = "Results/Indexes/NDVI/"

    fig, axis = plotutils.plot_overlay_withcolorbar(gamma_corr_rgb, 
                                        masked_ndre, 
                                        figsize=(14,7), 
                                        title='NDRE filtered to only plants over RGB base layer',
                                        vmin=min_display_ndre,vmax=max_display_ndre)
    fig.savefig("Results/Indexes/NDVI/"+thecapture.uuid+'_ndre_over_rgb.png')

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
        # thecapture.plot_undistorted_reflectance(panel_irradiance)
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
    
    rgb=saveVisualizationOfAlignetImages(thecapture, im_aligned, panchroCam, sharpened_stack)
    gama_corr_rgb = imageEnhancement(thecapture, panchroCam, rgb)
    stackExport(thecapture, panchroCam)
    NDVIEnhancement(thecapture, im_aligned , panchroCam, sharpened_stack,img_type,gama_corr_rgb )






def main(): 
    panelImageName = 'IMG_0000_*.tif'
    currImageName = 'IMG_0001_*.tif'
    showMultispectralBandHistogram = True
    # testing()
    Calybrate(panelImageName, currImageName, showMultispectralBandHistogram)

if __name__=="__main__": 
    main() 