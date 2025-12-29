
import os
import csv
import cv2
import math
import glob
import numpy as np
import scipy.ndimage
import nibabel as nib
import pydicom as pdcm
import lu_ca_Settings as set
import SimpleITK as sitk
from random import shuffle
from skimage import measure
from skimage import morphology
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)  # z, y, x axis load
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return img_rotation(numpyImage, 2), numpyOrigin, numpySpacing

def load_DCM_image(CT_path):

    for name in set.CT_dir_name:
        # print('utils_def/load_DCM_image/',
        #       len(sorted(glob.glob(set.LC_img_dir_path + '/' + CT_path + '/' + name + '/*.dcm'))))
        if len(sorted(glob.glob(set.LC_img_dir_path + '/' + CT_path + '/' + name + '/*.dcm'))) > 0:
            CT_list = sorted(glob.glob(set.LC_img_dir_path + '/' + CT_path + '/' + name + '/*.dcm'))
    gdcmFlag = pdcm.read_file(CT_list[0])

    # CT_thick = float(gdcmFlag.get("SliceThickness"))
    # CT_spacing = gdcmFlag.get("PixelSpacing")
    # CT_spacing = [CT_thick, float(CT_spacing[0]), float(CT_spacing[1])]

    if gdcmFlag.file_meta.TransferSyntaxUID == '1.2.840.10008.1.2.1':

        CT_datas = [pdcm.read_file(ct) for ct in CT_list]
        norm_scan = get_pixels_hu(CT_datas, "CT")

    elif gdcmFlag.file_meta.TransferSyntaxUID == '1.2.840.10008.1.2.4.90':
        try:
            import pydicom.pixel_data_handlers.gdcm_handler as gdcm_handler
        except:
            import dicom.pixel_data_handlers.gdcm_handler as gdcm_handler
        pdcm.config.image_handlers = [gdcm_handler, ]

        CT_datas = [pdcm.read_file(ct) for ct in CT_list]
        norm_scan = get_pixels_hu(CT_datas, "CT")

        slide = [(norm_scan.shape[0] - 1) - i for i in range(norm_scan.shape[0])]
        norm_scan = norm_scan[slide]

    if norm_scan.shape[0] == 512:
        print(norm_scan.shape)
        return norm_scan.transpose(2, 0, 1)

    return norm_scan

def read_NIFTY_image(Nifty_ID, Nifty_shape):
    lbl_path = sorted(glob.glob(set.lbl_dir_path + "/" + Nifty_ID + "_*.gz"))
    label = np.zeros(Nifty_shape)
    for lbl_id in lbl_path:
        lbl_part = nib.load(lbl_id)
        lbl = lbl_part.get_fdata()

        lbl = lbl.transpose(2, 1, 0)

        if lbl.shape != Nifty_shape:
            resizing_factor = np.asarray(list(Nifty_shape)) / np.asarray(list(lbl.shape))
            lbl = scipy.ndimage.interpolation.zoom(lbl, resizing_factor)
            lbl[lbl < 0.5] = 0
            lbl[lbl >= 0.5] = 1

        label += lbl

    return label, lbl_part.header.copy()

def save_NIFTY_image(Nifty_ID, NIFTY, HEADER):
    label = NIFTY.transpose(2, 1, 0)
    img = nib.nifti1.Nifti1Image(label.astype(np.uint16), None, HEADER)
    img.set_data_dtype(dtype=np.uint16)
    nib.save(img, set.save_NFT_dir_path + '/' + Nifty_ID + '_lung_lesion.nii.gz')
    print(set.save_NFT_dir_path + '/' + Nifty_ID + '_lung_lesion.nii.gz')

def make_label_mask(CT_list, Nifty_list, CT_ID):
    # attr = ["PatientID", "Rows", "Columns", "SliceThickness", "PixelSpacing"]

    CT_datas = [pdcm.read_file(ct) for ct in CT_list]

    # CT_ID = CT_datas[0].get("PatientID")
    # CT_row = int(CT_datas[0].get("Rows"))
    # CT_colum = int(CT_datas[0].get("Columns"))
    CT_thick = float(CT_datas[0].get("SliceThickness"))
    CT_spacing = CT_datas[0].get("PixelSpacing")
    CT_spacing = [CT_thick, float(CT_spacing[0]), float(CT_spacing[1])]

    norm_scan = get_pixels_hu(CT_datas, "CT")
    resize_scan, ct_spacing = CT_resample(norm_scan, modality="CT", spacing=CT_spacing)

    # LungMask = make_lungmask(resize_scan, False)

    CT_shape = np.array(resize_scan.shape)

    labels = np.memmap(filename=set.save_NFT_dir_path + '/' + CT_ID + '.lbl', mode='w+',
                       dtype=np.uint8, shape=resize_scan.shape)
    patient_labels = []
    for lbl_path in Nifty_list:
        if CT_ID in lbl_path:
            patient_labels.append(lbl_path)
        else:
            pass

    if len([patient_labels]) is 1:
        label = nib.load(patient_labels[0])

        lbl = label.get_fdata()
        lbl = np.swapaxes(lbl, 0, -1)
        lbl = np.swapaxes(lbl, 1, -1)

        resizing_factor = CT_shape / lbl.shape

        label = scipy.ndimage.interpolation.zoom(lbl, resizing_factor)

        label[label > 0.5] = 1
        label[label != 1] = 0


        # labels[:, :, :] = LungMask + label
        labels[:, :, :] = label.copy()

    else:
        label = np.zeros(CT_shape)

        for path in patient_labels:
            lbl_part = nib.load(path)

            lbl = lbl_part.get_fdata()
            lbl = np.swapaxes(lbl, 0, -1)
            lbl_part = np.swapaxes(lbl, 1, -1)
            resizing_factor = CT_shape / lbl_part.shape

            lbl = scipy.ndimage.interpolation.zoom(lbl_part, resizing_factor)
            lbl[lbl > 0.5] = 1
            lbl[lbl != 1] = 0

            label += lbl

        label[label > 1] = 1
        # labels[:, :, :] = LungMask + label
        labels[:, :, :] = label.copy()

def make_train_label(CT_list, Nifty_list, CT_ID):
    CT_datas = [pdcm.read_file(ct) for ct in CT_list]

    norm_scan = get_pixels_hu(CT_datas, "CT")

    labels = np.memmap(filename=set.save_NFT_dir_path + '/' + CT_ID + '.lbl', mode='w+',
                       dtype=np.uint8, shape=resize_scan.shape)
    patient_labels = []
    for lbl_path in Nifty_list:
        if CT_ID in lbl_path:
            patient_labels.append(lbl_path)
        else:
            pass

    if len([patient_labels]) is 1:
        label = nib.load(patient_labels[0])

        lbl = label.get_fdata()
        lbl = np.swapaxes(lbl, 0, -1)
        lbl = np.swapaxes(lbl, 1, -1)

        resizing_factor = CT_shape / lbl.shape

        label = scipy.ndimage.interpolation.zoom(lbl, resizing_factor)

        label[label > 0.5] = 1
        label[label != 1] = 0


        # labels[:, :, :] = LungMask + label
        labels[:, :, :] = label.copy()

    else:
        label = np.zeros(CT_shape)

        for path in patient_labels:
            lbl_part = nib.load(path)

            lbl = lbl_part.get_fdata()
            lbl = np.swapaxes(lbl, 0, -1)
            lbl_part = np.swapaxes(lbl, 1, -1)
            resizing_factor = CT_shape / lbl_part.shape

            lbl = scipy.ndimage.interpolation.zoom(lbl_part, resizing_factor)
            lbl[lbl > 0.5] = 1
            lbl[lbl != 1] = 0

            label += lbl

        label[label > 1] = 1
        # labels[:, :, :] = LungMask + label
        labels[:, :, :] = label.copy()

def make_train_data(CT_list, Nifty_list, CT_ID):
    CT_datas = [pdcm.read_file(ct) for ct in CT_list]

    CT_thick = float(CT_datas[0].get("SliceThickness"))
    CT_spacing = CT_datas[0].get("PixelSpacing")
    CT_spacing = [CT_thick, float(CT_spacing[0]), float(CT_spacing[1])]

    norm_scan = get_pixels_hu(CT_datas, "CT")
    resize_scan, ct_spacing = CT_resample(norm_scan, modality="CT", spacing=CT_spacing)

    # LungMask = make_lungmask(resize_scan, False)

    CT_shape = np.array(resize_scan.shape)

    labels = np.memmap(filename=set.save_NFT_dir_path + '/' + CT_ID + '.lbl', mode='w+',
                       dtype=np.uint8, shape=resize_scan.shape)
    patient_labels = []
    for lbl_path in Nifty_list:
        if CT_ID in lbl_path:
            patient_labels.append(lbl_path)
        else:
            pass

    if len([patient_labels]) is 1:
        label = nib.load(patient_labels[0])

        lbl = label.get_fdata()
        lbl = np.swapaxes(lbl, 0, -1)
        lbl = np.swapaxes(lbl, 1, -1)

        resizing_factor = CT_shape / lbl.shape

        label = scipy.ndimage.interpolation.zoom(lbl, resizing_factor)

        label[label > 0.5] = 1
        label[label != 1] = 0


        # labels[:, :, :] = LungMask + label
        labels[:, :, :] = label.copy()

    else:
        label = np.zeros(CT_shape)

        for path in patient_labels:
            lbl_part = nib.load(path)

            lbl = lbl_part.get_fdata()
            lbl = np.swapaxes(lbl, 0, -1)
            lbl_part = np.swapaxes(lbl, 1, -1)
            resizing_factor = CT_shape / lbl_part.shape

            lbl = scipy.ndimage.interpolation.zoom(lbl_part, resizing_factor)
            lbl[lbl > 0.5] = 1
            lbl[lbl != 1] = 0

            label += lbl

        label[label > 1] = 1
        # labels[:, :, :] = LungMask + label
        labels[:, :, :] = label.copy()

def make_label_mask_gdcm(CT_list, Nifty_list, CT_ID):
    # attr = ["PatientID", "Rows", "Columns", "SliceThickness", "PixelSpacing"]
    try:
        import pydicom.pixel_data_handlers.gdcm_handler as gdcm_handler
    except:
        import dicom.pixel_data_handlers.gdcm_handler as gdcm_handler
    
    pdcm.config.image_handlers = [gdcm_handler, ]
    CT_datas = [pdcm.read_file(ct) for ct in CT_list]

    # CT_ID = CT_datas[0].get("PatientID")
    # CT_row = int(CT_datas[0].get("Rows"))
    # CT_colum = int(CT_datas[0].get("Columns"))
    CT_thick = float(CT_datas[0].get("SliceThickness"))
    CT_spacing = CT_datas[0].get("PixelSpacing")
    CT_spacing = [CT_thick, float(CT_spacing[0]), float(CT_spacing[1])]

    norm_scan = get_pixels_hu(CT_datas, "CT")
    resize_scan, ct_spacing = CT_resample(norm_scan, modality="CT", spacing=CT_spacing)

    # LungMask = make_lungmask(resize_scan, False)

    CT_shape = np.array(resize_scan.shape)

    labels = np.memmap(filename=set.save_NFT_dir_path + '/' + CT_ID + '.lbl', mode='w+',
                       dtype=np.uint8, shape=resize_scan.shape)
    patient_labels = []
    for lbl_path in Nifty_list:
        if CT_ID in lbl_path:
            patient_labels.append(lbl_path)
        else:
            pass

    if len([patient_labels]) is 1:
        label = nib.load(patient_labels[0])

        lbl = label.get_fdata()
        lbl = np.swapaxes(lbl, 0, -1)
        lbl = np.swapaxes(lbl, 1, -1)

        resizing_factor = CT_shape / lbl.shape

        label = scipy.ndimage.interpolation.zoom(lbl, resizing_factor)

        label[label > 0.5] = 1
        label[label != 1] = 0


        # labels[:, :, :] = LungMask + label
        labels[:, :, :] = label.copy()

    else:
        label = np.zeros(CT_shape)

        for path in patient_labels:
            lbl_part = nib.load(path)

            lbl = lbl_part.get_fdata()
            lbl = np.swapaxes(lbl, 0, -1)
            lbl_part = np.swapaxes(lbl, 1, -1)
            resizing_factor = CT_shape / lbl_part.shape

            lbl = scipy.ndimage.interpolation.zoom(lbl_part, resizing_factor)
            lbl[lbl > 0.5] = 1
            lbl[lbl != 1] = 0

            label += lbl

        label[label > 1] = 1
        # labels[:, :, :] = LungMask + label
        labels[:, :, :] = label.copy()

def Display_overlap(CT, PT):
    PT_dis = PT * 255
    CT_dis = CT * 255

    PT_map = cv2.applyColorMap(PT_dis.astype(np.uint8), cv2.COLORMAP_JET)
    CT_map = cv2.cvtColor(CT_dis.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    overlay_map = cv2.addWeighted(PT_map, 0.9, CT_map, 0.5, 0)

    fig, ax = plt.subplots(2, 2)

    ax[0, 0].imshow(CT_dis, cmap="gray")
    ax[0, 1].imshow(PT_dis)
    ax[1, 1].imshow(overlay_map)

    ax[0, 0].axis('off')
    ax[0, 1].axis('off')
    ax[1, 0].axis('off')
    ax[1, 1].axis('off')

    plt.show()

    plt.close()

def Display_PT_CT_overlap(CT, PT):
    PT_dis = PT * 255
    CT_dis = CT * 255

    PT_map = cv2.applyColorMap(PT_dis.astype(np.uint8), cv2.COLORMAP_JET)
    CT_map = cv2.cvtColor(CT_dis.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    overlay_map = cv2.addWeighted(PT_map, 0.9, CT_map, 0.5, 0)

    fig, ax = plt.subplots(2, 2)

    ax[0, 0].imshow(CT_dis, cmap="gray")
    ax[0, 1].imshow(PT_dis)
    ax[1, 1].imshow(overlay_map)

    ax[0, 0].axis('off')
    ax[0, 1].axis('off')
    ax[1, 0].axis('off')
    ax[1, 1].axis('off')

    plt.show()

    plt.close()

def get_pixels_hu(scans, modality):
    image = np.stack([s.pixel_array for s in scans])
    if modality is "CT":
        # Convert to int16 (from sometimes int16),
        # should be possible as values should always be low enough (<32k)
        image = image.astype(np.int16)
        # Set outside-of-scan pixels to 0
        # The intercept is usually -1024, so air is approximately 0
        image[image == -2000] = 0

        # Convert to Hounsfield units (HU)
        # intercept = scans[0].get("RescaleIntercept")
        # # if intercept != -1024:
        # #     intercept = -1024
        # slope = scans[0].get("RescaleSlope")
        # print("intercept:", intercept, "slope", slope)
        # if slope != 1:
        #     image = slope * image.astype(np.float64)
        #     image = image.astype(np.int16)
        #
        # image += np.int16(intercept)
        #
        # return normalizePlanes(np.array(image, dtype=np.int16))

        return minmaxNorm(np.array(image, dtype=np.int16))

    elif modality is "PT":
        return minmaxNorm(np.array(image, dtype=np.int16))

def normalizePlanes(npzarray):
    maxHU = 400.
    minHU = -1000.
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray > 1] = 1.
    npzarray[npzarray < 0] = 0.
    return npzarray

def img_rotation(img, k):
    rot_img = np.rot90(img, k, axes=(0, 2))
    return rot_img

def CT_resample(image, modality, spacing):
    # Determine current pixel spacing
    spacing = np.array(list(spacing))
    new_spacing = spacing.copy()
    new_spacing[0] = 3.5

    if spacing[0] is new_spacing[0]:
        pass
    else:
        resize_factor = spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = spacing / real_resize_factor

        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    return image, new_spacing

def PT_resample(image, spacing, new_spacing, shape):
    import scipy.ndimage
    # Determine current pixel spacing
    resize_factor = np.array(list(spacing)) / np.array(list(new_spacing))

    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    new_shape = np.array(list(shape).copy())
    real_resize_factor = new_shape / np.array(image.shape)
    new_spacing /= real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing

def minmaxNorm(nparray):
    n = nparray.min()
    m = nparray.max()

    return ((nparray - n) / (m - n))

def make_lungmask(img, display=False):
    """
    # Standardize the pixel value by subtracting the mean and dividing by the standard deviation
    # Identify the proper threshold by creating 2 KMeans clusters comparing centered on soft tissue/bone vs lung/air.
    # Using Erosion and Dilation which has the net effect of removing tiny features like pulmonary vessels or noise
    # Identify each distinct region as separate image labels (think the magic wand in Photoshop)
    # Using bounding boxes for each image label to identify which ones represent lung and which ones represent “every thing else”
    # Create the masks for lung fields.
    # Apply mask onto the original image to erase voxels outside of the lung fields.
    """

    slice_size, row_size, col_size = img.shape

    mean = np.mean(img)
    std = np.std(img)
    img = img - mean
    img = img / std

    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = img[int(slice_size / 5):int(slice_size / 5 * 4), int(col_size / 5):int(col_size / 5 * 4), int(row_size / 5):int(row_size / 5 * 4)]
    mean = np.mean(middle)
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, I’m moving the
    # underflow and overflow on the pixel spectrum
    img[img >= max] = mean
    img[img <= min] = mean

    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img < threshold, 1.0, 0.0)  # threshold the image

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.
    # We don’t want to accidentally clip the lung.

    eroded = morphology.erosion(thresh_img, np.ones([3, 3, 3]))
    dilation = morphology.dilation(eroded, np.ones([8, 8, 8]))

    labels = measure.label(dilation)  # Different labels are displayed in different colors
    # labels[labels > 1] = 0
    mask = fill_holes(labels, threshold=0)
    mask = morphology.dilation(mask, np.ones([10, 10, 10]))  # one last dilation

    if display:
        fig, ax = plt.subplots(3, 2, figsize=[12, 12])
        ax[0, 0].set_title("Original")
        ax[0, 0].imshow(img[:,256,:], cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 1].set_title("Threshold")
        ax[0, 1].imshow(thresh_img[:,256,:], cmap='gray')
        ax[0, 1].axis('off')
        ax[1, 0].set_title("After Erosion and Dilation")
        ax[1, 0].imshow(dilation[:,256,:], cmap='gray')
        ax[1, 0].axis('off')
        ax[1, 1].set_title("Color Labels")
        ax[1, 1].imshow(labels[:,256,:])
        ax[1, 1].axis('off')
        ax[2, 0].set_title("Final Mask")
        ax[2, 0].imshow(mask[:,256,:], cmap='gray')
        ax[2, 0].axis('off')
        ax[2, 1].set_title("Apply Mask on Original")
        res = mask * img
        ax[2, 1].imshow(res[:,256,:], cmap='gray')
        ax[2, 1].axis('off')

        plt.show()
        plt.close()
        # plt.savefig('t.png')

    return mask * img

def make_lungmask_2D(img, id, display=False):
    """
    # Standardize the pixel value by subtracting the mean and dividing by the standard deviation
    # Identify the proper threshold by creating 2 KMeans clusters comparing centered on soft tissue/bone vs lung/air.
    # Using Erosion and Dilation which has the net effect of removing tiny features like pulmonary vessels or noise
    # Identify each distinct region as separate image labels (think the magic wand in Photoshop)
    # Using bounding boxes for each image label to identify which ones represent lung and which ones represent “every thing else”
    # Create the masks for lung fields.
    # Apply mask onto the original image to erase voxels outside of the lung fields.
    """

    row_size, col_size = img.shape

    # mean = np.mean(img)
    # std = np.std(img)
    # img = img - mean
    # img = img / std

    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = img[int(col_size / 5):int(col_size / 5 * 4), int(row_size / 5):int(row_size / 5 * 4)]
    mean = np.mean(middle)
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, I’m moving the
    # underflow and overflow on the pixel spectrum
    img[img >= max] = mean
    img[img <= min] = mean

    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img < threshold, 1.0, 0.0)  # threshold the image

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.
    # We don’t want to accidentally clip the lung.

    eroded = morphology.erosion(thresh_img, np.ones([5, 5]))
    dilation = morphology.dilation(eroded, np.ones([8, 8]))

    labels = measure.label(dilation)  # Different labels are displayed in different colors
    # labels[labels > 1] = 0
    mask = fill_holes_2D(labels, threshold=0)
    # find lung contour by contour area
    filled_mask = np.zeros_like(mask).astype(np.uint8)

    contours = []
    _, contour_cand, _ = cv2.findContours(mask, 1, 2)
    for cand in contour_cand:
        if cv2.contourArea(cand) > 100:
            contours.append(cand)

    if len(contours) == 0:
        contours = []
        for candi in contour_cand:
            contours.append(candi)

    v = 10
    for cnt in contours:
        filled_mask = cv2.drawContours(filled_mask, [cnt], -1, (v), -1)
        v += 10

    filled_mask[filled_mask > 0] = 1

    mask = morphology.dilation(filled_mask, np.ones([10, 10]))  # one last dilation

    if display:
        fig, ax = plt.subplots(3, 2, figsize=[12, 12])
        ax[0, 0].set_title("Original")
        ax[0, 0].imshow(img, cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 1].set_title("Threshold")
        ax[0, 1].imshow(thresh_img, cmap='gray')
        ax[0, 1].axis('off')
        ax[1, 0].set_title("After Erosion and Dilation")
        ax[1, 0].imshow(dilation, cmap='gray')
        ax[1, 0].axis('off')
        ax[1, 1].set_title("Color Labels")
        ax[1, 1].imshow(labels)
        ax[1, 1].axis('off')
        ax[2, 0].set_title("Final Mask")
        ax[2, 0].imshow(mask, cmap='gray')
        ax[2, 0].axis('off')
        ax[2, 1].set_title("Apply Mask on Original")
        res = mask * img
        ax[2, 1].imshow(res, cmap='gray')
        ax[2, 1].axis('off')

        # plt.show()
        plt.savefig('%s.png' % id)
        plt.close()
        print(id)

    return mask * img

def fill_holes(imInput, threshold):
    """
    The method used in this function is found from
    https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/

    """

    # Threshold.
    th, thImg = cv2.threshold(imInput.astype(np.uint8), threshold, 1, cv2.THRESH_BINARY_INV)

    # Copy the thresholded image.
    imFloodfill = thImg.copy()

    # Get the mask.
    h, w, z = thImg.shape
    # mask = np.zeros((w+2, z+2), np.uint8)
    # lbl = np.zeros((h, w, z), np.uint8)
    #
    # for n_slice in range(h):
    #     cv2.floodFill(imFloodfill[n_slice, :, :], mask, (0, 0), 1)

    # Floodfill from point (0, 0).
    # cv2.floodFill(imFloodfill, mask, (0, 0), 50)
    # imFloodfill[imFloodfill == 0] = 1
    img = thImg.copy() # + imFloodfill
    # img[img > 1] = 0

    # _, contours_cand, _ = cv2.findContours(img, 1, 2)

    # contours = []
    # for candi in contours_cand:
    #     print(cv2.contourArea(candi))
    #     if 5000 > int(cv2.contourArea(candi)) > 1500:
    #         contours.append(candi)
    #
    # new_img = np.zeros_like(img).astype(np.uint8)
    #
    # v = 10
    # for cnt in contours:
    #
    #     new_img = cv2.drawContours(new_img, [cnt], -1, (v), -1)
    #     v += 10
    # new_img[new_img > 0] = 1
    # # Invert the floodfilled image.
    # imFloodfillInv = cv2.bitwise_not(new_img)
    #
    # # Combine the two images.
    # imOut = thImg | imFloodfillInv
    # imOut = 255 - imOut

    new_img = np.zeros_like(img).astype(np.uint8)
    for i in range(h):
        _, contours_cand, _ = cv2.findContours(img[i, :, :], 1, 2)
        contours = []
        for candi in contours_cand:
            contours.append(candi)

        for cnt in contours:
            new_img[i, :, :] = cv2.drawContours(new_img[i, :, :], [cnt], -1, (255), -1)
    new_img[new_img > 0] = 1
    # Invert the floodfilled image.
    imFloodfillInv = cv2.bitwise_not(new_img)

    # Combine the two images.
    imOut = thImg | imFloodfillInv
    imOut = 255 - imOut

    return imOut

def fill_holes_2D(imInput, threshold):
    """
    The method used in this function is found from
    https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/

    """

    # Threshold.
    th, thImg = cv2.threshold(imInput.astype(np.uint8), threshold, 1, cv2.THRESH_BINARY_INV)

    # Copy the thresholded image.
    imFloodfill = thImg.copy()

    # Get the mask.
    h, w = thImg.shape

    img = thImg.copy() # + imFloodfill

    new_img = np.zeros_like(img).astype(np.uint8)
    _, contours_cand, _ = cv2.findContours(img, 1, 2)
    contours = []
    for candi in contours_cand:
        # if cv2.contourArea(candi) > 100:
        contours.append(candi)

    if len(contours) == 0:
        for candi in contours_cand:
            contours.append(candi)

    for cnt in contours:
        new_img = cv2.drawContours(new_img, [cnt], -1, (255), -1)

    new_img[new_img > 0] = 1

    # Invert the floodfilled image.
    imFloodfillInv = cv2.bitwise_not(new_img)

    # Combine the two images.
    imOut = thImg | imFloodfillInv
    imOut = 255 - imOut

    return imOut

def CT_data_load_norm(train_path):
    data_path = os.path.join(set.LC_img_dir_path, train_path)
    lbl_path = os.path.join(set.save_NFT_dir_path, train_path)
    # print('utils_def/CT_data_load/data_path', data_path)
    norm_img = load_DCM_image(data_path)
    lbl_img = load_NIFTY_image(lbl_path, norm_img.shape)

    if norm_img.shape == lbl_img.shape:
        norm_img = np.swapaxes(norm_img, 1, -1)
        norm_img = np.swapaxes(norm_img, 0, -1)

        lbl_img = np.swapaxes(lbl_img, 1, -1)
        lbl_img = np.swapaxes(lbl_img, 0, -1)

        return norm_img, lbl_img
    else:
        print("Shape ERROR!!")

def CT_data_load(train_path):
    data_path = os.path.join(set.LC_img_dir_path, train_path)
    lbl_path = os.path.join(set.save_NFT_dir_path, train_path)
    # print('utils_def/CT_data_load/data_path', data_path)

    norm_img = load_DCM_image(data_path)
    lbl_img, _ = read_NIFTY_image(train_path, norm_img.shape)
    print(norm_img.shape, lbl_img.shape)

    norm_slide, lbl_slide = slide_selection(norm_img, lbl_img)

    # return norm_img, lbl_img
    return norm_slide, lbl_slide

def Test_data_load(test_path):
    data_path = os.path.join(set.LC_img_dir_path, test_path)
    lbl_path = os.path.join(set.save_NFT_dir_path, test_path)

    norm_img = load_DCM_image(data_path)
    lbl_img = read_NIFTY_image(test_path, norm_img.shape)
    print(norm_img.shape, lbl_img.shape)

    LungMask = make_lungmask(norm_img, False)

    norm_slide, lbl_slide, num = slide_selection(LungMask, lbl_img)

    return norm_slide, lbl_slide

def Test_slide_load(test_path):
    """
    For test, we need find lung area's slide number.
    So, we find slide area in center slide of sagittal plain.

    :param test_path: test dataset path
    :return: lung area slide and lung area slide number
    """

    norm_img = load_DCM_image(test_path)
    lbl_img, _ = read_NIFTY_image(test_path, norm_img.shape)

    sample_slide = norm_img[:, 256, :]

    Mask_Lung = make_lungmask_2D(sample_slide, test_path, False)
    num_slide = lung_slide_selection(Mask_Lung)
    num_label = label_slide_selection(lbl_img)


    return norm_img, lbl_img, num_slide, num_label


def lung_slide_selection(CT):
    ct_slides = CT.sum(-1)

    ct_slide_num = sorted(np.where(ct_slides > 0)[0])

    slide_num = [i for i in range(ct_slide_num[0], ct_slide_num[-1])]

    if 10 in slide_num:
        num = []
        for i in slide_num:
            if i > 11:
                num.append(i)
        return num
    else:
        return slide_num


def label_slide_selection(lbl):
    lbl_slides = lbl.sum(-1).sum(-1)

    num_slide = np.where(lbl_slides > 0)[0]

    return num_slide


def slide_selection(CT, lbl):
    lbl_slides = lbl.sum(-1).sum(-1)

    num_slide = np.where(lbl_slides > 0)[0]

    return CT[num_slide], lbl[num_slide], num_slide

def slide_patch_selection(CTs, SLs, LBLs):
    # slice based mathod change to patch based classification

    normal, abnormal = [], []
    for l, lbl, slide, sld, area in zip(range(LBLs.shape[0]), LBLs, CTs, SLs, LBLs.sum(-1).sum(-1)):

        lbl_patches = cutup(lbl, (64, 64), (32, 32))
        patches = cutup(sld, (64, 64), (32, 32))
        sld_patches = cutup(slide, (64, 64), (32, 32))

        x_patches, y_patches, _, _ = lbl_patches.shape

        lbl_patches = lbl_patches.reshape((-1, 64, 64))
        patches = patches.reshape((-1, 64, 64))
        sld_patches = sld_patches.reshape((-1, 64, 64))

        a = int(area * 0.7)

        for p, p_lbl, p_sld, d_sld in zip(range(lbl_patches.shape[0]), lbl_patches, patches, sld_patches):

            if p_lbl.sum(-1).sum(-1) >= a:
                abnormal.append(p)
            elif p_lbl.sum(-1).sum(-1) == 0:
                if d_sld.max() != 0.0:
                    if len(np.where(p_sld[p_sld > 0])[0]) >= 2048:
                        normal.append(p)
                else:
                    print('stop')

        print("normal:", len(normal), "abnormal", len(abnormal))

        if len(normal) == 0:
            print('ZERO NORMAL ERROR')

        normal_lbl = lbl_patches[normal]
        abnormal_lbl = lbl_patches[abnormal]

        normal_sld = sld_patches[normal]
        abnormal_sld = sld_patches[abnormal]

        if l == 0:
            normal_lbl_patches = normal_lbl.copy()
            abnormal_lbl_patches = abnormal_lbl.copy()

            normal_sld_patches = normal_sld.copy()
            abnormal_sld_patches = abnormal_sld.copy()
        else:
            normal_lbl_patches = np.concatenate((normal_lbl_patches, normal_lbl), axis=0)
            abnormal_lbl_patches = np.concatenate((abnormal_lbl_patches, abnormal_lbl), axis=0)

            normal_sld_patches = np.concatenate((normal_sld_patches, normal_sld), axis=0)
            abnormal_sld_patches = np.concatenate((abnormal_sld_patches, abnormal_sld), axis=0)

    # slide_patches = np.concatenate((normal_sld_patches, abnormal_sld_patches), axis=0)
    # label_patches = np.concatenate((normal_lbl_patches, abnormal_lbl_patches), axis=0)

        # x = p // (x_patches - 1)
        # y = p - ((x_patches - 1) * x)

    return normal_sld_patches, abnormal_sld_patches, normal_lbl_patches, abnormal_lbl_patches

def test_patch_extraction(slides):

    patches = cutup(slides, (64, 64), (8, 8))

    p_x, p_y, _, _ = patches.shape

    patches = patches.reshape((-1, 64, 64))

    return p_x, p_y, patches

def test_patch_selection(slide, lbl):
    lbls = []
    # slice based mathod change to patch based classification
    area = lbl.sum(-1).sum(-1)

    lbl_patches = cutup(lbl, (64, 64), (64, 64))
    sld_patches = cutup(slide, (64, 64), (64, 64))

    x_patches, y_patches, _, _ = lbl_patches.shape

    lbl_patches = lbl_patches.reshape((-1, 64, 64))
    sld_patches = sld_patches.reshape((-1, 64, 64))

    # x = p // (x_patches - 1)
    # y = p - ((x_patches - 1) * x)

    a = int(area * 0.7)

    ll = []
    for p_lbl in lbl_patches:
        if p_lbl.sum(-1).sum(-1) >= a:
            ll.append(1)
        else:
            ll.append(0)

    lbls.append(ll)

    return sld_patches, lbls

def test_slide_patch_selection(CTs, LBLs):
    lbls = []
    # slice based mathod change to patch based classification
    for l, lbl, slide, area in zip(range(LBLs.shape[0]), LBLs, CTs, LBLs.sum(-1).sum(-1)):

        lbl_patches = cutup(lbl, (64, 64), (64, 64))
        sld_patches = cutup(slide, (64, 64), (64, 64))

        x_patches, y_patches, _, _ = lbl_patches.shape

        lbl_patches = lbl_patches.reshape((-1, 64, 64))
        sld_patches = sld_patches.reshape((-1, 64, 64))

        # x = p // (x_patches - 1)
        # y = p - ((x_patches - 1) * x)

        a = int(area * 0.7)

        ll = []
        for p_lbl in lbl_patches:
            if p_lbl.sum(-1).sum(-1) >= a:
                ll.append(1)
            else:
                ll.append(0)

        lbls.append(ll)

    return sld_patches, lbls

def train_data_load(shape_nm, shape_abn, shape_nm_l, shape_abn_l):

    train_normal_data = np.memmap(filename='./Data/train_normal_data4.dat', dtype=np.float32, mode='r+',
                                  shape=(shape_nm[0], shape_nm[1], shape_nm[2]))
    train_abnormal_data = np.memmap(filename='./Data/train_abnormal_data4.dat', dtype=np.float32, mode='r+',
                                    shape=(shape_abn[0], shape_abn[1], shape_abn[2]))

    # train_normal_label = np.memmap(filename='./Data/train_normal_label4.lbl', dtype=np.uint8, mode='r+',
    #                                shape=(shape_nm_l[0], shape_nm_l[1], shape_nm_l[2]))
    # train_abnormal_label = np.memmap(filename='./Data/train_abnormal_label4.lbl', dtype=np.uint8, mode='r+',
    #                                  shape=(shape_abn_l[0], shape_abn_l[1], shape_abn_l[2]))

    train_normal_label = np.zeros([shape_nm_l[0]])
    train_abnormal_label = np.ones([shape_abn_l[0]])

    return train_normal_data, train_abnormal_data, train_normal_label, train_abnormal_label

def FP_reduction(test_lists, lbl_slides):

    for t, test_res in enumerate(test_lists):
        print(np.where(test_res > 0))
        plt.imsave('test_res_%d' % t, test_res)




def gaussian_normalization(train, test):
    mu = np.mean(train, axis=2)
    std = np.std(train, axis=2)

    temp_train = np.subtract(train, mu)
    train = np.divide(temp_train, std)

    temp_test = np.subtract(test, mu)
    test = np.divide(temp_test, std)

    return train, test

def norm(train, test):
    return train/255, test/255

def test_results(predict, real):
        from sklearn.metrics import confusion_matrix

        aa = confusion_matrix(real, predict)
        print("TP", aa[1, 1], aa[1, 1] * 100. / (aa[1,1] + aa[1,0]))
        print("FN", aa[1, 0], aa[1, 0] * 100. / (aa[1,1] + aa[1,0]))
        print("TN", aa[0, 0], aa[0, 0] * 100. / (aa[0,1] + aa[0,0]))
        print("FP", aa[0, 1], aa[0, 1] * 100. / (aa[0,1] + aa[0,0]))

        print("ACC", (aa[1, 1] + aa[0,0]) * 100. / np.sum(aa))
        print("SEN", aa[1, 1]  * 100. / (aa[1,1] + aa[1,0]))
        print("SPC", aa[0, 0] * 100. / (aa[0,0] + aa[0,1]))

        acc = (aa[1, 1] + aa[0,0]) * 100. / np.sum(aa)
        sen = aa[1, 1]  * 100. / (aa[1,1] + aa[1,0])
        spc = aa[0, 0] * 100. / (aa[0,0] + aa[0,1])
        return acc, sen, spc

def readCSV(filename):
    lines = []
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines[1:]

def k_fold_cross_validation(items, k, randomize=False):
    if randomize:
        items = list(items)
        shuffle(items)

    slices = [items[i::k] for i in range(k)]

    # for i in range(k):
    # validation = slices[0]
    test = slices[1]
    training = slices[0] + slices[2] + slices[3] + slices[4]
    return training, test  # validation,
    # return training, test

def rolling_window_lastaxis(a, window):
    if window < 1:
        raise ValueError("`window` must be at least 1.")
    if window > a.shape[-1]:
        raise ValueError("`window` is too long.")
    shape = (128, 128) + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    print(shape, strides)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def rolling_window(a: object, window: object) -> object:
    if not hasattr(window, '__iter__'):
        return rolling_window_lastaxis(a, window)
    for i, win in enumerate(window):
        if win > 1:
            a = a.swapaxes(i, -1)
            a = rolling_window_lastaxis(a, window)
            a = a.swapaxes(-2, i)
    return a

def cutup(data, blck, strd):
    sh = np.array(data.shape)
    blck = np.asanyarray(blck)
    strd = np.asanyarray(strd)
    nbl = (sh - blck) // strd + 1
    strides = np.r_[data.strides * strd, data.strides]
    dims = np.r_[nbl, blck]
    data6 = np.lib.stride_tricks.as_strided(data, strides=strides, shape=dims)
    return data6#.reshape(-1, *blck)
