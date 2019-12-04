import os

import numpy as np
import pydicom
from PIL import Image
import scipy.ndimage
import shutil

from crop_image import crop_img_from_largest_connected, image_orientation

basepath = 'D:/!_DATA/DICOM/RIDER Breast MRI/RIDER-1125105682/09-09-1880-coffee break exam - t0 mins-53837/2-early anatomical reference-77018'

views = ['RCC', 'LCC', 'LMLO', 'RMLO']

def GetDefaultImageWindowLevel(data, intercept=0, slope=1):
    """Determine the default window/level for the DICOM image."""

    wmax = 0
    wmin = 0
    # Rescale the slope and intercept of the image if present
    pixel_array = data * slope + intercept

    if (pixel_array.max() > wmax):
        wmax = pixel_array.max()
    if (pixel_array.min() < wmin):
        wmin = pixel_array.min()
    # Default window is the range of the data array
    window = int(wmax - wmin)
    # Default level is the range midpoint minus the window minimum
    level = int(window / 2 - abs(wmin))
    return window, level


def GetLUTValue(data, window, level):
    """Apply the RGB Look-Up Table for the data and window/level value."""

    lutvalue = np.piecewise(data,
                            [data <= (level - 0.5 - (window - 1) / 2),
                             data > (level - 0.5 + (window - 1) / 2)],
                            [0, 255, lambda data:
                            ((data - (level - 0.5)) / (window-1) + 0.5) *
                            (255 - 0)])
    # Convert the resultant array to an unsigned 8-bit array to create
    # an 8-bit grayscale LUT since the range is only from 0 to 255
    return np.array(lutvalue, dtype=np.uint16)


def GetImage(data, window=0, level=0, intercept=0, slope=1):
    """Return the image from a DICOM image storage file."""

    pixel_array = data

    if ((window == 0) and (level == 0)):
        window, level = GetDefaultImageWindowLevel(pixel_array)

    rescaled_image = pixel_array * slope + intercept

    image = GetLUTValue(rescaled_image, window, level)
    return Image.fromarray(image).convert('L')


def main():
    files = []
    print(basepath)
    for fname in os.listdir(basepath):
        if os.path.isfile(os.path.join(basepath, fname)):
            print("loading: {}".format(basepath + fname))
            files.append(pydicom.read_file(basepath + '/' + fname))

    slices = []
    skipcount = 0
    for f in files:
        if hasattr(f, 'SliceLocation'):
            slices.append(f)
        else:
            skipcount = skipcount + 1

    print("skipped, no SliceLocation: {}".format(skipcount))

    slices = sorted(slices, key=lambda s: s.SliceLocation)

    window = slices[0].WindowWidth
    level = slices[0].WindowCenter
    patient_id = slices[0].PatientID
    output_dir = patient_id + '_output_pngs/'
    ps = slices[0].PixelSpacing
    ss = slices[0].SliceThickness
    ax_aspect = ps[1]/ps[0]
    sag_aspect = ps[1]/ss
    cor_aspect = ss/ps[0]
    aspect_ratio = [1, cor_aspect]
    img_smooth_filter = 5

    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    img3d = np.zeros(img_shape, dtype=np.uint16)

    for i, s in enumerate(slices):
        img2d = s.pixel_array
        img3d[:, :, i] = img2d

    basepath_output_dir = basepath + '/' + output_dir
    basepath_output_dir = basepath_output_dir.replace('//', '/')
    output_rcc_dir = basepath_output_dir + views[0] + '/'
    output_lcc_dir = basepath_output_dir + views[1] + '/'
    output_lmlo_dir = basepath_output_dir + views[2] + '/'
    output_rmlo_dir = basepath_output_dir + views[3] + '/'

    shutil.rmtree(basepath_output_dir, ignore_errors=True)
    os.makedirs(os.path.dirname(basepath_output_dir), exist_ok=True)
    os.makedirs(os.path.dirname(output_rcc_dir), exist_ok=True)
    os.makedirs(os.path.dirname(output_lcc_dir), exist_ok=True)
    os.makedirs(os.path.dirname(output_lmlo_dir), exist_ok=True)
    os.makedirs(os.path.dirname(output_rmlo_dir), exist_ok=True)

    # TODO calc correct middle slice position
    middle_lateral_slice = img_shape[1]//2

    black_color_threshold = 0.1

    """ RMLO / LMLO """
    non_empty_slices = []
    current_slice_lenght = 0
    current_slice_width = 0
    roi_box_biggest = [0,0,0,0]

    for i in range(0, img_shape[2]):
        if is_empty_image(img3d, i, black_color_threshold):
            continue
        current_slice = img3d[:, :middle_lateral_slice, i]

        roi_box = crop_img_from_largest_connected(current_slice, image_orientation('YES', 'L'), iterations=33, buffer_size=10)
        if roi_box is None:
            continue
        non_empty_slices.append(i)
        current_slice_lenght, current_slice_width, roi_box_biggest = \
            compare_crop_size(current_slice_lenght, current_slice_width, roi_box[0], roi_box_biggest)

    top, bottom, left, right = roi_box_biggest

    # save image
    for i in non_empty_slices:
        current_slice = img3d[:, :middle_lateral_slice, i] ##  == non_empty_slices[i]??
        GetImage(current_slice[top:bottom, left:right], window, level) \
            .save(output_lmlo_dir + patient_id + views[2] + str(i) + '.png')

    non_empty_slices = []
    current_slice_lenght = 0
    current_slice_width = 0
    roi_box_biggest = [0,0,0,0]

    for i in range(0, img_shape[2]):
        if is_empty_image(img3d, i, black_color_threshold):
            continue
        current_slice = img3d[:, middle_lateral_slice+1:, i]
        roi_box = crop_img_from_largest_connected(current_slice, image_orientation('YES', 'R'), iterations=33, buffer_size=10)
        if roi_box is None:
            continue
        non_empty_slices.append(i)
        current_slice_lenght, current_slice_width, roi_box_biggest = \
            compare_crop_size(current_slice_lenght, current_slice_width, roi_box[0], roi_box_biggest)

    top, bottom, left, right = roi_box_biggest

    # save image
    for i in non_empty_slices:
        current_slice = img3d[:, middle_lateral_slice+1:, i]
        GetImage(current_slice[top:bottom, left:right], window, level) \
            .save(output_rmlo_dir + patient_id + views[3] + str(i) + '.png')

    """RCC"""
    non_empty_slices = []
    current_slice_lenght = 0
    current_slice_width = 0
    roi_box_biggest = [0,0,0,0]

    for i in range(0, middle_lateral_slice):
        if is_empty_image(img3d, i, black_color_threshold):
            continue
        scaled_img = scipy.ndimage.zoom(img3d[:, i, :], aspect_ratio, order=img_smooth_filter, mode='wrap')
        scaled_img = np.rot90(scaled_img)
        roi_box = crop_img_from_largest_connected(scaled_img, image_orientation('NO', 'L'), iterations=33, buffer_size=10)
        if roi_box is None:
            continue
        non_empty_slices.append(i)
        current_slice_lenght, current_slice_width, roi_box_biggest = \
            compare_crop_size(current_slice_lenght, current_slice_width, roi_box[0], roi_box_biggest)
        # top, bottom, left, right = roi_box[0]

    top, bottom, left, right = roi_box_biggest

    # save image
    for i in non_empty_slices:
        scaled_img = scipy.ndimage.zoom(img3d[:, i, :], aspect_ratio, order=img_smooth_filter, mode='wrap')
        scaled_img = np.rot90(scaled_img)
        GetImage(scaled_img[top:bottom, left:right], window, level) \
            .save(output_rcc_dir + patient_id + views[0] + str(i) + '.png')

    """LCC"""
    non_empty_slices = []
    current_slice_lenght = 0
    current_slice_width = 0
    roi_box_biggest = [0,0,0,0]

    for i in range(middle_lateral_slice, img_shape[1]):
        if is_empty_image(img3d, i, black_color_threshold):
            continue
        scaled_img = scipy.ndimage.zoom(img3d[:, i, :], aspect_ratio, order=img_smooth_filter, mode='wrap')
        scaled_img = np.rot90(scaled_img)
        roi_box = crop_img_from_largest_connected(scaled_img, image_orientation('NO', 'R'), iterations=33, buffer_size=10)
        if roi_box is None:
            continue
        non_empty_slices.append(i)
        current_slice_lenght, current_slice_width, roi_box_biggest = \
            compare_crop_size(current_slice_lenght, current_slice_width, roi_box[0], roi_box_biggest)

    top, bottom, left, right = roi_box_biggest

    # save image
    for i in non_empty_slices:
        scaled_img = scipy.ndimage.zoom(img3d[:, i, :], aspect_ratio, order=img_smooth_filter, mode='wrap')
        scaled_img = np.rot90(scaled_img)
        GetImage(scaled_img[top:bottom, left:right], window, level)\
            .save(output_lcc_dir + patient_id + views[1] + str(i) + '.png')

    print('finished')


def is_empty_image(img3d, index, black_color_threshold):
    return (np.count_nonzero(img3d[:, index, :]) / img3d[:, index, :].size) < black_color_threshold


def compare_crop_size(current_slice_lenght, current_slice_width, roi_box_params, roi_box_biggest):
    current_slice_lenght_tmp = roi_box_params[1] - roi_box_params[0]
    current_slice_width_tmp = roi_box_params[3] - roi_box_params[2]
    if current_slice_lenght_tmp > current_slice_lenght:
        current_slice_lenght = current_slice_lenght_tmp
        roi_box_biggest[0] = roi_box_params[0]
        roi_box_biggest[1] = roi_box_params[1]
    if current_slice_width_tmp > current_slice_width:
        current_slice_width = current_slice_width_tmp
        roi_box_biggest[2] = roi_box_params[2]
        roi_box_biggest[3] = roi_box_params[3]
    return current_slice_lenght, current_slice_width, roi_box_biggest


if __name__ == "__main__":
    main()
