import time
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from matplotlib.animation import FuncAnimation



def get_img_point(images, grid=(9, 6)):
    ob_points = []
    img_points = []
    for img in images:
        ob_point = np.zeros((grid[0]*grid[1],3), np.float32)
        ob_point[:, :2] = np.mgrid[0:grid[0], 0:grid[1]].T.reshape(-1,2)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, grid, None)
        if ret:
            ob_points.append(ob_point)
            img_points.append(corners)
    return ob_points, img_points


def cal_undistort(img, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst


def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))

    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output


def magthresh(img, sobel_kernel=3, mag_thresh=(0, 255)):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    gradmag = np.sqrt(sobelx**2 + sobely**2)

    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


def hls_select(img,channel='s',thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if channel=='h':
        channel = hls[:,:,0]
    elif channel=='l':
        channel=hls[:,:,1]
    else:
        channel=hls[:,:,2]
    binary_output = np.zeros_like(channel)
    binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 1
    return binary_output


def luv_select(img, thresh=(0, 255)):
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    l_channel = luv[:,:,0]
    binary_output = np.zeros_like(l_channel)
    binary_output[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 1
    return binary_output


def lab_select(img, thresh=(0, 255)):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    b_channel = lab[:,:,2]
    binary_output = np.zeros_like(b_channel)
    binary_output[(b_channel > thresh[0]) & (b_channel <= thresh[1])] = 1
    return binary_output


def get_M_Minv(setsize):
    src = np.float32([[(203, 720), (585, 460), (695, 460), (1127, 720)]])
    dst = np.float32([[(320, 720), (320, 0), (960, 0), (960, 720)]])
    src = src // (1/setsize)
    dst = dst // (1/setsize)
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    return M, Minv


def thresholding(img):
    x_thresh = abs_sobel_thresh(img, orient='x', thresh_min=10 ,thresh_max=230)
    mag_thresh = magthresh(img, sobel_kernel=3, mag_thresh=(30, 150))
    dir_thresh = dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3))
    hls_thresh = hls_select(img, thresh=(180, 255))
    lab_thresh = lab_select(img, thresh=(155, 200))
    luv_thresh = luv_select(img, thresh=(225, 255))
    # Thresholding combination
    threshholded = np.zeros_like(x_thresh)
    threshholded[((x_thresh == 1) & (mag_thresh == 1)) | ((dir_thresh == 1) & (hls_thresh == 1)) | (lab_thresh == 1) | (luv_thresh == 1)] = 1

    return threshholded

def find_lane(binary_warped):
    # take a histogram of bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    midepoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midepoint])
    rightx_base = np.argmax(histogram[midepoint:]) + midepoint

    # Choose the num of the sliding Windows
    nwindows = 9
    # Set height of windows
    window_height = binary_warped.shape[0]/nwindows

    # Indentify the x and y positions of all nonzero pixels in the imaage
    non_zero = binary_warped.nonzero()      # got a list of non zero element's sublebal as a ndim array
    non_zero_x = np.array(non_zero[1])
    non_zero_y = np.array(non_zero[0])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of windows +/- margin
    margin = 50 #100
    # Set minimum num of the pixels found to recenter window
    minpix = 50
    # Create the empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the window one by one
    for win in range(nwindows):
        # Identify window boundaries in x and y
        win_y_low = binary_warped.shape[0] - (win+1)*window_height
        win_y_high = binary_warped.shape[0] - win*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((non_zero_y >= win_y_low) & (non_zero_y < win_y_high) &
                          (non_zero_x >= win_xleft_low) & (non_zero_x < win_xleft_high)).nonzero()[0]
        good_right_inds = ((non_zero_y >= win_y_low) & (non_zero_y < win_y_high) &
                           (non_zero_x >= win_xright_low) & (non_zero_x < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(non_zero_x[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(non_zero_x[good_right_inds]))
        # rect_l = plt.Rectangle((leftx_current-margin, win_y_low), 2*margin, window_height, edgecolor='g', alpha=1, facecolor='none')
        # rect_r = plt.Rectangle((rightx_current-margin, win_y_low), 2 * margin, window_height, edgecolor='g', alpha=1,
        #                        facecolor='none')
        # ax.add_patch(rect_l)
        # ax.add_patch(rect_r)
                    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = non_zero_x[left_lane_inds]
    lefty = non_zero_y[left_lane_inds]
    rightx = non_zero_x[right_lane_inds]
    righty = non_zero_y[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit, left_lane_inds, right_lane_inds




def calculate_curv_and_pos(binary_warped, left_fit, right_fit):
    # Define y-value where we want radius of curvature
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    leftx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    rightx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    y_eval = np.max(ploty)
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    curvature = ((left_curverad + right_curverad) / 2)
    # print(curvature)
    lane_width = np.absolute(leftx[719] - rightx[719])
    lane_xm_per_pix = 3.7 / lane_width
    veh_pos = (((leftx[719] + rightx[719]) * lane_xm_per_pix) / 2.)
    cen_pos = ((binary_warped.shape[1] * lane_xm_per_pix) / 2.)
    distance_from_center = veh_pos - cen_pos
    return curvature, distance_from_center


def fit_curv(binary_warped, left_fit, right_fit):
    fity = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    fit_leftx = left_fit[0] * fity ** 2 + left_fit[1] * fity + left_fit[2]
    fit_rightx = right_fit[0] * fity ** 2 + right_fit[1] * fity + right_fit[2]

    plt.plot(fit_leftx, fity, color='red')
    plt.plot(fit_rightx, fity, color='red')
    plt.xlim(0, binary_warped.shape[1])
    plt.ylim(binary_warped.shape[0], 0)
    plt.ioff()
    plt.show()




def draw_area(undist, binary_warped, Minv, left_fit, right_fit):
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result

def draw_values(img, curvature, distance_from_center):
    font = cv2.FONT_HERSHEY_SIMPLEX
    radius_text = "Radius of Curvature: %sm" % (round(curvature))

    if distance_from_center > 0:
        pos_flag = 'right'
    else:
        pos_flag = 'left'

    cv2.putText(img, radius_text, (100, 100), font, 1, (255, 255, 255), 2)
    center_text = "Vehicle is %.3fm %s of center" % (abs(distance_from_center), pos_flag)
    cv2.putText(img, center_text, (100, 150), font, 1, (255, 255, 255), 2)
    return img

def resize_img(img, set_size):
    new_img = cv2.resize(img, (0, 0), fx=set_size, fy=set_size, interpolation=cv2.INTER_NEAREST)
    return new_img


# Main Algorithm Begin:


# prepare object points
nx = 8 #TODO: enter the number of inside corners in x
ny = 6 #TODO: enter the number of inside corners in y

# set paths
# cal_paths = 'C:\Users\kangk\Desktop\chessboard\L/camera_cal'
cal_paths = 'C:\Users\kangk\Desktop\chessboard\L/sgcamera_cal'
test_paths = 'C:\Users\kangk\Desktop\chessboard\VGA/test'

# read images
img_names = os.listdir(cal_paths)
img_paths = [cal_paths+'/'+img_name for img_name in img_names]
imgs = [cv2.imread(path) for path in img_paths]
# cap_video = cv2.VideoCapture('C:\Users\kangk\Desktop\chessboard\project_video.mp4')
cap_video = cv2.VideoCapture('C:\Users\kangk\Desktop\chessboard\own3.mp4')
ob_point, im_point = get_img_point(imgs)

# modify camera
M, Mi = get_M_Minv(setsize=0.5)

# ret, frame = cap_video.read()
# rframe = resize_img(frame, 0.5)
#
# cv2.namedWindow("rframe", 1)
#
# cv2.imshow()
ret, frame = cap_video.read()
frame = resize_img(frame, 0.5)
print frame.shape, frame.shape[:2]
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(ob_point, im_point, frame.shape[1::-1], None, None)
mapx, mapy = cv2.initUndistortRectifyMap(mtx,dist,None,mtx,(640,360),5)
dst = cv2.remap(frame,mapx,mapy,cv2.INTER_LINEAR)
print dst.shape, mapx.shape
timage = cal_undistort(frame, ob_point, im_point)

while(True):
    ret, frame = cap_video.read()
    frame = resize_img(frame, 0.5)
    start_time = time.time()
    dst = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

    threshed_image = thresholding(dst)
    thresholded_wraped = cv2.warpPerspective(threshed_image, M, threshed_image.shape[1::-1], flags=cv2.INTER_LINEAR)
    end_time = time.time()
    dtime = end_time - start_time
    print dtime
    left_fit, right_fit, left_lane_inds, right_lane_inds = find_lane(thresholded_wraped)
    back2original = draw_area(frame, thresholded_wraped, Mi, left_fit, right_fit)
    cv2.imshow('line', back2original)
    cv2.waitKey(1)
    if cv2.waitKey(1) == ord('q'):
        break

