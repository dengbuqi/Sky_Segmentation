import cv2
import numpy as np

def channel_dis(img): 
    '''
        For the pixels in the sky area, the values of the R, G, and B components are very large and almost equal. 
        -- "Single Image Defogging Algorithm Based on Sky Region Segmentation" Section 2.4.1 by Shufang Xu
    '''
    B, G, R = cv2.split(img)
    R_G = np.absolute(R-G)
    R_B = np.absolute(R-B)
    B_G = np.absolute(B-G)
    return R_G+R_B+B_G

def topNavarage(img,N=10):
    flattened_img = img.flatten()
    sorted_img = np.sort(flattened_img)[::-1]
    top_N_percent_index = int(len(sorted_img) * N/100)
    top_N_percent_values = sorted_img[:top_N_percent_index]
    average_top_N_percent = np.mean(top_N_percent_values)
    return average_top_N_percent

def topmean(img):
    mean_value = np.mean(img)
    greater_than_mean = img[img > mean_value]
    average_greater_than_mean = np.mean(greater_than_mean)
    return average_greater_than_mean

def GradientMagnitude(gray_img):
    '''
        Gradient of sky area is small
    '''
    grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return gradient_magnitude

def sky_seg(img):  # 1 is sky, 0 is other
    mean_rgb = np.mean(img, axis=2)
    gradient_magnitude = GradientMagnitude(mean_rgb)
    gradient_magnitude = gradient_magnitude/np.max(gradient_magnitude)
    rgb_dis = channel_dis(img)
    # seg = (((1-rgb_dis)+mean_rgb)/2)*(1-gradient_magnitude)
    seg = ((1-rgb_dis)+mean_rgb)*(1-gradient_magnitude)
    # th = topNavarage(rgb_dis,50)
    th = topmean(seg)
    # print(th)
    seg = (seg > th)
    
    return seg

def run(image_path, save_name):
    # Read the image
    image = cv2.imread(image_path)/255

    # Sky segmentation
    seg = sky_seg(image) # 1 is sky, 0 is other

    # Save the sky segmentation result
    cv2.imwrite(save_name, seg*255)

if __name__=='__main__':
    run('19_0.7_1.99.png', '19_0.7_1.99_sky_seg.png')
    run('21_0.7_1.86.png', '21_0.7_1.86_sky_seg.png')
    run('26_0.93_1.97.png', '26_0.93_1.97_sky_seg.png')
    run('29_0.85_1.7.png', '29_0.85_1.7_sky_seg.png')
    run('36_0.83_1.99.png', '36_0.83_1.99_sky_seg.png')
    run('361_low.png', '361_sky_seg.png')
    run('486_low.png', '486_sky_seg.png')



