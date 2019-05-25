
import sys
import argparse
import math
import cv2 as cv
import numpy as np



def parse_arguments(argv, prog=''):
    # Initialize the command-line parser
    parser = argparse.ArgumentParser(prog,
                                     description='Script for patch match.')

    #
    # Main input/output arguments
    #

    parser.add_argument('--source',
                        type=str,
                        help='Path to source image',
                        required=True)
    parser.add_argument('--radius',
                        type=int,
                        help='The radius of the image patch',
                        required=True)
    parser.add_argument('--output',
                        type=str,
                        help='Path to save the results',
                        default='output',
                        required=False)
    parser.add_argument('--custom',
                        type=str,
                        help='Whether use custom Canny edge or build in Canny edge',
                        default='output',
                        required=True)
    parser.add_argument('--max',
                        type=int,
                        help='the maximum threshold',
                        default='output',
                        required=True)
    parser.add_argument('--min',
                        type=int,
                        help='the minimum threshold',
                        default='output',
                        required=True)
    # Run the python argument-parsing routine, leaving any
    #  unrecognized arguments intact
    args, unprocessed_argv = parser.parse_known_args(argv)

    success = True
    msg = ''

    # return any arguments that were not recognized by the parser
    return success, args, unprocessed_argv, msg  #

def read_image(location):
    img = cv.imread(location, cv.IMREAD_COLOR)
    return img

def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    
    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255
                
               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass
    
    return Z
def threshold(self, img):

    highThreshold = img.max() * self.highThreshold;
    lowThreshold = highThreshold * self.lowThreshold;

    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)

    weak = np.int32(self.weak_pixel)
    strong = np.int32(self.strong_pixel)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return (res)

def hysteresis(self, img):

    M, N = img.shape
    weak = self.weak_pixel
    strong = self.strong_pixel

    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass

    return img
def main(argv, prog=''):

    # Parse the command line arguments
    success, args, unprocessed_argv, msg = parse_arguments(argv, prog)

    source = read_image(args.source)
    source = cv.cvtColor(source, cv.COLOR_BGR2GRAY)
    radius = args.radius
    if(args.custom == "True"):
        g_smooth = cv.GaussianBlur(source, (5,5), 0)
        sobel_x = cv.Sobel(g_smooth, cv.CV_16S, 1, 0, 3)
        sobel_y = cv.Sobel(g_smooth, cv.CV_16S, 0, 1, 3)
        edge_gradient = np.hypot(sobel_x, sobel_y)
        angle = np.arctan2(sobel_y, sobel_x)
        #non maxima suppresion
        edge_gradient = non_max_suppression(edge_gradient, angle)
        cv.imwrite(args.output, edge_gradient)
        
    else:
        edges = cv.Canny(source, 100, 200, True)
        cv.imwrite(args.output, edges)
    
    




# Include these lines so we can run the script from the command line
if __name__ == '__main__':
    main(sys.argv[1:], sys.argv[0])