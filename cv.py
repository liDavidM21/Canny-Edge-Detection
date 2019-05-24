
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


def main(argv, prog=''):

    # Parse the command line arguments
    success, args, unprocessed_argv, msg = parse_arguments(argv, prog)

    source = read_image(args.source)
    source = cv.cvtColor(source, cv.COLOR_BGR2GRAY)
    radius = args.radius
    g_smooth = cv.GaussianBlur(source, (5,5), 0)
    sobel_x = cv.Sobel(g_smooth, cv.CV_64F, 1, 0, 3)
    sobel_y = cv.Sobel(g_smooth, cv.CV_64F, 0, 1, 3)
    edge_gradient = np.zeros(sobel_x.shape)
    angle = np.zeros(sobel_x.shape)
    for i in range(sobel_x.shape[0]):
        for j in range(sobel_x.shape[1]):
            edge_gradient[i][j] = math.sqrt(sobel_x[i][j]**2 + sobel_y[i][j]**2)
            angle[i][j] = math.degrees(math.atan(sobel_y[i][j]/sobel_x[i][j]))

    print(edge_gradient)
    cv.imwrite("output.jpg", edge_gradient)
    
    




# Include these lines so we can run the script from the command line
if __name__ == '__main__':
    main(sys.argv[1:], sys.argv[0])
