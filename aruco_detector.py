#!/usr/bin/python3

import numpy as np
import cv2
import cv2.aruco as aruco

cap = cv2.VideoCapture(0)  # Get the camera source

def load_coefficients(path):
    """ Loads camera matrix and distortion coefficients. """
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]

def track(matrix_coefficients, distortion_coefficients):    
    while True:
        ret, frame = cap.read()
        # operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Change grayscale
        gray = cv2.GaussianBlur(gray, [9,9], 1)
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)  # Use 5x5 dictionary to find markers
        parameters = aruco.DetectorParameters()  # Marker detection parameters
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        parameters.cornerRefinementMinAccuracy = 0.0001
        parameters.cornerRefinementMaxIterations = 10000
        # lists of ids and the corners beloning to each id
        corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict,
                                                                parameters=parameters)
        if np.all(ids is not None):  # If there are markers found by detector
            for i in range(0, len(ids)):  # Iterate in markers
                # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
                rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], 0.048, matrix_coefficients,
                                                                           distortion_coefficients)
                # (rvec - tvec).any()  # get rid of that nasty numpy value array error
                aruco.drawDetectedMarkers(frame, corners, ids)  # Draw A square around the markers
                cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.048)  # Draw Axis
                # corners_aruco = np.array(corners).reshape((4, 2))
                # (topLeft, topRight, bottomRight, bottomLeft) = corners_aruco
                print(f'Aruco ID: {ids.squeeze()}')
                print(tvec)
                # cv2.putText(frame, str(aruco_dict) + " " + str(int(ids)),
                            # (int(topLeft[0] - 5), int(topLeft[1])), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255))

        # Display the resulting frame
        cv2.imshow('frame', frame)
        # Wait 3 milisecoonds for an interaction. Check the key and do the corresponding job.
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    track(*load_coefficients('./cal.yml'))