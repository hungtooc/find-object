from __future__ import print_function
import cv2
import sys
import getopt
import numpy as np
import cv2 as cv


def line(p1, p2):
    """ 
    calculate linear equations
    return: hệ số của phương trình"""
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C


def intersection(L1, L2):
    ''' 
    calculate intersection coordinates
    return: return coordinate. return None if there is no intersection'''
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return int(x), int(y)
    else:
        return False


def init_feature(name):
    """ 
    init SIFT algorithm 
    return detector - feature extractor algorithm
            matcher - matching algorithm
    """
    detector = cv.SIFT_create()
    norm = cv.NORM_L2
    matcher = cv.BFMatcher(norm)
    return detector, matcher


def filter_matches(kp1, kp2, matches, ratio=0.65):
    """ 
    hàm lọc bỏ các features không giống với object"""
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append(kp1[m.queryIdx])
            mkp2.append(kp2[m.trainIdx])
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, list(kp_pairs)


def get_object_coord(object_image, frame, H, show_conner=False):
    '''
    hàm tính tọa độ của object
    return: tọa độ object, tọa độ 4 đỉnh của object
    '''
    h1, w1 = object_image.shape[:2]

    corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
    corners = np.int32(cv.perspectiveTransform(
        corners.reshape(1, -1, 2), H).reshape(-1, 2))

    L1 = line(corners[0], corners[2])  # đường chéo AC
    L2 = line(corners[1], corners[3])  # đường chéo BD

    # trung tâm object là giao của hai đường chéo
    center = intersection(L1, L2)
    return center, corners


def find_homography(object_image, frame, detector, matcher, desc1, object_feature):
    """ 
    hàm tìm điểm tương đồng giữa object và image. 
    return: Homography"""
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_feature, desc2 = detector.detectAndCompute(
        frame_gray, None)  # calculate feature of frame

    raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)
    p1, p2, kp_pairs = filter_matches(
        object_feature, frame_feature, raw_matches)  # compare features
    if len(p1) >= 4:
        Homography = cv.findHomography(p1, p2, cv.RANSAC, 5.0)[
            0]  # tìm điểm tương đồng
    else:
        Homography = None
    return Homography


if __name__ == '__main__':
    # init
    detector, matcher = init_feature('sift')
    cap = cv2.VideoCapture(0)  # 0 là index camera
    cv2.namedWindow("find_object", cv2.WINDOW_NORMAL)

    # load object & calculate object feature
    image_path = 'object_01.jpg'
    object_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    object_image = cv2.resize(object_image, (120, 120))
    object_feature, desc1 = detector.detectAndCompute(object_image, None)  # get feature of object

    # debug
    show_corner = False

    while True:

        _, frame = cap.read()  # read frame from camera
        homography = find_homography(object_image, frame)
        if homography is not None:
            object_coord, object_corners = get_object_coord(
                object_image, frame, homography)
            if object_coord:
                cv2.circle(frame, object_coord, 4, (255, 0, 0), -1)
                cv2.putText(frame, f"{object_coord}", object_coord,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)

                if show_corner == True:
                    for corner in object_corners:
                        cv2.circle(frame, corner, 3, (0, 255, 0), -1)

        cv2.imshow("find_object", frame)
        k = cv.waitKey(1)
        if k == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
