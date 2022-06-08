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

    
    index_params = dict(algorithm=1, trees=30)
    search_params = dict(checks=20)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    return detector, matcher


def filter_matches(kp1, kp2, matches, ratio=0.75):
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
    corners = np.int32(cv.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2))

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
    p1, p2, kp_pairs = filter_matches(object_feature, frame_feature, raw_matches)  # compare features
    if len(p1) >= 4:
        Homography = cv.findHomography(p1, p2, cv.RANSAC, 5.0)[0]  # tìm điểm tương đồng
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
            object_coord, object_corners = get_object_coord(object_image, frame, homography)
            if object_coord:
                cv2.circle(frame, object_coord, 4, (255, 0, 0), -1)
                cv2.putText(frame, f"{object_coord}", object_coord,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

                if show_corner == True:
                    for corner in object_corners:
                        cv2.circle(frame, corner, 3, (0, 255, 0), -1)

        cv2.imshow("find_object", frame)
        k = cv.waitKey(1)
        if k == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

GOOD_IDCARD_MATCHER = 150



def ObjectMatching(image: object, object_image: object, matcher, object_feature, image_feature) -> object:
    """
compare two images.
    :param image: checked image
    :param object_image: check image
    :return: bestest_object, match_rating
    """

    kpts1, descs1 = object_feature
    kpts2, descs2 = image_feature
    
    try:
        if descs1 is not None and descs2 is not None:
            matches = matcher.knnMatch(descs1, descs2, 2)
        else:
            return None, 0, (None, None)
    except:
        print("error matcher.knnMatch", descs1, descs2)
        return None, 0, (None, None)

    # Sort by their distance.
    matches = sorted(matches, key=lambda x: x[0].distance)
    # Ratio test, to get good matches.
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    #
    # queryIndex for the small object, trainIndex for the scene )
    if len(good) > GOOD_IDCARD_MATCHER:
        # print("len(good)", len(good))
        src_pts = np.float32([kpts1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpts2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        Homography = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)[0]
        
        if Homography is not None:
            try:
                h, w = image.shape[:2]
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                h, w = object_image.shape[:2]
                dst = cv2.perspectiveTransform(pts, Homography)
                perspectiveM = cv2.getPerspectiveTransform(np.float32(dst), pts)
                bestest_object = cv2.warpPerspective(image, perspectiveM, (w, h), borderMode=cv2.BORDER_TRANSPARENT)
            except:
                bestest_object = None
            #
            object_coord, object_corners = get_object_coord(object_image, image, Homography)
            return bestest_object, len(good), (object_coord, object_corners)

    return None, len(good), (None, None)

def getobject(image, object_images, detector, matcher, object_features):
    '''
    detect idcard im image & idcard type
    nhập vào một image và các card cần tìm, trả về thông tin: detected_card, card_type, rating
    :param image: image input
    :param idcards: the idcard you want to find
    :return: detected_image_idcard, idcardtype, rating
    '''
    max = 0
    result = [None, -1, 0]
    image_feature = detector.detectAndCompute(image, None)

    for i, (object_image, object_feature) in enumerate(zip(object_images, object_features)):
        matched_info = ObjectMatching(image, object_image, matcher, object_feature, image_feature)
        if matched_info[1] >= max:
            object_type = i
            final = matched_info
            max = matched_info[1]
    return final, object_type