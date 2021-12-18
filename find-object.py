#!/usr/bin/env python

'''
Feature-based image matching sample.

Note, that you will need the https://github.com/opencv/opencv_contrib repo for SIFT and SURF

USAGE
  find_obj.py [--feature=<sift|surf|orb|akaze|brisk>[-flann]] [ <image1> <image2> ]

  --feature  - Feature to use. Can be sift, surf, orb or brisk. Append '-flann'
               to feature name to use Flann-based matcher instead bruteforce.

  Press left mouse button on a feature point to see its matching point.
'''

# Python 2/3 compatibility
from __future__ import print_function
import cv2

fourcc = cv2.VideoWriter_fourcc(*'XVID')  # cv2.VideoWriter_fourcc() does not exist
video_writer = cv2.VideoWriter("output.avi", fourcc, 10, (200, 152))
index = 0
save_flag = False
cap = cv2.VideoCapture(2)
# cap = cv2.VideoCapture("sample-video.mp4")
# cv2.namedWindow("cap", cv2.WINDOW_NORMAL)
import numpy as np
import cv2 as cv


# from common import anorm, getsize

def getsize(img):
    h, w = img.shape[:2]
    return w, h


def anorm2(a):
    return (a * a).sum(-1)


def anorm(a):
    return np.sqrt(anorm2(a))


def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C


def intersection(L1, L2):
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return int(x), int(y)
    else:
        return False


def getSpPoint(A, B, C):
    # x1 = A[0]
    # y1 = A[1]
    # x2 = B[0]
    # y2 = B[1]
    # x3 = C[0]
    # y3 = C[1]
    px = B[0] - A[0]
    py = B[1] - A[1]
    dAB = px * px + py * py
    u = ((C[0] - A[0]) * px + (C[1] - A[1]) * py) / dAB
    x = A[0] + u * px
    y = A[1] + u * py
    return int(x), int(y)  # this is D


FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH = 6


def init_feature(name):
    chunks = name.split('-')
    if chunks[0] == 'sift':
        detector = cv.SIFT_create()
        norm = cv.NORM_L2
    elif chunks[0] == 'surf':
        detector = cv.xfeatures2d.SURF_create(800)
        norm = cv.NORM_L2
    elif chunks[0] == 'orb':
        detector = cv.ORB_create(400)
        norm = cv.NORM_HAMMING
    elif chunks[0] == 'akaze':
        detector = cv.AKAZE_create()
        norm = cv.NORM_HAMMING
    elif chunks[0] == 'brisk':
        detector = cv.BRISK_create()
        norm = cv.NORM_HAMMING
    else:
        return None, None
    if 'flann' in chunks:
        if norm == cv.NORM_L2:
            flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        else:
            flann_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,  # 12
                                key_size=12,  # 20
                                multi_probe_level=1)  # 2
        matcher = cv.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    else:
        matcher = cv.BFMatcher(norm)
    return detector, matcher


def filter_matches(kp1, kp2, matches, ratio=0.75):
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


def explore_match(win, img1, img2, kp_pairs, status=None, H=None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    # vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
    # vis[:h1, :w1] = img1
    # vis[:h2, w1:w1 + w2] = img2
    vis = img2.copy()
    # vis = cv.cvtColor(vis, cv.COLOR_GRAY2BGR)
    # cv2.circle(img2, (1, 1), 4, (0, 0, 255), -1)
    # cv2.circle(img2, (1, img2.shape[0]), 4, (0, 0, 255), -1)
    # cv2.circle(img2, (img2.shape[1], 1), 4, (0, 0, 255), -1)
    # cv2.circle(img2, (img2.shape[1], img2.shape[0]), 4, (0, 0, 255), -1)

    cv2.putText(img2, f"image size: {img2.shape[:-1]}", (1, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
    # cv2.putText(img2, )
    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        # corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32(cv.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2))

        try:
            L1 = line(corners[0], corners[2])
            L2 = line(corners[1], corners[3])

            center = intersection(L1, L2)
            cv2.circle(img2, center, 4, (255, 0, 0), -1)
            cv2.putText(img2, f"{center}", center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
            sp = getSpPoint(corners[0], corners[1], center)
            cv2.line(img2, center, sp, (255, 255, 255), 3)
        except:
            pass
        # cv2.putText(img2, )
        # corners = np.int32(cv.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
        # cv.polylines(img2, [corners], False, (255, 255, 255), 3)
        for corner in corners:
            cv2.circle(img2, corner, 3, (0, 255, 0), -1)
            # cv2.imshow(win, img2)
            # cv2.waitKey()
    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)
    p1, p2 = [], []  # python 2 / python 3 change of zip unpacking
    for kpp in kp_pairs:
        p1.append(np.int32(kpp[0].pt))
        p2.append(np.int32(np.array(kpp[1].pt) + [w1, 0]))

    green = (0, 255, 0)
    red = (0, 0, 255)
    kp_color = (51, 103, 236)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            # cv.circle(vis, (x1, y1), 2, col, -1)
            # cv.circle(vis, (x2, y2), 2, col, -1)
        else:
            col = red
            r = 2
            thickness = 3
            # cv.line(vis, (x1 - r, y1 - r), (x1 + r, y1 + r), col, thickness)
            # cv.line(vis, (x1 - r, y1 + r), (x1 + r, y1 - r), col, thickness)
            # cv.line(vis, (x2 - r, y2 - r), (x2 + r, y2 + r), col, thickness)
            # cv.line(vis, (x2 - r, y2 + r), (x2 + r, y2 - r), col, thickness)
    # vis0 = img2.copy()
    # for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
    #     if inlier:
    #         cv.line(vis, (x1, y1), (x2, y2), green)

    cv.imshow(win, img2)

    # def onmouse(event, x, y, flags, param):
    #     cur_vis = vis
    #     if flags & cv.EVENT_FLAG_LBUTTON:
    #         cur_vis = vis0.copy()
    #         r = 8
    #         m = (anorm(np.array(p1) - (x, y)) < r) | (anorm(np.array(p2) - (x, y)) < r)
    #         idxs = np.where(m)[0]
    #
    #         kp1s, kp2s = [], []
    #         for i in idxs:
    #             (x1, y1), (x2, y2) = p1[i], p2[i]
    #             col = (red, green)[status[i][0]]
    #             cv.line(cur_vis, (x1, y1), (x2, y2), col)
    #             kp1, kp2 = kp_pairs[i]
    #             kp1s.append(kp1)
    #             kp2s.append(kp2)
    #         cur_vis = cv.drawKeypoints(cur_vis, kp1s, None, flags=4, color=kp_color)
    #         cur_vis[:, w1:] = cv.drawKeypoints(cur_vis[:, w1:], kp2s, None, flags=4, color=kp_color)
    #
    #     cv.imshow(win, cur_vis)

    # cv.setMouseCallback(win, onmouse)
    return img2


def main():
    cv2.namedWindow("find_obj", cv2.WINDOW_NORMAL)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # cv2.VideoWriter_fourcc() does not exist
    # video_writer = cv2.VideoWriter("output.mp4", fourcc, 10, (640, 480))

    import sys, getopt
    opts, args = getopt.getopt(sys.argv[1:], '', ['feature='])
    opts = dict(opts)
    feature_name = opts.get('--feature', 'sift')
    try:
        fn1, fn2 = args
    except:
        fn1 = 'object_01.jpg'
        # fn2 = 'box_in_scene.png'
    fn1 = 'object_01.jpg'
    img1 = cv.imread(cv.samples.findFile(fn1), cv.IMREAD_GRAYSCALE)
    img1 = cv2.resize(img1, (120, 120))
    # img2 = cv.imread(cv.samples.findFile(fn2), cv.IMREAD_GRAYSCALE)
    detector, matcher = init_feature(feature_name)

    if img1 is None:
        print('Failed to load fn1:', fn1)
        sys.exit(1)

    # if img2 is None:
    #     print('Failed to load fn2:', fn2)
    #     sys.exit(1)

    if detector is None:
        print('unknown feature:', feature_name)
        sys.exit(1)

    print('using', feature_name)

    kp1, desc1 = detector.detectAndCompute(img1, None)

    # def match_and_draw(win):
    #     raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)  # 2
    #     p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
    #     if len(p1) >= 4:
    #         H, status = cv.findHomography(p1, p2, cv.RANSAC, 5.0)
    #         # print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
    #     else:
    #         H, status = None, None
    #         print('%d matches found, not enough for homography estimation' % len(p1))
    #
    #     _vis = explore_match(win, img1, img2, kp_pairs, status, H)

    while True:
        try:
            ret, img2 = cap.read()
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            kp2, desc2 = detector.detectAndCompute(img2_gray, None)
            # print('img1 - %d features, img2 - %d features' % (len(kp1), len(kp2)))

            # match_and_draw('find_obj', source=img2)
            raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)  # 2
            p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
            if len(p1) >= 4:
                H, status = cv.findHomography(p1, p2, cv.RANSAC, 5.0)
                # print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
            else:
                H, status = None, None
                print('%d matches found, not enough for homography estimation' % len(p1))

            _vis = explore_match('find_obj', img1, img2, kp_pairs, status, H)
            # video_writer.write(_vis)
            k = cv.waitKey(1)
            if k == ord('q'):
                # video_writer.release()
                cap.release()
                exit(1)
        except:
            exit(1)


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
