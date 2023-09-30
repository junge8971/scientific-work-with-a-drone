import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from stitching import Stitcher


def matching_points_with_ORB(img1: np.ndarray, img2: np.ndarray):
    # Initiate ORB detector
    orb = cv.ORB_create()

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches.
    img3 = cv.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        matches[:10],
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    plt.imshow(img3), plt.show()


def matching_points_with_SIFT(img1: np.ndarray, img2: np.ndarray):
    # Initiate SIFT detector
    sift = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.1 * n.distance:
            good.append([m])

    print(good)
    print(type(good))
    print(good[0][0])
    print(type(good[0][0]))

    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(
        img1,
        kp1,
        img2,
        kp2,
        good,
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    plt.imshow(img3), plt.show()


def matching_points_with_FLANN(img1: np.ndarray, img2: np.ndarray):
    # Initiate SIFT detector
    sift = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.1 * n.distance:
            matchesMask[i] = [1, 0]

    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        matchesMask=matchesMask,
        flags=cv.DrawMatchesFlags_DEFAULT,
    )
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    plt.imshow(
        img3,
    ), plt.show()


def stiching_img_with_build_in_funtions(img1: np.ndarray, img2: np.ndarray):
    """https://www.geeksforgeeks.org/opencv-panorama-stitching/"""
    # for img in imgs:
    #    img = cv.resize(img, (0,0), fx=0.4, fy=0.4)

    stitchy = cv.Stitcher.create(cv.Stitcher_PANORAMA)
    print(f"{stitchy=}")
    (dummy, output) = stitchy.stitch(img1, img2)
    print(f"{dummy=} {output=}")

    if dummy != cv.STITCHER_OK:
        # checking if the stitching procedure is successful
        # .stitch() function returns a true value if stitching is
        # done successfully
        print("stitching ain't successful")

    else:
        # final output
        cv.imshow("1", img1)
        cv.imshow("2", img2)
        cv.imshow("final result", output)
        cv.imwrite("1_and_2.TIF", output)

        cv.waitKey(0)
        print("Your Panorama is ready!!!")


def test_affine_stitcher_budapest():
    """https://github.com/OpenStitching/stitching/"""
    settings = {
        "detector": "sift",
        "confidence_threshold": 0.1,
    }

    stitcher = Stitcher(**settings)
    imgs = ["budapest?.jpg"]
    max_derivation = 50
    expected_shape = (1155, 2310)
    name = "budapest"
    images1 = [
        "budapest1.jpg",
        "budapest2.jpg",
        "budapest3.jpg",
        "budapest4.jpg",
        "budapest5.jpg",
        "budapest6.jpg",
    ]
    images2 = ["1.TIF", "2.TIF"]
    result = stitcher.stitch(
        images=images2,
    )
    cv.imwrite("budapest.jpg", result)
    # self.stitch_test(stitcher, imgs, expected_shape, max_derivation, name)
    print(result)


if __name__ == "__main__":
    img1 = cv.imread(
        r"s1.jpg",
    )  # queryImage
    img2 = cv.imread(
        r"s2.jpg",
    )  # trainImage

    # matching_points_with_ORB(img1=img1, img2=img2)
    # matching_points_with_SIFT(img1=img1, img2=img2)
    # matching_points_with_FLANN(img1=img1, img2=img2)
    # stiching_img_with_build_in_funtions(img1, img2)
    # test_affine_stitcher_budapest()
