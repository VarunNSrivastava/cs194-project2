# CS194-26 (CS294-26): Project 2
# author: Varun Neal Srivastava

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as skio
from scipy.signal import convolve2d
from skimage.filters import try_all_threshold
from skimage.util import montage
import skimage as sk
import cv2
import os

matplotlib.use('TkAgg')
skio.use_plugin('matplotlib')


def gkern(kernlen, std=None):
    """Returns a 2D Gaussian kernel array."""
    if not std:
        std = 0.3 * ((kernlen - 1) * 0.5 - 1) + 0.8
    gkern1d = cv2.getGaussianKernel(kernlen, std)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


def save(img, name):
    skio.imsave(os.path.join("out", f"{name}.jpeg"), img)


def part_one():
    cameraman = skio.imread("lib/cameraman.png", as_gray=True)

    # Part 1.1
    D_x = [[1, -1]]
    D_y = [[1], [-1]]
    dx_cameraman = convolve2d(cameraman, D_x, mode='same')
    dy_cameraman = convolve2d(cameraman, D_y, mode='same')
    grad_mag = np.sqrt(np.square(dx_cameraman) + np.square(dy_cameraman))
    threshold = 0.25
    binarized = grad_mag > threshold
    save(dx_cameraman, "pt1-1_dx")
    save(dy_cameraman, "pt1-1_dy")
    save(grad_mag, "pt1-1_grad")
    save(binarized, "pt1-1_binarized")

    # Part 1.2
    G = gkern(5)
    g_cameraman = convolve2d(cameraman, G, mode='same')
    dx_g_cameraman = convolve2d(g_cameraman, D_x, mode='same')
    dy_g_cameraman = convolve2d(g_cameraman, D_y, mode='same')
    g_grad_mag = np.sqrt(np.square(dx_g_cameraman) + np.square(dy_g_cameraman))
    threshold = 0.09
    binarized = g_grad_mag > threshold
    save(dx_g_cameraman, "pt1-2_dx")
    save(dy_g_cameraman, "pt1-2_dy")
    save(g_grad_mag, "pt1-2_grad")
    save(binarized, "pt1-2_binarized")

    g_dx_cameraman = convolve2d(dx_cameraman, G, mode='same')
    g_dy_cameraman = convolve2d(dy_cameraman, G, mode='same')
    g_grad_mag = np.sqrt(np.square(g_dx_cameraman) + np.square(g_dy_cameraman))
    threshold = 0.09
    binarized = g_grad_mag > threshold
    save(binarized, "pt1-2_binarized_flipped")


def blur(img, amount, std=None):
    G = gkern(amount, std)
    if len(img.shape) == 2:
        return convolve2d(img, G, mode='same')
    r = convolve2d(img[:, :, 0], G, mode='same')
    g = convolve2d(img[:, :, 1], G, mode='same')
    b = convolve2d(img[:, :, 2], G, mode='same')

    return normalize(np.dstack([r, g, b]), hard=True)


def normalize(img, hard=False):
    if hard:
        return (img - np.min(img)) / (np.max(img) - np.min(img))
    if np.mean(img) > 1:
        return img / 256
    else:
        return img


def sharpen(img, amnt):
    blurred_img = blur(img, amnt)
    hifi_img = normalize(img, hard=True) - blurred_img
    sharp = img + hifi_img
    return normalize(sharp)


def part_two_one():
    taj = skio.imread("lib/taj.jpeg")
    sharpened_taj = sharpen(taj, 9)
    save(sharpened_taj, "pt2-1_sharpened_taj")

    zyzz = skio.imread("lib/zyzz.jpeg")
    sharpened_zyzz = sharpen(zyzz, 9)
    save(sharpened_zyzz, "pt2-1_sharpened_zyzz")

    sophia = normalize(skio.imread("lib/sophia.jpeg"))
    blurred_sophia = blur(sophia, 9)
    sharpened_blurred_sophia = sharpen(blurred_sophia, 11)

    save(sophia, "pt2-1_sophia")
    save(blurred_sophia, "pt2-1_blurred_sophia")
    save(sharpened_blurred_sophia, "pt2-1_sharpened_blurred_sophia")


def hybrid(img1, img2, sigma1, sigma2):
    blurry_img1 = normalize(blur(img1, sigma1), hard=True)
    hifi_img2 = normalize(img2 - blur(img2, sigma2), hard=True)
    return normalize(blurry_img1 + hifi_img2, hard=True)


def fft_img(img):
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(img))))


def part_two_two():
    cat = skio.imread("lib/nutmeg_aligned.jpeg", as_gray=True)
    boy = skio.imread("lib/derek_aligned.jpeg", as_gray=True)
    catboy = hybrid(boy, cat, 50, 50)
    save(catboy, "pt2-2_catboy")

    anakin = skio.imread("lib/anakin.jpeg", as_gray=True)
    vader = skio.imread("lib/vader.jpeg", as_gray=True)
    anakinvader = hybrid(anakin, vader, 20, 10)
    save(anakinvader, "pt2-2_anakinvader")

    lighthouse = skio.imread("lib/lighthouse.jpeg", as_gray=True)
    pencil = skio.imread("lib/pencil.jpeg", as_gray=True)
    lightpencil = hybrid(lighthouse, pencil, 20, 20)
    save(lightpencil, "pt2-2_lightpencil")

    mike = skio.imread("lib/mike_aligned.jpeg", as_gray=True)
    finger = skio.imread("lib/finger_aligned.jpeg", as_gray=True)
    kid_named_finger = hybrid(mike, finger, 20, 20)
    save(kid_named_finger, "pt2-2_kidnamedfinger")

    mike_fft = fft_img(mike)
    save(mike_fft, "pt2-2_mikefft")
    finger_fft = fft_img(finger)
    save(finger_fft, "pt2-2_fingerfft")
    blur_mike = normalize(blur(mike, 20), hard=True)
    hifi_finger = normalize(finger - blur(finger, 20), hard=True)
    blur_mike_fft = fft_img(blur_mike)
    save(blur_mike_fft, "pt2-2_blurmikefft")
    hifi_finger_fft = fft_img(hifi_finger)
    save(hifi_finger_fft, "pt2-2_hififingerfft")
    kid_named_finger_fft = fft_img(kid_named_finger)
    save(kid_named_finger_fft, "pt2-2_kidnamedfingerfft")


def stacks(img, sigma, N):
    gaussian = []
    for i in range(N):
        gaussian.append(img)
        img = blur(img, sigma)
    laplacian = []
    for i in range(N - 1):
        laplacian.append(gaussian[i] - gaussian[i+1])
    laplacian.append(gaussian[N - 1])
    return gaussian, laplacian


def blend(A, B, sigma, N, mask=None, s=False):
    if mask is None:
        # vertical spline
        mask = np.ones_like(A)
        mask[:, :A.shape[1] // 2] = 0
        mask = normalize(blur(mask, 3*sigma, std=2*sigma), hard=True)
    _, LA = stacks(A, sigma, N)
    _, LB = stacks(B, sigma, N)
    GR, _ = stacks(mask, sigma, N)
    LS = []
    for i in range(N):
        L_i = GR[i] * LA[i]
        R_i = (1 - GR[i]) * LB[i]
        LS_i = normalize(L_i + R_i, hard=True)
        if s:
            save(L_i, f"pt2-3_orange{i}")
            save(R_i, f"pt2-3_apple{i}")
            save(LS_i, f"pt2-3_oraple{i}")
        if i == 0:
            LS.append(LS_i)
        else:
            LS.append(LS_i)
            #LS_sum = normalize(LS_i + LS[i - 1], hard=True)
            # LS.append(LS_sum)
        # if s:
        #     save(LS[i], f"pt2-3_layer{i}")
    collapsed = normalize(sum(LS), hard=True)
    # if s:
    #     save(collapsed, f"pt2-4_pmarcegg_collapsed")
    return sum(LS)


def part_two_three():
    orange = normalize(skio.imread("lib/orange.jpeg", as_gray=True), hard=True)
    apple = normalize(skio.imread("lib/apple.jpeg", as_gray=True), hard=True)
    oraple = blend(orange, apple, 20, 4, s=True)

def part_two_four():
    pmarca = normalize(skio.imread("lib/pmarca_aligned.jpeg", as_gray=True), hard=True)
    egg = normalize(skio.imread("lib/egg_aligned.jpeg", as_gray=True), hard=True)
    pmarcegg = blend(pmarca, egg, 20, 4, s=True)
    save(pmarcegg, "pt2-4_pmarcegg")

    joker = normalize(skio.imread("lib/joker_aligned.jpeg", as_gray=True), hard=True)
    bateman = normalize(skio.imread("lib/bateman_aligned.jpeg", as_gray=True), hard=True)
    jokeman = blend(joker, bateman, 20, 4)
    save(jokeman, "pt2-4_jokeman")

    burning_ship = normalize(skio.imread("lib/burning_ship.jpeg", as_gray=True), hard=True)
    titanic = normalize(skio.imread("lib/titanic.jpeg", as_gray=True), hard=True)
    mask = skio.imread("lib/titanic_mask.jpeg", as_gray=True)
    burning_titanic = blend(titanic, burning_ship, 20, 4, mask=mask)
    save(burning_titanic, "pt2-4_burning_titanic")


def main():
    part_one()
    part_two_one()
    part_two_two()
    part_two_three()
    part_two_four()


if __name__ == '__main__':
    main()
