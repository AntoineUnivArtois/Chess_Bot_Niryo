import numpy as np
import cv2
from .hough_circle_parameter_tuner import HoughCircleParameterTuner

class ImgTreatment:
    def __init__(self, image):
        self.warp = image
            
    def traitement_image(self):
        """Test interactif de Canny + Hough avec trackbars."""
        if self.warp is None:
            self.skipTest("warp not available")

        lab = cv2.cvtColor(self.warp, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)

        # ENLEVER LE FOND
        A = cv2.GaussianBlur(A, (5, 5), 0)
        B = cv2.GaussianBlur(B, (5, 5), 0)
        A_vert = 255-A
        A_vert = cv2.GaussianBlur(A_vert, (5, 5), 0)
        background_B = 255-B 
        background_B = cv2.GaussianBlur(background_B, (5, 5), 0)
        _, white_mask_1 = cv2.threshold(L, 180, 255, cv2.THRESH_BINARY)
        _, white_mask_2 = cv2.threshold(B, 150, 255, cv2.THRESH_BINARY)
        white_mask_1 = cv2.bitwise_not(white_mask_1)
        white_mask = cv2.bitwise_or(white_mask_1, white_mask_2)
        _, mask_gommettes_1 = cv2.threshold(B, 180, 255, cv2.THRESH_BINARY)
        _, mask_gommettes_2 = cv2.threshold(A, 140, 255, cv2.THRESH_BINARY)
        _, mask_vert = cv2.threshold(A_vert, 130, 255, cv2.THRESH_BINARY)
        _, mask_background = cv2.threshold(background_B, 150, 255, cv2.THRESH_BINARY)
        mask_gommettes = cv2.bitwise_or(mask_gommettes_1, mask_gommettes_2)
        mask_gommettes = cv2.bitwise_or(mask_gommettes, mask_vert)
        # mask = cv2.bitwise_or(mask_gommettes, mask_background)
        mask = cv2.bitwise_and(mask_gommettes, white_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # combler trous
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # enlever bruit

        masked_warp = cv2.bitwise_and(self.warp, self.warp, mask=mask)

        lab = cv2.cvtColor(masked_warp, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)


        A = cv2.GaussianBlur(A, (5, 5), 0)
        B = cv2.GaussianBlur(B, (5, 5), 0)
        chrom = cv2.absdiff(A, 128) + cv2.absdiff(B, 128)
        chrom = cv2.convertScaleAbs(chrom, alpha=2.5)

        
        tuner = HoughCircleParameterTuner(chrom)
        circles = tuner.detect_circular_objects()

        ## CREATION DU MASQUE DES CERCLE ET RECUPERATION DES CENTRES
        centers = []
        h, w = self.warp.shape[:2]
        circle_mask = np.zeros((h, w), dtype=np.uint8)
        if circles is not None:
            for c in circles:
                center = c["center"]
                centers.append(center)
                radius = c["radius"]
                cv2.circle(circle_mask, center, radius, (255, 255, 255), -1)

        ## VISUALISATION DES CENTRES
        # for x, y in centers:
        #     cv2.circle(self.warp, (x, y), 3, (0, 0, 255), -1)

        # cv2.namedWindow('Centres gommettes - Press Q to Exit', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Centres gommettes - Press Q to Exit', 800, 600)

        # cv2.imshow('Centres gommettes - Press Q to Exit',self.warp)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        final_img = cv2.bitwise_and(self.warp, self.warp, mask=circle_mask)

        # cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Mask', 1200, 1000)

        # cv2.imshow('Mask',chrom)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return final_img, centers