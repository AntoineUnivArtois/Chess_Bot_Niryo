import cv2
import unittest
import numpy as np
import matplotlib.pyplot as plt
from image_to_board import (extract_img_markers_with_margin, get_cell_boxes,detect_circles_hough)

class HoughCircleParameterTuner:
    """Widget interactif pour ajuster les paramètres de Hough."""
    
    def __init__(self, edge_image: np.ndarray):
        self.edge_image = edge_image
        self.min_dist = 30
        self.param1 = 30
        self.param2 = 18
        self.min_radius = 5
        self.max_radius = 20
        
    def create_trackbars(self):
        """Crée les trackbars OpenCV."""
        cv2.namedWindow('Hough Circle Parameters')
        cv2.createTrackbar('min_dist', 'Hough Circle Parameters', self.min_dist, 300, self._update_min_dist)
        cv2.createTrackbar('param1', 'Hough Circle Parameters', self.param1, 200, self._update_param1)
        cv2.createTrackbar('param2', 'Hough Circle Parameters', self.param2, 60, self._update_param2)
        cv2.createTrackbar('min_radius', 'Hough Circle Parameters', self.min_radius, 50, self._update_min_radius)
        cv2.createTrackbar('max_radius', 'Hough Circle Parameters', self.max_radius, 80, self._update_max_radius)
    
    def _update_min_dist(self, val):
        self.min_dist = max(1, val)
    
    def _update_param1(self, val):
        self.param1 = max(1, val)
    
    def _update_param2(self, val):
        self.param2 = max(1, val)
    
    def _update_min_radius(self, val):
        self.min_radius = val
    
    def _update_max_radius(self, val):
        self.max_radius = val

    def detect(self):
        """Détecte les cercles avec les paramètres actuels."""
        circles = detect_circles_hough(
            self.edge_image,
            min_dist=self.min_dist,
            param1=self.param1,
            param2=self.param2,
            min_radius=self.min_radius,
            max_radius=self.max_radius
        )
        return circles
    
    def detect_and_display(self):
        """Détecte et affiche les cercles en temps réel."""
        while True:
            circles = detect_circles_hough(
                self.edge_image,
                min_dist=self.min_dist,
                param1=self.param1,
                param2=self.param2,
                min_radius=self.min_radius,
                max_radius=self.max_radius
            )
            
            display = cv2.cvtColor(self.edge_image, cv2.COLOR_GRAY2BGR)
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    center = (i[0], i[1])
                    radius = i[2]
                    cv2.circle(display, center, radius, (0, 255, 0), 2)
                    cv2.circle(display, center, 2, (0, 0, 255), 3)
            
            cv2.imshow('Hough Circle Detection', display)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        return self.min_dist, self.param1, self.param2, self.min_radius, self.max_radius
    

class TestImageToBoardInteractive(unittest.TestCase):

    def setUp(self):
            """Initialize camera and capture."""
            self.base_size = 800
            self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                raise RuntimeError("Impossible d'ouvrir la caméra.")
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            self.frame = self.wait_for_space_and_capture()
            self.warp = None
            self.boxes = None
            
            # Extraction markers
            warp, M, dims, corners, margin_px = extract_img_markers_with_margin(
                self.frame,
                workspace_ratio=1.0,
                base_size=self.base_size,
            )
            
            if warp is not None:
                self.warp = warp
                self.boxes = get_cell_boxes(warp, margin_px, player_plays_white=True)

    def tearDown(self):
        self.cap.release()
        cv2.destroyAllWindows()


    def wait_for_space_and_capture(self):
            """Capture depuis la caméra."""
            print("\n[INFO] Positionnez l'échiquier avec les 4 marqueurs visibles.")
            print("[INFO] Appuyez sur ESPACE pour capturer...")
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    raise RuntimeError("Impossible de capturer.")
                
                cv2.imshow('Camera Feed - Press SPACE', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 32:  # SPACE
                    cv2.destroyAllWindows()
                    return frame
                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    raise RuntimeError("Capture annulée.")
                
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
        _, mask_gommettes_1 = cv2.threshold(B, 150, 255, cv2.THRESH_BINARY)
        _, mask_gommettes_2 = cv2.threshold(A, 140, 255, cv2.THRESH_BINARY)
        _, mask_vert = cv2.threshold(A_vert, 130, 255, cv2.THRESH_BINARY)
        _, mask_background = cv2.threshold(background_B, 120, 255, cv2.THRESH_BINARY)
        mask_gommettes_1 = cv2.bitwise_or(mask_gommettes_1, mask_vert)
        mask_gommettes = cv2.bitwise_or(mask_gommettes_1, mask_gommettes_2)
        mask = cv2.bitwise_or(mask_gommettes, mask_background)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # combler trous
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # enlever bruit

        masked_warp = cv2.bitwise_and(self.warp, self.warp, mask=mask)

        lab = cv2.cvtColor(masked_warp, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)


        A = cv2.GaussianBlur(A, (5, 5), 0)
        B = cv2.GaussianBlur(B, (5, 5), 0)
        chrom = cv2.absdiff(A, 128) + cv2.absdiff(B, 128)
        
        tuner = HoughCircleParameterTuner(chrom)
        circles = tuner.detect()

        ## CREATION DU MASQUE DES CERCLE ET RECUPERATION DES CENTRES
        centers = []
        h, w = self.warp.shape[:2]
        circle_mask = np.zeros((h, w), dtype=np.uint8)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                centers.append(center)
                radius = i[2]
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

        # cv2.namedWindow('Centres gommettes - Press Q to Exit', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Centres gommettes - Press Q to Exit', 800, 600)

        # cv2.imshow('Centres gommettes - Press Q to Exit',final_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return final_img, centers
        



if __name__ == '__main__':
    unittest.main()