import cv2
import numpy as np

class HoughCircleParameterTuner:
    """Widget interactif pour ajuster les paramètres de Hough."""
    
    def __init__(self, edge_image: np.ndarray):
        self.edge_image = edge_image
        self.min_dist = 60
        self.param1 = 26
        self.param2 = 14
        self.min_radius = 10
        self.max_radius = 14
        
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

    def detect_circles_hough(self,edge_image: np.ndarray, min_dist: int, param1: int, param2: int, min_radius: int, max_radius: int):
        """
        Détecte les cercles via la Transformée de Hough.
        
        Args:
            edge_image: Image des contours (résultat de Canny)
            min_dist: Distance minimale entre les centres des cercles
            param1: Seuil supérieur pour Canny (si recalculé)
            param2: Seuil d'accumulation
            min_radius: Rayon minimum
            max_radius: Rayon maximum
        
        Returns:
            circles: Array de shape (1, N, 3) avec (cx, cy, radius) ou None
        """
        circles = cv2.HoughCircles(
            edge_image,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min_dist,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        return circles
    
    def detect_circular_objects(self, min_area=140, max_area=700, min_circularity=0.6):
        """
        Détecte des objets circulaires dans un masque binaire
        Retourne une liste de cercles
        """
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        contours, _ = cv2.findContours(cv2.threshold(self.edge_image, 10, 255, cv2.THRESH_BINARY)[1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(f"Found {len(contours)} contours")

        circles = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < min_circularity:
                continue

            (x, y), radius = cv2.minEnclosingCircle(cnt)

            circles.append({
                "center": (int(x), int(y)),
                "radius": int(radius),
                "area": area,
                "circularity": circularity
            })

        display = cv2.cvtColor(cv2.threshold(self.edge_image, 10, 255, cv2.THRESH_BINARY)[1], cv2.COLOR_GRAY2BGR)
            
        if circles is not None:
            for c in circles:
                center = c["center"]
                radius = c["radius"]
                cv2.circle(display, center, radius, (0, 255, 0), 2)
                cv2.circle(display, center, 2, (0, 0, 255), 3)
        
        # cv2.imshow('Hough Circle Detection', display)
        
        key = cv2.waitKey(30) & 0xFF

        return circles
    