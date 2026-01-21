import unittest
import numpy as np
import cv2
from Img_treatment import ImgTreatment
from image_to_board import (
    extract_img_markers_with_margin,
    get_cell_boxes,
    detect_colored_stickers,
    board_state_from_colored_stickers,
    create_virtual_detection_grid,
    visualize_real_and_virtual_grids,
    COLOR_RANGES
)

class TestImageToBoard(unittest.TestCase):
    def setUp(self):
            """Initialize camera and test parameters."""
            self.base_size = 800
            self.boxes = None
            self.warp = None
            self.centers = None
            self.virtual_boxes = None
            self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                raise RuntimeError("Impossible d'ouvrir la caméra. Vérifiez la connexion USB.")
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            # Capturer une seule image au démarrage
            cap = self.wait_for_space_and_capture()
            # Extraction markers
            self.img, M, dims, corners, margin_px = extract_img_markers_with_margin(
                cap,
                workspace_ratio=1.0,
                base_size=self.base_size,
            )
            
            if self.img is not None:
                self.boxes = get_cell_boxes(self.img, margin_px, player_plays_white=True)

            #  Récupérer les 4 coins détectés (image points) dans l'ordre [tl,tr,br,bl]
            tl = (self.boxes['a8'][0], self.boxes['a8'][1])
            tr = (self.boxes['h8'][2], self.boxes['h8'][1])
            br = (self.boxes['h1'][2], self.boxes['h1'][3])
            bl = (self.boxes['a1'][0], self.boxes['a1'][3])
            image_corners = np.array([tl, tr, br, bl], dtype=np.float64)

            real_corners_3d = np.array([
                [0.0, 0.0, 0.0],  # tl
                [240.0, 0.0, 0.0],  # tr
                [240.0, 240.0, 0.0], # br
                [0.0, 240.0, 0.0],  # bl
            ], dtype=np.float64)

            # Estimation simple des intrinsics à partir de la taille de l'image (aucune entrée manuelle requise)
            h, w = self.img.shape[:2]
            f_est = max(w, h)  # estimation du foyer (px). Simple et suffisant pour la visualisation.
            cx = w / 2.0
            cy = h / 2.0
            camera_matrix = np.array([[f_est, 0, cx], [0, f_est, cy], [0, 0, 1]], dtype=np.float64)
            dist_coeffs = np.zeros((5, 1), dtype=np.float64)

            # Créer la grille virtuelle projetée (hauteur réglable ici)
            self.virtual_boxes, _ = create_virtual_detection_grid(
                real_corners_3d,
                image_corners,
                camera_matrix,
                dist_coeffs=dist_coeffs,
                grid_size=8,
                plane_height_cm=22.0,   # hauteur de la grille virtuelle en cm (configurable)
                margin_percent=0.05
            )

            self.img_tr = ImgTreatment(self.img)
            self.warp, self.centers = self.img_tr.traitement_image()
            cv2.imshow("Warp", self.warp)

    def tearDown(self):
        """Release camera resources."""
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

    # def test_virtual_grid_visualization(self):
    #     """
    #     Test visuel : crée une grille virtuelle projetée et l'affiche avec la grille réelle.
    #     """
    #     if self.img is None or self.boxes is None:
    #         print("[ERROR] Impossible d'extraire les marqueurs")
    #         return
    #     # Visualiser la grille réelle (bleu) vs la grille virtuelle (rouge)
    #     visualize_real_and_virtual_grids(self.img, self.boxes, self.virtual_boxes)

    
# # ─────────────────────────────────────────────────────────────────
#     # A) CALIBRATION HSV - Outils de tuning des couleurs
#     # ─────────────────────────────────────────────────────────────────
    
    # def test_pick_hsv_color_from_cell(self):
    #     """
    #     ⭐ A — Outil interactif pour lire les valeurs HSV au clic.
        
    #     Instructions :
    #     1. Lance le test
    #     2. Une fenêtre affiche l'image warpée
    #     3. Clique sur une gommette pour afficher ses valeurs HSV
    #     4. Copie les H, S, V affichés dans la console
    #     5. Ajuste COLOR_RANGES en conséquence
    #     """
    #     if self.warp is None:
    #         print("[ERROR] Impossible d'extraire les marqueurs")
    #         return
        
    #     # Conversion en HSV
    #     hsv = cv2.cvtColor(self.warp, cv2.COLOR_BGR2HSV)
        
    #     clicked_colors = []
        
    #     def pick_color(event, x, y, flags, param):
    #         if event == cv2.EVENT_LBUTTONDOWN:
    #             h, s, v = hsv[y, x]
    #             clicked_colors.append((h, s, v))
    #             print(f"\n[COLOR PICKED] Pixel ({x}, {y})")
    #             print(f"  H: {int(h)}  (hue, 0-180 in OpenCV)")
    #             print(f"  S: {int(s)}  (saturation, 0-255)")
    #             print(f"  V: {int(v)}  (value/brightness, 0-255)")
    #             print(f"  → Copie : (({int(h)}, {int(s)}, {int(v)}), ({int(h)+10}, {int(s)}, {int(v)}))")
        
    #     # Afficher l'image avec les cell boxes
    #     display_img = self.warp.copy()
    #     for cell_name, (x1, y1, x2, y2) in self.boxes.items():
    #         cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), (200, 200, 200), 1)
    #         if cell_name in ['a8', 'a1', 'h8', 'h1']:
    #             cv2.putText(display_img, cell_name, (int(x1)+5, int(y1)+15),
    #                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
    #     cv2.imshow('HSV Color Picker - Click on stickers to read values', display_img)
    #     cv2.setMouseCallback('HSV Color Picker - Click on stickers to read values', pick_color)
        
    #     print("\n[INSTRUCTIONS]")
    #     print("1. Clique sur les gommettes pour lire leurs valeurs HSV")
    #     print("2. Regarde les valeurs dans la console")
    #     print("3. Appuie sur Q pour fermer")
    #     print()
        
    #     while True:
    #         key = cv2.waitKey(1) & 0xFF
    #         if key == ord('q'):
    #             cv2.destroyAllWindows()
    #             break
        
    #     print(f"\n[SUMMARY] {len(clicked_colors)} couleurs cliquées :")
    #     for i, (h, s, v) in enumerate(clicked_colors):
    #         print(f"  {i+1}. H={h}, S={s}, V={v}")

# # ─────────────────────────────────────────────────────────────────
# #     B) CORRECTION GAMMA - Test des différents facteurs
# #     ─────────────────────────────────────────────────────────────────
    
#     def test_gamma_correction_comparison(self):
#         """
#         ⭐ B — Compare différents facteurs gamma pour choisir le meilleur.
        
#         gamma < 1.0 → éclaircit
#         gamma > 1.0 → assombrit
#         gamma = 1.0 → pas de changement
#         """
#         if self.warp is None:
#             print("[ERROR] Impossible d'extraire les marqueurs")
#             return
        
#         gamma_values = [0.6, 0.8, 1.0, 1.2, 1.4]
        
#         print("\n[GAMMA CORRECTION TEST]")
#         print("Gamma < 1.0 éclaircit, > 1.0 assombrit")
#         print("Cherche celui où les gommettes sont les plus visibles.\n")
        
#         for gamma in gamma_values:
#             corrected = apply_gamma(self.warp, gamma)
#             title = f'Gamma = {gamma} - Press Q to next, Q again to exit'
            
#             Afficher avec boxes
#             display_img = corrected.copy()
#             for cell_name, (x1, y1, x2, y2) in self.boxes.items():
#                 cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            
#             cv2.imshow(title, display_img)
#             print(f"[VIEWING] Gamma = {gamma}")
            
#             while True:
#                 key = cv2.waitKey(1) & 0xFF
#                 if key == ord('q'):
#                     cv2.destroyWindow(title)
#                     break
        
#         print("\n[RECOMMENDATION]")
#         print("Choisissez le gamma qui rend les gommettes les plus contrastées.")
#         print("Mettez-le à jour dans votre test : detect_colored_stickers(..., gamma=YOUR_CHOICE)")

    # def test_detect_colored_stickers_in_mask(self):
    #     # 1) Corrections préliminaires
    #     img_work = self.warp.copy()
        
    #     # Conversion en HSV
    #     img_hsv = cv2.cvtColor(img_work, cv2.COLOR_BGR2HSV)
        
    #     # 3) Détection par case
    #     result = {}
    #     boxes = self.virtual_boxes
    #     taille_box = (boxes['a1'][2] - boxes['a1'][0])
        
    #     for cell_name, (x1, y1, x2, y2) in boxes.items():
    #         # Extraction ROI
    #         roi_hsv = img_hsv[y1:y2, x1:x2]

    #         best_color = None
    #         best_ratio = 0.0
    #         best_center = None
    #         best_area = 0  
    #         for color_name, ranges in COLOR_RANGES.items():
    #             mask = np.zeros(roi_hsv.shape[:2], dtype=np.uint8)
                
    #             # Union de toutes les sous-plages pour cette couleur
    #             for low, high in ranges:
    #                 sub_mask = cv2.inRange(roi_hsv, low, high)
    #                 mask = cv2.bitwise_or(mask, sub_mask)
                
    #             # Closing pour éliminer bruit
    #             cv2.imshow("mask_"+color_name, mask)
    #             cv2.waitKey(0)
    #             cv2.destroyWindow("mask_"+color_name)
                
    #             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #             if not contours:
    #                 continue

    #             cnt = max(contours, key=cv2.contourArea)
    #             area = cv2.contourArea(cnt)

    #             x, y, w, h = cv2.boundingRect(cnt)
    #             bbox_area = w * h
    #             if bbox_area == 0:
    #                 continue

    #             local_ratio = area / bbox_area

    #             if local_ratio >= 2 / 100.0 and local_ratio > best_ratio:
    #                 M = cv2.moments(cnt)
    #                 if M["m00"] != 0:
    #                     cx = int(M["m10"] / M["m00"])
    #                     cy = int(M["m01"] / M["m00"])
    #                 else:
    #                     cx = cy = None

    #                 best_color = color_name
    #                 best_ratio = local_ratio
    #                 best_center = (cx, cy)
    #                 best_area = area

    #             print("TAILLE_BOX:",best_ratio)    


    def test_detect_colored_stickers_default_params(self):
            """Test la détection de gommettes avec paramètres par défaut."""
            print("\n[STICKER DETECTION - Default Parameters]")
            
            detections = detect_colored_stickers(
                self.warp,
                self.virtual_boxes,
                min_area_percent=50,
                color_ranges=COLOR_RANGES 
            )
            
            # print("TAILLE D'UNE CASE PROJETEE:", taille_box)
            self.assertIsNotNone(detections)
            self.assertEqual(len(detections), 64)

            
            
            # Count detected colors
            detected_colors = {}
            for cell_name, info in detections.items():
                color = info.get("color")
                if color is not None:
                    detected_colors[color] = detected_colors.get(color, 0) + 1
            
            print(f"[RESULTS] Gommettes détectées par couleur :")
            for color, count in detected_colors.items():
                print(f"  {color}: {count}")
            
            # Visualize results
            vis_img = self.warp.copy()
            files = 'abcdefgh'
            ranks = '87654321'
            
            color_to_bgr = {
                "red": (0, 0, 255),
                "orange": (0, 140, 255),
                "yellow": (0, 255, 255),
                "green": (0, 255, 0),
                "purple": (255, 0, 255),
                "blue": (255, 0, 0),
            }
            
            for cell_name, info in detections.items():
                if info["color"] is None:
                    continue
                
                x1, y1, x2, y2 = self.boxes[cell_name]
                color = color_to_bgr.get(info["color"], (128, 128, 128))
                confidence = info["confidence"]
                piece_side = info["side"]
                letter = board_state_from_colored_stickers(info["color"], piece_side)
                
                # Draw rectangle with color
                cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Draw
                cv2.putText(vis_img, f"{letter}",
                        (int(x1)+15, int(y1)+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
            cv2.namedWindow('Colored Stickers Detection - Press Q to close', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Colored Stickers Detection - Press Q to close', 1000, 800)
            cv2.imshow('Colored Stickers Detection - Press Q to close', vis_img)
            print("[VIEWING] Résultat de détection...")
            
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cv2.destroyWindow('Colored Stickers Detection - Press Q to close')
                    break

if __name__ == '__main__':
    unittest.main()