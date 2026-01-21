import unittest
import numpy as np
import cv2
from Img_treatment import ImgTreatment
from image_to_board import (
    extract_img_markers_with_margin,
    get_cell_boxes,
    detect_colored_stickers,
    board_state_from_colored_stickers,
    apply_gamma,
    gray_world_wb,
    create_virtual_detection_grid,
    visualize_real_and_virtual_grids,
    COLOR_RANGES,
    apply_clahe_lab,
    detect_colored_stickers_robust,
    detect_color_adaptive,
    adapt_color_ranges_from_samples
)

class TestImageToBoard(unittest.TestCase):
    
    def setUp(self):
        """Initialize camera and test parameters."""
        self.base_size = 800
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            raise RuntimeError("Impossible d'ouvrir la caméra. Vérifiez la connexion USB.")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Capturer une seule image au démarrage
        self.frame = self.wait_for_space_and_capture()
        self.warp = None
        self.boxes = None
        self.margin_px = None
        self.virtual_boxes = None
        
        # Extraire les marqueurs
        warp, M, dims, corners, margin_px = extract_img_markers_with_margin(
            self.frame,
            workspace_ratio=1.0,
            base_size=self.base_size,
        )

        
        if warp is not None:
            self.warp = warp
            self.boxes = get_cell_boxes(warp, margin_px, player_plays_white=True)
            self.margin_px = margin_px

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
        h, w = self.warp.shape[:2]
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


    def tearDown(self):
        """Release camera resources."""
        self.cap.release()
        cv2.destroyAllWindows()
    
    # # ─────────────────────────────────────────────────────────────────
    # # A) CALIBRATION HSV - Outils de tuning des couleurs
    # # ─────────────────────────────────────────────────────────────────
    
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
    # # B) CORRECTION GAMMA - Test des différents facteurs
    # # ─────────────────────────────────────────────────────────────────
    
    # def test_gamma_correction_comparison(self):
    #     """
    #     ⭐ B — Compare différents facteurs gamma pour choisir le meilleur.
        
    #     gamma < 1.0 → éclaircit
    #     gamma > 1.0 → assombrit
    #     gamma = 1.0 → pas de changement
    #     """
    #     if self.warp is None:
    #         print("[ERROR] Impossible d'extraire les marqueurs")
    #         return
        
    #     gamma_values = [0.6, 0.8, 1.0, 1.2, 1.4]
        
    #     print("\n[GAMMA CORRECTION TEST]")
    #     print("Gamma < 1.0 éclaircit, > 1.0 assombrit")
    #     print("Cherche celui où les gommettes sont les plus visibles.\n")
        
    #     for gamma in gamma_values:
    #         corrected = apply_gamma(self.warp, gamma)
    #         title = f'Gamma = {gamma} - Press Q to next, Q again to exit'
            
    #         # Afficher avec boxes
    #         display_img = corrected.copy()
    #         for cell_name, (x1, y1, x2, y2) in self.boxes.items():
    #             cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            
    #         cv2.imshow(title, display_img)
    #         print(f"[VIEWING] Gamma = {gamma}")
            
    #         while True:
    #             key = cv2.waitKey(1) & 0xFF
    #             if key == ord('q'):
    #                 cv2.destroyWindow(title)
    #                 break
        
    #     print("\n[RECOMMENDATION]")
    #     print("Choisissez le gamma qui rend les gommettes les plus contrastées.")
    #     print("Mettez-le à jour dans votre test : detect_colored_stickers(..., gamma=YOUR_CHOICE)")
    
    # # ─────────────────────────────────────────────────────────────────
    # # C) GRAY WORLD WHITE BALANCE
    # # ─────────────────────────────────────────────────────────────────
    
    # def test_white_balance_comparison(self):
    #     """
    #     ⭐ C — Compare avant/après "gray world" white balance.
        
    #     Utile si l'éclairage est jaune (néon) ou bleu (jour).
    #     """
    #     if self.warp is None:
    #         print("[ERROR] Impossible d'extraire les marqueurs")
    #         return
        
    #     print("\n[WHITE BALANCE TEST]")
        
    #     # Sans white balance
    #     display_img_without = self.warp.copy()
    #     for cell_name, (x1, y1, x2, y2) in self.boxes.items():
    #         cv2.rectangle(display_img_without, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
        
    #     cv2.imshow('WITHOUT White Balance - Press Q to next', display_img_without)
    #     print("[VIEWING] Sans white balance...")
        
    #     while True:
    #         key = cv2.waitKey(1) & 0xFF
    #         if key == ord('q'):
    #             cv2.destroyWindow('WITHOUT White Balance - Press Q to next')
    #             break
        
    #     # Avec white balance
    #     balanced = gray_world_wb(self.warp)
    #     display_img_with = balanced.copy()
    #     for cell_name, (x1, y1, x2, y2) in self.boxes.items():
    #         cv2.rectangle(display_img_with, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
        
    #     cv2.imshow('WITH White Balance - Press Q to close', display_img_with)
    #     print("[VIEWING] Avec white balance (gray world)...")
        
    #     while True:
    #         key = cv2.waitKey(1) & 0xFF
    #         if key == ord('q'):
    #             cv2.destroyWindow('WITH White Balance - Press Q to close')
    #             break
        
    #     print("\n[DECISION]")
    #     print("Si les couleurs sont plus stables AVEC, activez :")
    #     print("  apply_white_balance=True")
    
    # ─────────────────────────────────────────────────────────────────
    # D) CALIBRATION GÉOMÉTRIQUE - Vérifier tailles et marges
    # ─────────────────────────────────────────────────────────────────
    
    # def test_geometric_calibration(self):
    #     """
    #     ⭐ D — Vérifie que la géométrie est correcte.
        
    #     Affiche :
    #     1. Taille des cases en pixels
    #     2. Superposition des boxes sur l'image
    #     3. Marges pour vérifier qu'elles ne coupent pas les gommettes
    #     """
    #     if self.warp is None or self.boxes is None:
    #         print("[ERROR] Impossible d'extraire les marqueurs")
    #         return
        
    #     print("\n[GEOMETRIC CALIBRATION]")
    #     print(f"  Base size: {self.base_size} px")
    #     print(f"  Margin: {self.margin_px} px")
        
    #     # Calcul des dimensions des cases
    #     cell_w = self.base_size/ 8.0 
    #     cell_h = self.base_size / 8.0
        
    #     print(f"  Cell width: {cell_w:.2f} px")
    #     print(f"  Cell height: {cell_h:.2f} px")
    #     print()
        
    #     # Afficher l'image avec les boxes
    #     display_img = self.warp.copy()
        
    #     # Dessiner toutes les cases
    #     for cell_name, (x1, y1, x2, y2) in self.boxes.items():
    #         # Couleur différente selon la case
    #         color = (0, 255, 0) if cell_name in ['a1', 'h1', 'a8', 'h8'] else (100, 100, 100)
    #         thickness = 2 if cell_name in ['a1', 'h1', 'a8', 'h8'] else 1
    #         cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            
    #         # Label quelques cases
    #         if cell_name in ['a8', 'a1', 'h8', 'h1', 'e4']:
    #             cv2.putText(display_img, cell_name, (int(x1)+5, int(y1)+15),
    #                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
    #     # Dessiner les limites de la marge
    #     cv2.rectangle(display_img,
    #                  (self.margin_px, self.margin_px),
    #                  (self.margin_px + self.base_size, self.margin_px + self.base_size),
    #                  (255, 0, 0), 2)
        
    #     cv2.imshow('Geometric Calibration - Boxes overlay - Press Q to close', display_img)
    #     print("[VIEWING] Boîtes des cases...")
    #     print("[CHECK] Les boîtes s'alignent-elles bien sur l'échiquier ?")
        
    #     while True:
    #         key = cv2.waitKey(1) & 0xFF
    #         if key == ord('q'):
    #             cv2.destroyWindow('Geometric Calibration - Boxes overlay - Press Q to close')
    #             break
        
    #     print("\n[RECOMMENDATION]")
    #     print("Si les gommettes touchent les bords, augmentez margin_cells :")
    #     print("  extract_img_markers_with_margin(..., margin_cells=2)")
    
    # ─────────────────────────────────────────────────────────────────
    # TESTS FONCTIONNELS - Détection de gommettes
    # ─────────────────────────────────────────────────────────────────
    
    def test_detect_colored_stickers_default_params(self):
        """Test la détection de gommettes avec paramètres par défaut."""
        if self.warp is None or self.boxes is None:
            print("[ERROR] Impossible d'extraire les marqueurs")
            return
        print("\n[STICKER DETECTION - Default Parameters]")

        # img_p = apply_gamma(self.warp, 1.5)
        # img_p = apply_clahe_lab(img_p, clip_limit=2.0, tile_size=8)
        # img_p = cv2.GaussianBlur(img_p, (5, 5), 1.0)
        # img_hsv = cv2.cvtColor(img_p, cv2.COLOR_BGR2HSV)

        # sample_rois = self.sample_roi(img_p,img_hsv)

        # adapted_color_ranges= adapt_color_ranges_from_samples(img_hsv, sample_rois, COLOR_RANGES)
        
        detections = detect_colored_stickers(
            self.warp,
            self.virtual_boxes,
            color_ranges=COLOR_RANGES, #adapted_color_ranges
            apply_gamma_correction=True,
            gamma=1.6,
            apply_white_balance=False,
            apply_wood_mask=True,
            min_area_percent=2,
            use_robust=True
        )
        
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
            
            # Draw rectangle with color
            cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw confidence
            cv2.putText(vis_img, f"{info['color'][:1]} {confidence:.0%}",
                       (int(x1)+5, int(y1)+15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        cv2.imshow('Colored Stickers Detection - Press Q to close', vis_img)
        print("[VIEWING] Résultat de détection...")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyWindow('Colored Stickers Detection - Press Q to close')
                break
    
    # def test_detect_colored_stickers_with_custom_params(self):
    #     """Test la détection avec paramètres personnalisés."""
    #     if self.warp is None or self.boxes is None:
    #         print("[ERROR] Impossible d'extraire les marqueurs")
    #         return
        
    #     print("\n[STICKER DETECTION - Custom Parameters]")
    #     print("Testez différentes combinaisons de paramètres :\n")
        
    #     configs = [
    #         {
    #             "name": "Brillant (gamma=0.6)",
    #             "gamma": 0.6,
    #             "white_balance": False,
    #             "wood_mask": True,
    #         },
    #         {
    #             "name": "Normal (gamma=0.8)",
    #             "gamma": 0.8,
    #             "white_balance": False,
    #             "wood_mask": True,
    #         },
    #         {
    #             "name": "Sombre (gamma=1.2)",
    #             "gamma": 1.2,
    #             "white_balance": False,
    #             "wood_mask": True,
    #         },
    #         {
    #             "name": "Sans filtre bois",
    #             "gamma": 0.8,
    #             "white_balance": False,
    #             "wood_mask": False,
    #         },
    #         {
    #             "name": "Brillant (gamma=0.6) sans filtre bois",
    #             "gamma": 0.6,
    #             "white_balance": False,
    #             "wood_mask": False,
    #         },
    #         {
    #             "name": "Sombre (gamma=1.2) sans filtre bois",
    #             "gamma": 1.2,
    #             "white_balance": False,
    #             "wood_mask": False,
    #         },
    #     ]
        
    #     for config in configs:
    #         print(f"\n[TEST] {config['name']}")
            
    #         detections = detect_colored_stickers(
    #             self.warp,
    #             self.boxes,
    #             apply_gamma_correction=True,
    #             gamma=config["gamma"],
    #             apply_white_balance=config["white_balance"],
    #             apply_wood_mask=config["wood_mask"],
    #             min_area_percent=2
    #         )
            
    #         # Count detections
    #         detected_count = sum(1 for info in detections.values() if info["color"] is not None)
    #         print(f"  → {detected_count} gommettes détectées")
            
    #         # Visualize
    #         vis_img = self.warp.copy()
    #         color_to_bgr = {
    #             "red": (0, 0, 255),
    #             "orange": (0, 140, 255),
    #             "yellow": (0, 255, 255),
    #             "green": (0, 255, 0),
    #             "purple": (255, 0, 255),
    #             "blue": (255, 0, 0),
    #         }
            
    #         for cell_name, info in detections.items():
    #             if info["color"] is None:
    #                 continue
    #             x1, y1, x2, y2 = self.boxes[cell_name]
    #             color = color_to_bgr.get(info["color"], (128, 128, 128))
    #             cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
    #         cv2.imshow(f'{config["name"]} - Press Q for next', vis_img)
            
    #         while True:
    #             key = cv2.waitKey(1) & 0xFF
    #             if key == ord('q'):
    #                 cv2.destroyWindow(f'{config["name"]} - Press Q for next')
    #                 break
    
    # def test_board_state_from_stickers(self):
    #     """Convertit détections de gommettes en état du plateau."""
    #     if self.warp is None or self.boxes is None:
    #         print("[ERROR] Impossible d'extraire les marqueurs")
    #         return
        
    #     print("\n[BOARD STATE FROM STICKERS]")
        
    #     detections = detect_colored_stickers(
    #         self.warp,
    #         self.virtual_boxes,
    #         apply_gamma_correction=True,
    #         gamma=1.2,
    #         apply_white_balance=False,
    #         apply_wood_mask=True,
    #         min_area_percent=2
    #     )

    #     self.assertIsNotNone(detections)
    #     self.assertEqual(len(detections), 64)
        
    #     # # Définir le mapping couleur → pièce
    #     # color_to_piece = {
    #     #     "red": "K",      
    #     #     "orange": "Q",    
    #     #     "yellow": "F",     
    #     #     "green": "T",
    #     #     "purple": "C",
    #     #     "blue": "P",   
    #     # }
        
    #     board_state = board_state_from_colored_stickers(detections)
        
    #     print("\n[BOARD STATE]")
    #     print("  a b c d e f g h")
    #     ranks = ['8', '7', '6', '5', '4', '3', '2', '1']
    #     for i, row in enumerate(board_state):
    #         rank = ranks[i]
    #         row_str = ' '.join(piece if piece != ' ' else '.' for piece in row)
    #         print(f"{rank} {row_str} {rank}")
    #     print("  a b c d e f g h")
        
    #     self.assertIsNotNone(board_state)
    
    def wait_for_space_and_capture(self):
        """Display live camera feed and wait for SPACE key to capture."""
        print("\n[INFO] Positionner l'échiquier avec les 4 marqueurs visibles.")
        print("[INFO] Appuyer sur ESPACE pour capturer...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Impossible de capturer une image.")
            
            cv2.imshow('Camera Feed - Press SPACE to capture', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE
                cv2.destroyAllWindows()
                print("[INFO] Image capturée. Traitement en cours...")
                return frame
            elif key == ord('q'):
                cv2.destroyAllWindows()
                raise RuntimeError("Capture annulée par l'utilisateur.")

    # def test_virtual_grid_visualization(self):
    #     """
    #     Test visuel : crée une grille virtuelle projetée et l'affiche avec la grille réelle.
    #     Ne nécessite pas de connaissance préalable des focales : on estime fx,fy depuis la taille de l'image.
    #     """
    #     if self.warp is None or self.boxes is None:
    #         print("[ERROR] Impossible d'extraire les marqueurs")
    #         return
    #     # Visualiser la grille réelle (bleu) vs la grille virtuelle (rouge)

    #     visualize_real_and_virtual_grids(self.warp, self.boxes, self.virtual_boxes)


    # def test_preprocessing_pipeline_visual(self):
    #     """
    #     ⭐ PRÉTRAITEMENT — Visualise chaque étape de la pipeline.
    #     Aide à choisir : CLAHE, blur, gamma, white balance.
    #     """
    #     if self.warp is None:
    #         print("[ERROR] Impossible d'extraire les marqueurs")
    #         return
        
    #     print("\n[PREPROCESSING PIPELINE VISUALIZATION]")
        
    #     steps = [
    #         ("Original", self.warp),
    #         ("+ Gamma (1.5)", apply_gamma(self.warp, 1.5)),
    #         ("+ Gamma + Blur", cv2.GaussianBlur(apply_gamma(self.warp, 1.5), (5, 5), 1.0)),
    #         ("+ CLAHE", apply_clahe_lab(self.warp, clip_limit=2.0, tile_size=8)),
    #         ("+ CLAHE + Gamma + Blur", cv2.GaussianBlur(
    #             apply_clahe_lab(apply_gamma(self.warp, 1.5), clip_limit=2.0), (5, 5), 1.0)),
    #     ]
        
    #     for title, img in steps:
    #         display = img.copy()
    #         for cell_name, (x1, y1, x2, y2) in self.boxes.items():
    #             cv2.rectangle(display, (int(x1), int(y1)), (int(x2), int(y2)), (100, 100, 100), 1)
            
    #         cv2.imshow(f'{title} - Press Q to next', display)
    #         print(f"[VIEWING] {title}")
            
    #         while True:
    #             key = cv2.waitKey(1) & 0xFF
    #             if key == ord('q'):
    #                 cv2.destroyWindow(f'{title} - Press Q to next')
    #                 break
        
    #     print("\n[RECOMMENDATION]")
    #     print("Choisissez la version où les gommettes ressortent le mieux.")
    #     print("Paramètres à ajuster dans detect_colored_stickers():")
    #     print("  - apply_clahe=True/False")
    #     print("  - apply_blur=True/False")
    #     print("  - gamma=0.6...1.4")
    
    # def test_hsv_mask_comparison(self):
    #     """
    #     ⭐ MASQUES HSV — Compare simple vs. robuste (morphologie).
    #     Affiche les masques générés pour chaque couleur.
    #     """
    #     if self.warp is None:
    #         print("[ERROR] Impossible d'extraire les marqueurs")
    #         return
        
    #     print("\n[HSV MASK COMPARISON - Simple vs Robust]")
        
    #     # Préparation
    #     img_work = apply_gamma(self.warp, 1.2)
    #     img_work = apply_clahe_lab(img_work, clip_limit=2.0)
    #     img_work = cv2.GaussianBlur(img_work, (5, 5), 1.0)
    #     img_hsv = cv2.cvtColor(img_work, cv2.COLOR_BGR2HSV)
        
    #     # Choisir une case avec gommette
    #     test_cell = 'e4'
    #     x1, y1, x2, y2 = self.boxes[test_cell]
    #     roi_hsv = img_hsv[y1:y2, x1:x2]
        
    #     print(f"[REGION] Analysant la case {test_cell}...")
        
    #     # Afficher le ROI original
    #     roi_bgr = self.warp[y1:y2, x1:x2]
    #     cv2.imshow(f'ROI {test_cell} - Press Q to continue', roi_bgr)
    #     while cv2.waitKey(1) & 0xFF != ord('q'):
    #         pass
    #     cv2.destroyWindow(f'ROI {test_cell} - Press Q to continue')
        
    #     # Comparer chaque couleur
    #     for color_name, ranges in COLOR_RANGES.items():
    #         mask_simple = np.zeros(roi_hsv.shape[:2], dtype=np.uint8)
    #         for low, high in ranges:
    #             sub_mask = cv2.inRange(roi_hsv, low, high)
    #             mask_simple = cv2.bitwise_or(mask_simple, sub_mask)
            
    #         mask_robust = mask_simple.copy()
    #         kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    #         kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #         mask_robust = cv2.morphologyEx(mask_robust, cv2.MORPH_OPEN, kernel_open, iterations=1)
    #         mask_robust = cv2.morphologyEx(mask_robust, cv2.MORPH_CLOSE, kernel_close, iterations=2)
            
    #         # Affichage côte à côte
    #         combined = np.hstack([mask_simple, mask_robust])
    #         cv2.imshow(f'{color_name} - SIMPLE (left) vs ROBUST (right) - Q to next', combined)
            
    #         simple_count = cv2.countNonZero(mask_simple)
    #         robust_count = cv2.countNonZero(mask_robust)
    #         print(f"  {color_name}: simple={simple_count}px, robust={robust_count}px")
            
    #         while cv2.waitKey(1) & 0xFF != ord('q'):
    #             pass
    #         cv2.destroyWindow(f'{color_name} - SIMPLE (left) vs ROBUST (right) - Q to next')
        
    #     print("\n[OBSERVATION]")
    #     print("La version ROBUST élimine le bruit et solidifie les gommettes.")
    #     print("C'est celle-ci qui devrait être utilisée par défaut.")
    
    # def test_detect_colored_stickers_robust_params(self):
    #     """
    #     ⭐ PARAMÈTRES ROBUSTES — Teste différentes combinaisons.
    #     """
    #     if self.warp is None or self.virtual_boxes is None:
    #         print("[ERROR] Impossible d'extraire les marqueurs")
    #         return
        
    #     print("\n[ROBUST STICKER DETECTION - Parameter Sweep]")
        
    #     configs = [
    #         {
    #             "name": "Brutal (gamma=1.5, CLAHE, blur)",
    #             "gamma": 1.5,
    #             "clahe": True,
    #             "blur": True,
    #         },
    #         {
    #             "name": "Doux (gamma=0.8, CLAHE, blur)",
    #             "gamma": 0.8,
    #             "clahe": True,
    #             "blur": True,
    #         },
    #         {
    #             "name": "Normal (gamma=0.8, sans CLAHE)",
    #             "gamma": 0.8,
    #             "clahe": False,
    #             "blur": True,
    #         },
    #         {
    #             "name": "Minimal (gamma=1.0, sans CLAHE, sans blur)",
    #             "gamma": 1.0,
    #             "clahe": False,
    #             "blur": False,
    #         },
    #     ]
        
    #     for config in configs:
    #         print(f"\n[TEST] {config['name']}")
            
    #         detections = detect_colored_stickers_robust(
    #             self.warp,
    #             self.virtual_boxes,
    #             apply_gamma_correction=True,
    #             gamma=config["gamma"],
    #             apply_clahe=config["clahe"],
    #             apply_blur=config["blur"],
    #             apply_white_balance=False,
    #             apply_wood_mask=True,
    #             min_area_percent=2,
    #             debug=False
    #         )
            
    #         # Compter les détections
    #         detected_count = sum(1 for info in detections.values() if info["color"] is not None)
    #         avg_confidence = np.mean([info["confidence"] for info in detections.values() if info["color"] is not None])
            
    #         print(f"  → {detected_count} gommettes détectées")
    #         print(f"  → Confiance moyenne: {avg_confidence:.1%}")
            
    #         # Visualiser
    #         vis_img = self.warp.copy()
    #         color_to_bgr = {
    #             "red": (0, 0, 255),
    #             "orange": (0, 140, 255),
    #             "yellow": (0, 255, 255),
    #             "green": (0, 255, 0),
    #             "purple": (255, 0, 255),
    #             "blue": (255, 0, 0),
    #         }
            
    #         for cell_name, info in detections.items():
    #             if info["color"] is None:
    #                 continue
    #             x1, y1, x2, y2 = self.boxes[cell_name]
    #             color = color_to_bgr.get(info["color"], (128, 128, 128))
    #             cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    #             cv2.putText(vis_img, f"{info['color'][:1]} {info['confidence']:.0%}",
    #                        (int(x1)+5, int(y1)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
    #         cv2.imshow(f"{config['name']} - Press Q for next", vis_img)
            
    #         while cv2.waitKey(1) & 0xFF != ord('q'):
    #             pass
    #         cv2.destroyWindow(f"{config['name']} - Press Q for next")
        
    #     print("\n[CONCLUSION]")
    #     print("Choisissez la configuration qui détecte le mieux vos gommettes.")
    #     print("Utilisez ces paramètres dans votre code de production.")
    def sample_roi(self,img_p,img_hsv):
        sample_rois = {}
        click = {"pt": None}
        win_name = "Click sample - s=skip, q=abort"

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                click["pt"] = (x, y)

        for color in COLOR_RANGES.keys():
            click["pt"] = None
            prompt = img_p.copy()
            cv2.putText(prompt, f"Click sample for color: {color} (s=skip,q=abort)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(win_name, prompt)
            cv2.setMouseCallback(win_name, on_mouse)

            while True:
                key = cv2.waitKey(1) & 0xFF
                if click["pt"] is not None:
                    x, y = click["pt"]
                    sz = 6
                    x1 = max(0, x - sz); y1 = max(0, y - sz)
                    x2 = min(img_hsv.shape[1], x + sz); y2 = min(img_hsv.shape[0], y + sz)
                    sample_rois[color] = (x1, y1, x2, y2)
                    cv2.rectangle(prompt, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.imshow(win_name, prompt)
                    cv2.waitKey(200)
                    break
                if key == ord('s'):
                    print(f"Skip color {color}")
                    break
                if key == ord('q'):
                    cv2.destroyWindow(win_name)
                    self.skipTest("User aborted sample collection")
            cv2.destroyWindow(win_name)

        return sample_rois
    
    # def test_adapt_color_ranges_from_samples(self):
    #     """
    #     Interactive test for adapt_color_ranges_from_samples.
    #     For each color in COLOR_RANGES you are prompted to click a sample sticker.
    #     - Click a sample pixel for the shown color (small ROI is stored).
    #     - Press 's' to skip a color, 'q' to abort the whole test.
    #     The test prints the adapted HSV ranges and shows per-color masks briefly.
    #     """
    #     if self.warp is None or self.boxes is None:
    #         self.skipTest("warp or boxes not available")

    #     # Preprocess similar to detection pipeline to stabilise HSV measurements
    #     img_p = apply_gamma(self.warp, 1.5)
    #     img_p = apply_clahe_lab(img_p, clip_limit=2.0, tile_size=8)
    #     img_hsv = cv2.cvtColor(img_p, cv2.COLOR_BGR2HSV)

    #     sample_rois = {}
    #     click = {"pt": None}
    #     win_name = "Click sample - s=skip, q=abort"

    #     def on_mouse(event, x, y, flags, param):
    #         if event == cv2.EVENT_LBUTTONDOWN:
    #             click["pt"] = (x, y)

    #     for color in COLOR_RANGES.keys():
    #         click["pt"] = None
    #         prompt = img_p.copy()
    #         cv2.putText(prompt, f"Click sample for color: {color} (s=skip,q=abort)",
    #                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    #         cv2.imshow(win_name, prompt)
    #         cv2.setMouseCallback(win_name, on_mouse)

    #         while True:
    #             key = cv2.waitKey(1) & 0xFF
    #             if click["pt"] is not None:
    #                 x, y = click["pt"]
    #                 sz = 12
    #                 x1 = max(0, x - sz); y1 = max(0, y - sz)
    #                 x2 = min(img_hsv.shape[1], x + sz); y2 = min(img_hsv.shape[0], y + sz)
    #                 sample_rois[color] = (x1, y1, x2, y2)
    #                 cv2.rectangle(prompt, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #                 cv2.imshow(win_name, prompt)
    #                 cv2.waitKey(200)
    #                 break
    #             if key == ord('s'):
    #                 print(f"Skip color {color}")
    #                 break
    #             if key == ord('q'):
    #                 cv2.destroyWindow(win_name)
    #                 self.skipTest("User aborted sample collection")
    #         cv2.destroyWindow(win_name)
            

    #     # Compute adapted ranges
    #     new_ranges = adapt_color_ranges_from_samples(img_hsv, sample_rois, COLOR_RANGES, h_tol=20, s_tol=60, v_tol=60)
    #     print("\nAdaptive HSV ranges:")
    #     for c, ranges in new_ranges.items():
    #         print(f"  {c}: {ranges}")

    #     # Quick visual check: show adapted mask on the sampled ROI for each color
    #     for color, roi in sample_rois.items():
    #         x1, y1, x2, y2 = roi
    #         roi_hsv = img_hsv[y1:y2, x1:x2]
    #         mask = np.zeros(roi_hsv.shape[:2], dtype=np.uint8)
    #         for low, high in new_ranges.get(color, []):
    #             mask = cv2.bitwise_or(mask, cv2.inRange(roi_hsv, low, high))
    #         cv2.imshow(f"{color} adapted mask", mask)
    #         cv2.waitKey(500)
    #         cv2.destroyWindow(f"{color} adapted mask")

    #     self.assertIsInstance(new_ranges, dict)


if __name__ == '__main__':
    unittest.main()