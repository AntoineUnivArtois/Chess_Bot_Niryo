"""
Pipeline temps réel pour la détection des pièces d'échecs.

Utilise les fonctions de image_to_board.py pour :
1. Détecter l'échiquier une seule fois au démarrage
2. Attendre la validation utilisateur pour placer les pièces
3. Détecter les pièces en temps réel avec lissage par vote majoritaire (10 frames)
4. Afficher l'état de l'échiquier en direct sur le flux caméra
"""

import numpy as np
import cv2
from collections import defaultdict, deque
from Img_treatment import ImgTreatment
from image_to_board import (
    extract_img_markers_with_margin,
    get_cell_boxes,
    detect_colored_stickers,
    board_state_from_colored_stickers,
    create_virtual_detection_grid,
    COLOR_RANGES
)


class ChessBoardStreamingPipeline:
    """Pipeline temps réel pour la détection d'un échiquier avec pièces colorées."""
    
    def __init__(self, camera_id=0, base_size=800, history_length=10):
        """
        Initialise la pipeline streaming.
        
        Args:
            camera_id: ID de la caméra USB (par défaut 0)
            base_size: Taille de base de l'échiquier en pixels
            history_length: Nombre de frames pour le lissage par vote majoritaire
        """
        self.camera_id = camera_id
        self.base_size = base_size
        self.history_length = history_length
        
        # État de l'échiquier
        self.board_state = None  # Dict {cell_name: letter}
        self.detection_history = defaultdict(lambda: deque(maxlen=history_length))
        self.treated_frame = None  # Image traitée courante
        self.raw_frame = None  # Image brute courante (flux caméra)
        
        # Paramètres détectés une fois
        self.cap = None
        self.warp = None
        self.boxes = None
        self.virtual_boxes = None
        self.real_corners_3d = None
        self.image_corners = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.M = None  # Matrice de transformation homographique
        self.M_inv = None  # Matrice inverse pour reprojeter vers l'image brute
        self.dims = None  # Dimensions de l'échiquier warpé
        
        # État du programme
        self.board_detected = False
        self.board_validated = False
        self.running = False
    
    def initialize_camera(self):
        """Initialise la capture vidéo."""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la caméra {self.camera_id}. Vérifiez la connexion USB.")
        
        # Paramètres de la caméra
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
    
    def detect_board(self):
        """
        Détecte l'échiquier une seule fois.
        
        Affiche le flux caméra et attends l'appui sur ESPACE pour valider
        la position des marqueurs.
        """
        print("\n" + "="*70)
        print("[ÉTAPE 1] Détection de l'échiquier")
        print("="*70)
        print("[INFO] Positionnez l'échiquier avec les 4 marqueurs visibles.")
        print("[INFO] Appuyez sur ESPACE pour capturer et détecter l'échiquier...")
        print()
        
        frame_capture = None
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Impossible de capturer depuis la caméra.")
            
            # Afficher le flux
            cv2.putText(frame, "Appuyez sur ESPACE pour capturer", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Detection Echiquier - Appuyez sur ESPACE', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE
                frame_capture = frame
                cv2.destroyAllWindows()
                break
            elif key == ord('q'):
                cv2.destroyAllWindows()
                raise RuntimeError("Détection annulée par l'utilisateur.")
        
        # Extraction des marqueurs
        print("[PROCESSING] Détection des marqueurs...")
        warp, M, dims, corners, margin_px = extract_img_markers_with_margin(
            frame_capture,
            workspace_ratio=1.0,
            base_size=self.base_size,
        )
        
        if warp is None or corners is None:
            raise RuntimeError("Impossible de détecter les marqueurs de l'échiquier.")
        
        self.warp = warp
        self.M = M  # Sauvegarder la matrice de transformation
        self.M_inv = cv2.invert(M)[1]  # Calculer la matrice inverse pour reprojeter vers l'image brute
        self.dims = dims  # Sauvegarder les dimensions
        self.boxes = get_cell_boxes(warp, margin_px, player_plays_white=True)
        
        # Création de la grille virtuelle projetée
        print("[PROCESSING] Création de la grille virtuelle projetée...")
        
        # Récupérer les 4 coins détectés (image points) dans l'ordre [tl,tr,br,bl]
        tl = (self.boxes['a8'][0], self.boxes['a8'][1])
        tr = (self.boxes['h8'][2], self.boxes['h8'][1])
        br = (self.boxes['h1'][2], self.boxes['h1'][3])
        bl = (self.boxes['a1'][0], self.boxes['a1'][3])
        self.image_corners = np.array([tl, tr, br, bl], dtype=np.float64)
        
        # Coins 3D du plateau réel (240 x 240 mm)
        self.real_corners_3d = np.array([
            [0.0, 0.0, 0.0],       # tl
            [240.0, 0.0, 0.0],     # tr
            [240.0, 240.0, 0.0],   # br
            [0.0, 240.0, 0.0],     # bl
        ], dtype=np.float64)
        
        # Estimation simple des intrinsics
        h, w = warp.shape[:2]
        f_est = max(w, h)
        cx = w / 2.0
        cy = h / 2.0
        self.camera_matrix = np.array(
            [[f_est, 0, cx], [0, f_est, cy], [0, 0, 1]],
            dtype=np.float64
        )
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        
        # Créer la grille virtuelle projetée
        self.virtual_boxes, _ = create_virtual_detection_grid(
            self.real_corners_3d,
            self.image_corners,
            self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            grid_size=8,
            plane_height_cm=22.0,
            margin_percent=0.05
        )
        
        self.board_detected = True
        print("[SUCCESS] Échiquier détecté avec succès ✓\n")
    
    def wait_board_validation(self):
        """
        Affiche un message et attends l'appui sur 'v' pour valider
        le placement des pièces.
        """
        print("="*70)
        print("[ÉTAPE 2] Placement des pièces")
        print("="*70)
        print("[INSTRUCTION] Placez les pions puis appuyez sur 'v' pour valider")
        print()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Impossible de capturer depuis la caméra.")
            
            # Afficher le message sur l'image
            cv2.putText(frame, "Placez les pions", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(frame, "Appuyez sur 'v' pour valider", (50, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            cv2.imshow('Placement des pieces', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('v'):
                cv2.destroyAllWindows()
                break
            elif key == ord('q'):
                cv2.destroyAllWindows()
                raise RuntimeError("Opération annulée par l'utilisateur.")
        
        self.board_validated = True
        print("[SUCCESS] Pièces validées, démarrage de la détection temps réel\n")
    
    def get_smoothed_state(self):
        """
        Retourne l'état de l'échiquier lissé avec une méthode robuste basée sur :
        1. Un seuil de majorité stricte (min 70% des frames)
        2. Exclusion des détections None (cases vides)
        3. Nécessité d'une cohérence minimale
        
        Returns:
            dict: {cell_name: letter} où letter est stable et fiable
        """
        smoothed_state = {}
        min_confidence_threshold = 0.70  # Demande 70% de consensus minimum
        
        for cell_name in self.boxes.keys():
            history = self.detection_history[cell_name]
            
            if not history:
                smoothed_state[cell_name] = None
                continue
            
            # Compter les occurrences de chaque détection
            vote_counts = defaultdict(int)
            for detection in history:
                vote_counts[detection] += 1
            
            # Obtenir le total et la meilleure détection
            total_votes = len(history)
            
            if not vote_counts:
                smoothed_state[cell_name] = None
                continue
            
            # Trouver la détection avec le plus de votes (excluant None)
            non_none_votes = {k: v for k, v in vote_counts.items() if k is not None}
            
            if not non_none_votes:
                # Aucune détection non-None dans l'historique
                smoothed_state[cell_name] = None
                continue
            
            # Prendre la meilleure détection parmi les non-None
            best_detection = max(non_none_votes.items(), key=lambda x: x[1])[0]
            best_vote_count = non_none_votes[best_detection]
            confidence = best_vote_count / total_votes
            
            # Appliquer le seuil de confiance : accepter seulement si confiance >= 70%
            if confidence >= min_confidence_threshold:
                smoothed_state[cell_name] = best_detection
            else:
                # Pas assez de consensus → rester None (case vide)
                smoothed_state[cell_name] = None
        
        return smoothed_state
    
    def process_frame(self, frame):
        """
        Traite une frame : détection et lissage.
        
        Args:
            frame: Image BGR capturée depuis la caméra
        
        Returns:
            dict: État lissé de l'échiquier {cell_name: letter}
        """
        # Sauvegarder la frame brute pour la visualisation
        self.raw_frame = frame.copy()
        
        # Warper la frame courante avec la même matrice que la détection initiale
        warp_frame = cv2.warpPerspective(frame, self.M, self.dims)
        
        # Appliquer le MÊME traitement d'image que dans test_image_to_board.py
        img_treatment = ImgTreatment(warp_frame)
        treated_frame, _ = img_treatment.traitement_image()
        
        # Sauvegarder l'image traitée pour la détection (pas la visualisation)
        self.treated_frame = treated_frame
        
        # Détecter les gommettes colorées dans les cases virtuelles
        detections = detect_colored_stickers(
            treated_frame,
            self.virtual_boxes,
            min_area_percent=50,
            color_ranges=COLOR_RANGES
        )
        
        # Convertir les détections en lettres (type de pion + camp)
        for cell_name, info in detections.items():
            color = info.get("color")
            side = info.get("side")
            
            if color is not None and side is not None:
                letter = board_state_from_colored_stickers(color, side)
            else:
                letter = None
            
            # Ajouter à l'historique
            self.detection_history[cell_name].append(letter)
        
        # Récupérer l'état lissé
        smoothed_state = self.get_smoothed_state()
        
        return smoothed_state
    
    def visualize_state(self, board_state):
        """
        Affiche l'état de l'échiquier sur l'image brute capturée par la caméra.
        Superpose l'état des pions sur le flux direct.
        Les coordonnées des virtual_boxes sont transformées de l'espace warpé
        vers l'espace de l'image brute en utilisant la matrice inverse.
        
        Args:
            board_state: Dict {cell_name: letter}
        
        Returns:
            Image annotée avec l'état des pions
        """
        if not hasattr(self, 'raw_frame') or self.raw_frame is None:
            return None
        
        vis_img = self.raw_frame.copy()
        
        for cell_name, letter in board_state.items():
            if letter is None:
                continue
            
            # Récupérer les coordonnées dans l'espace warpé
            x1_w, y1_w, x2_w, y2_w = self.virtual_boxes[cell_name]
            
            # Transformer les 4 coins du rectangle de l'espace warpé vers l'espace brute
            # en utilisant la matrice inverse de l'homographie
            corners_warp = np.array([
                [[x1_w, y1_w]],
                [[x2_w, y1_w]],
                [[x2_w, y2_w]],
                [[x1_w, y2_w]]
            ], dtype=np.float32)
            
            corners_raw = cv2.perspectiveTransform(corners_warp, self.M_inv)
            
            # Récupérer les coordonnées transformées
            x1_r = int(np.min(corners_raw[:, 0, 0]))
            x2_r = int(np.max(corners_raw[:, 0, 0]))
            y1_r = int(np.min(corners_raw[:, 0, 1]))
            y2_r = int(np.max(corners_raw[:, 0, 1]))
            
            # Choisir la couleur basée sur la lettre (noir/blanc)
            # MAJUSCULE = pion noir → couleur noire pour l'affichage
            # minuscule = pion blanc → couleur blanche pour l'affichage
            if letter.isupper():
                color = (0, 0, 0)  # Noir pour les pièces noires (MAJUSCULE)
            else:
                color = (255, 255, 255)  # Blanc pour les pièces blanches (minuscule)
            
            # Dessiner un rectangle autour de la case et ajouter la lettre
            cv2.rectangle(vis_img, (x1_r, y1_r), (x2_r, y2_r), color, 2)
            cx = (x1_r + x2_r) // 2
            cy = (y1_r + y2_r) // 2
            cv2.putText(vis_img, letter,
                       (cx - 10, cy + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        
        return vis_img
    
    def run(self):
        """Lance la pipeline temps réel complète."""
        try:
            # Initialisation
            self.initialize_camera()
            
            # Étape 1 : Détection de l'échiquier
            self.detect_board()
            
            # Étape 2 : Validation du placement des pièces
            self.wait_board_validation()
            
            # Étape 3 : Détection temps réel
            print("="*70)
            print("[ÉTAPE 3] Détection temps réel")
            print("="*70)
            print("[INFO] Détection en cours...")
            print("[CONTROL] Appuyez sur 'q' pour arrêter\n")
            
            self.running = True
            frame_count = 0
            
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("[ERROR] Impossible de capturer.")
                    break
                
                # Traiter la frame
                board_state = self.process_frame(frame)
                
                # Visualiser l'état (utilise self.treated_frame sauvegardée dans process_frame)
                vis_img = self.visualize_state(board_state)
                
                # Vérifier que la visualisation n'est pas None
                if vis_img is None:
                    print("[ERROR] Impossible de visualiser l'état")
                    break
                
                # Afficher l'état dans la console (toutes les 30 frames)
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"\n[FRAME {frame_count}] État de l'échiquier:")
                    for rank in '87654321':
                        row = []
                        for file in 'abcdefgh':
                            cell = f"{file}{rank}"
                            letter = board_state.get(cell, '.')
                            row.append(letter if letter else '.')
                        print(f"  {rank} | {' '.join(row)} |")
                    print("  +-+-+-+-+-+-+-+-+")
                    print("  a b c d e f g h\n")
                
                # Afficher le flux vidéo avec les annotations
                cv2.putText(vis_img, f"Frame: {frame_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(vis_img, "Appuyez sur 'q' pour arreter", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                cv2.namedWindow('Detection Temps Reel', cv2.WINDOW_FULLSCREEN)
                cv2.imshow('Detection Temps Reel', vis_img)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                    cv2.destroyAllWindows()
            
            # État final
            print("\n" + "="*70)
            print("[RÉSULTAT FINAL] État détecté de l'échiquier:")
            print("="*70)
            final_state = self.get_smoothed_state()
            for rank in '87654321':
                row = []
                for file in 'abcdefgh':
                    cell = f"{file}{rank}"
                    letter = final_state.get(cell, '.')
                    row.append(letter if letter else '.')
                print(f"{rank} | {' '.join(row)} |")
            print("  +-+-+-+-+-+-+-+-+")
            print("  a b c d e f g h\n")
            
        except Exception as e:
            print(f"\n[ERROR] {e}")
            raise
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Libère les ressources."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("[CLEANUP] Ressources libérées ✓")


def main():
    """Point d'entrée principal."""
    pipeline = ChessBoardStreamingPipeline(
        camera_id=0,
        base_size=800,
        history_length=10
    )
    pipeline.run()


if __name__ == "__main__":
    main()
