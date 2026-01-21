#!/usr/bin/env python3
'''
    This code allows the player to play against the Niryo Ned2 for a chess game
    It uses vision to detect board and pieces and when player moves
    It uses reinforcement learning for robot's next move decision
    You need to install all dependencies via conda environment.yml
    For more details, check README.md on GitHub

    I. First part provides modules imports
    II. Second part provides robot configuration
    III. Third part provides helper functions (convertions etc.
    IV. Fourth part describes the main
'''


# ─────────────────────────────────────────────────────────────────────────────
# I. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

# Standard modules import
import sys
import numpy as np
import cv2
import math
import time
import chess
import chess.engine

# Niryo modules import for Ned2 movements
from pyniryo import NiryoRobot, PoseObject, RobotAxis

# Utils imports (workspace gathering / image to board transcription funcs etc.)
sys.path.append("ChessUtils")
from ChessUtils.chess_board import board as ChessBoard
from collections import defaultdict, deque
from ChessUtils import (
    extract_img_markers_with_margin,
    get_cell_boxes,
    create_virtual_detection_grid,
    detect_colored_stickers,
    board_state_from_colored_stickers,
    Img_treatment,
    COLOR_RANGES

)

# ─────────────────────────────────────────────────────────────────────────────
# II. ROBOT CONFIG
# ─────────────────────────────────────────────────────────────────────────────

ROBOT_IP = '169.254.200.200' # Direct Ethernet

NUM_READS = 200
CELL_SIZE = 0.04  # 4 cm
ELECTROMAGNET_PIN = 'DO4'
ELECTROMAGNET_ELEVATION_COEFF = 0.029
CAMERA_ID = 0
BOARD_SIZE = CELL_SIZE*8   # m
OFFSET_M = 0.005   # 5 mm
COORD_BASE_MM = np.array([
    [436, -148],
    [442, 126],
    [154, -141],
    [158, 124]
], dtype=np.float32)
PTS_IDEAUX = np.array([[0, 0],[7, 0],[0, 7],[7, 7]], dtype=np.float32)
H, _ = cv2.findHomography(PTS_IDEAUX, COORD_BASE_MM)

# Individual pieces heights (Change this to your pieces heights if you cloned..
# ..this repo and changed STLs files) in meters
PIECE_HEIGHTS = {
    'P': 0.042, 'p': 0.042,
    'N': 0.053, 'n': 0.053,
    'B': 0.055, 'b': 0.055,
    'R': 0.047, 'r': 0.047,
    'Q': 0.062, 'q': 0.062,
    'K': 0.075, 'k': 0.075,
}
SHIFT_DIST = 0.015       # Distance de sécurité
SHIFT_DIST_EMPTY  = PIECE_HEIGHTS['k'] + SHIFT_DIST # Distance de déplacement à vide (au dessus de la plus grande pièce)
BOARD_THICKNESS = 0.014  # Épaisseur du plateau en m

FRAME_COUNT = 8
CONFIRM_RATIO = 0.6           
MIN_CONFIRM = int(FRAME_COUNT * CONFIRM_RATIO)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────
def game_result(b: ChessBoard):
    """'mat', 'pat' or None if game continues."""
    if b.actions():
        return None
    return 'mat' if b.check_status() else 'pat'

def square_to_index(sq: str):
    """Ex. 'e2' → (6,4) (line i, column j in current_board)."""
    file, rank = sq[0], int(sq[1])
    j = ord(file) - ord('a')
    i = 8 - rank
    return i, j

def init_sticky_state(boxes):
    sticky = {}
    for cell in boxes.keys():
        sticky[cell] = {
            "stable": " ",
            "candidate": None,
            "count": 0
        }
    return sticky

def stockfish_move_to_robot(move):
    from_sq = chess.square_name(move.from_square)
    to_sq = chess.square_name(move.to_square)
    return from_sq, to_sq

# ─────────────────────────────────────────────────────────────────────────────
# CAMERA FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def initialize_camera():
    """Initialise la capture vidéo."""
    cap = cv2.VideoCapture(CAMERA_ID,cv2.CAP_AVFOUNDATION)
    time.sleep(1)
    
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la caméra {CAMERA_ID}. Vérifiez la connexion USB.")
    
    # Paramètres de la caméra
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    return cap

def detect_board(frame):
    """
    Détecte l'échiquier une seule fois.
    
    """
    # Extraction des marqueurs
    warp, M, dims, corners, margin_px = extract_img_markers_with_margin(
        frame,
        workspace_ratio=1.0,
        base_size=800,
    )
    
    if warp is None or corners is None:
        return None, False, None, None
    
    M_inv = cv2.invert(M)[1]  # Calculer la matrice inverse pour reprojeter vers l'image brute
    boxes = get_cell_boxes(warp, margin_px, player_plays_white=True)
    
    # Création de la grille virtuelle projetée
    print("[PROCESSING] Création de la grille virtuelle projetée...")
    
    # Récupérer les 4 coins détectés (image points) dans l'ordre [tl,tr,br,bl]
    tl = (boxes['a8'][0], boxes['a8'][1])
    tr = (boxes['h8'][2], boxes['h8'][1])
    br = (boxes['h1'][2], boxes['h1'][3])
    bl = (boxes['a1'][0], boxes['a1'][3])
    image_corners = np.array([tl, tr, br, bl], dtype=np.float64)
    
    # Coins 3D du plateau réel (240 x 240 mm)
    real_corners_3d = np.array([
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
    camera_matrix = np.array(
        [[f_est, 0, cx], [0, f_est, cy], [0, 0, 1]],
        dtype=np.float64
    )
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    
    # Créer la grille virtuelle projetée
    virtual_boxes, _ = create_virtual_detection_grid(
        real_corners_3d,
        image_corners,
        camera_matrix,
        dist_coeffs=dist_coeffs,
        grid_size=8,
        plane_height_cm=22.0,
        margin_percent=0.05
    )
    
    board_detected = True
    print("[SUCCESS] Échiquier détecté avec succès ✓\n")

    return boxes, virtual_boxes, board_detected, M, dims

# ─────────────────────────────────────────────────────────────────────────────
# X, Y, Z TILES & BUFFERS' POSES CALCULATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def case_center(i, j):
    pt = np.array([[[i, j]]], dtype=np.float32)
    res = cv2.perspectiveTransform(pt, H)
    return np.array([res[0,0,0], res[0,0,1]], dtype=np.float32)

def chess_square_to_rel(square: str, player_plays_white=True):
    """
    Convertit une case d'échecs ('a1', 'e4', 'h8') en (x_rel, y_rel)
    en position pour robot.move()
    """

    square = square.lower().strip()
    file = square[0]   # 'a' .. 'h'
    rank = square[1]   # '1' .. '8'

    # Indices 0..7
    i = ord(file) - ord('a')        # a=0 ... h=7
    j = int(rank) - 1              # 8=0 (a8), 1=7 (a1)

    if not player_plays_white:
        # On retourne le plateau
        i = 7 - i
        j = 7 - j


    # Coordonnées relatives dans le workspace
    x_mm, y_mm = case_center(i, j)

    return x_mm*0.001, y_mm*0.001

def set_buffer_tiles_positions(buffer_tiles, cell_size=CELL_SIZE):
    """
    Compute and assign a PoseObject to each buffer slot around the board, based on a reference pose and square size.

    Inputs:
        buffer_tiles (dict[str, PoseObject]) – mapping from buffer slot names (e.g. "1 1", "1 2", ..., "4 4") to PoseObject placeholders.
        cell_size (float) – size of one chessboard square in meters, used to space buffer slots.

    Outputs:
        dict[str, PoseObject] – the same buffer_tiles dict, with each entry set to a PoseObject at the correct (x, y, z, roll, pitch, yaw) for that slot.
    """
    for ci, c in enumerate('1234'):
        for ri, r in enumerate('1234'):
            if c == '1':
                pos_x, pos_y = case_center(0,7)
                pos_x, pos_y = pos_x*0.001, pos_y*0.001
                pos_y -= cell_size
                dx = ri * cell_size
                dy = 0
            elif c == '2':
                pos_x, pos_y = case_center(0,7)
                pos_x, pos_y = pos_x*0.001, pos_y*0.001
                pos_x -= cell_size
                dx = 0
                dy = ri * cell_size
            elif c == '3':
                pos_x, pos_y = case_center(4,7)
                pos_x, pos_y = pos_x*0.001, pos_y*0.001
                pos_x -= cell_size
                dx = 0
                dy = ri * cell_size
            elif c == '4' :
                pos_x, pos_y = case_center(7,7)
                pos_x, pos_y = pos_x*0.001, pos_y*0.001
                pos_y += cell_size
                dx = ri * cell_size
                dy = 0
            # print(f"Position {c}{r} : pos_x = {pos_x}, pos_y = {pos_y}, coordonnées buffer : ({pos_x + dx}; {pos_y + dy})")
            buffer_tiles[f"{c}{r}"] = PoseObject(pos_x + dx, pos_y + dy, BOARD_THICKNESS + SHIFT_DIST_EMPTY, 0.00, math.pi/2, 0.00)
    return buffer_tiles

def wait_player_move(matrix_current, capture_fn):
    """
    Attend qu'un joueur humain ait effectué un coup valide.
    
    - Compare sans tenir compte de la casse pour détecter les changements
    - Vérifie le type exact de pièce pour connaître le pion déplacé
    - Recommence si plus de 2 changements détectés
    - Utilise une moyenne sur max_frames pour stabiliser la capture
    """
    
    valid_move = False

    while not valid_move:
        matrix_detected = capture_fn()

        removed = []
        added = []
        replaced = []

        # Analyse complète du plateau
        for r in range(8):
            for c in range(8):
                before = matrix_current[r][c]
                after = matrix_detected[r][c]

                # Départ
                if before != " " and after == " ":
                    removed.append((r, c, before))

                # Arrivée sur case vide
                elif before == " " and after != " ":
                    added.append((r, c, after))

                # Case occupée avant ET après → possible prise
                elif before != " " and after != " ":
                    if before != after :
                        replaced.append((r, c, before, after))

        # ─────────────────────────────
        # Rien n'a bougé
        if not removed and not added and not replaced:
            print("Pas encore bougé…")
            continue

        # ─────────────────────────────
        # Disparition isolée → bruit / main du joueur
        if len(removed) == 1 and not added and not replaced:
            print("Disparition isolée détectée → nouvelle capture")
            continue
        
        # ─────────────────────────────
        # Faux positif de vision : changement de casse sans déplacement
        if not removed and len(replaced) == 1:
            print("Changement de casse isolé → ignoré")
            continue

        # ─────────────────────────────
        # Déplacement simple
        if len(removed) == 1 and len(added) == 1 and not replaced:
            r_from, c_from, piece = removed[0]
            r_to, c_to, _ = added[0]

            matrix_new = [row.copy() for row in matrix_current]
            matrix_new[r_from][c_from] = " "
            matrix_new[r_to][c_to] = piece  # casse LOGIQUE conservée

            valid_move = True

            moves = {
                "type": "move",
                "from": (r_from, c_from),
                "to": (r_to, c_to),
                "piece": piece
            }

            print(f"Coup détecté : {piece}@({r_from},{c_from}) → ({r_to},{c_to})")
            break

        # ─────────────────────────────
        # Prise
        if len(removed) == 1 and len(replaced) == 1 and not added:
            r_from, c_from, piece = removed[0]
            r_to, c_to, victim, detected_piece = replaced[0]

            # La logique décide de la pièce déplacée
            piece_final = piece

            # La vision peut corriger la casse UNIQUEMENT si cohérente
            if detected_piece.lower() == piece.lower():
                piece_final = detected_piece

            matrix_new = [row.copy() for row in matrix_current]
            matrix_new[r_from][c_from] = " "
            matrix_new[r_to][c_to] = piece_final

            valid_move = True

            moves = {
                "type": "capture",
                "from": (r_from, c_from),
                "to": (r_to, c_to),
                "piece": piece_final,
                "captured": victim
            }

            print(
                f"Prise détectée : {piece_final}@({r_from},{c_from}) "
                f"x {victim}@({r_to},{c_to})"
            )
            break

        # ─────────────────────────────
        # Tout le reste → instable
        print("Coup incohérent ou instable → nouvelle capture")
        print(np.array(matrix_detected))
        continue

    return matrix_new, moves

# def wait_player_move(matrix_state_current, matrix_state_post, capture_fn):
    """
    Wait until the human has moved by repeatedly capturing and averaging board states.

    Inputs:
        robot (NiryoRobot) – robot instance for timing and pose adjustments.
        matrix_state_current (list[list[str]] or np.ndarray) – the previous stable board matrix.
        matrix_state_post (list[list[str]] or np.ndarray or None) – initial post-capture matrix or None if a hand was detected.
        capture_fn (callable) – function to perform a stabilized capture returning (image, matrix, yolo_result).

    Outputs:
        matrix_state_post (list[list[str]]) – the first averaged board matrix that differs from matrix_state_current.
    """

    # Tant qu'on détecte une main ou que l'échiquier est inchangé, on refait une moyenne
    print("MATRIX STATE CURRENT")
    print(matrix_state_current)
    print("MATRIX STATE POST")
    print(matrix_state_post)
    while (matrix_state_post is None) or np.array_equal(matrix_state_current, matrix_state_post):
        if matrix_state_post is None:
            print("Main détectée")
        if matrix_state_post is not None and np.array_equal(matrix_state_current, matrix_state_post):
            print("Pas encore bougé")

        matrix_state_post = capture_fn()

    print("Nouvelle position jouée")
    return matrix_state_post

def player_has_white(corners, yolo_result):
    """
    Determine whether the human player has the white pieces by comparing distances from the top-left marker to detected piece centers.

    Inputs:
        corners (list[Marker]) – sorted list of 4 detected markers, with corners[0] being the top-left.

    Outputs:
        bool or None – True if white pieces are closer on average (human plays White), False if black pieces are closer (human plays Black), or None if insufficient detections to decide.
    """

    # Centre du marqueur en haut à gauche (NiryoMaker 1)
    marker_tl = np.array(corners[0].get_center(), dtype=float)

    # Récupération des tableaux NumPy depuis yolo_result
    names = yolo_result.names
    classes = yolo_result.boxes.cls.cpu().numpy().astype(int)
    boxes = yolo_result.boxes.xyxy.cpu().numpy()

    # Initialisation des distances minimales
    min_dist_white = float('inf')
    min_dist_black = float('inf')

    # Parcours de chaque détection YOLO
    for box, cls_idx in zip(boxes, classes):
        class_name = names[cls_idx]

        # On ignore toute détection de mains
        if class_name.lower() == "hand":
            continue

        # Détermination de la couleur d'après la casse du nom :
        # - minuscules → pièce noire
        # - majuscules → pièce blanche
        if class_name.islower():
            couleur = 'black'
        elif class_name.isupper():
            couleur = 'white'
        else:
            # Si le nom mélange minuscules/majuscules ou n'est pas clair, on l'ignore
            continue

        # Calcul du centre de la bounding box
        x1, y1, x2, y2 = box
        box_center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)

        # Distance euclidienne entre le marqueur TL et le centre de la box
        dist = np.linalg.norm(marker_tl - box_center)

        # Mise à jour de la distance minimale selon la couleur
        if couleur == 'white' and dist < min_dist_white:
            min_dist_white = dist
        elif couleur == 'black' and dist < min_dist_black:
            min_dist_black = dist

    # Si aucune pièce blanche ou aucune pièce noire n’a été détectée, on ne peut pas décider
    if min_dist_white == float('inf') or min_dist_black == float('inf'):
        return None

    # Si la distance minimale à une pièce blanche est > que celle à une pièce noire,
    # alors le joueur prendra les pièces blanches
    return min_dist_white < min_dist_black

def process_frame(frame, M, dims, boxes, virtual_boxes, sticky_state):
        """
        Traite une frame : détection et lissage.
        
        Args:
            frame: Image BGR capturée depuis la caméra
        
        Returns:
            dict: État lissé de l'échiquier {cell_name: letter}
        """
        # Warper la frame courante avec la même matrice que la détection initiale
        # print(f"OK: {sticky_state['f4']}")
        warp_frame = cv2.warpPerspective(frame, M, dims)
        
        # Appliquer le MÊME traitement d'image que dans test_image_to_board.py
        img_treatment = Img_treatment.ImgTreatment(warp_frame)
        treated_frame, _ = img_treatment.traitement_image()
        
        # Détecter les gommettes colorées dans les cases virtuelles
        detections = detect_colored_stickers(
            treated_frame,
            virtual_boxes,
            min_area_percent=50,
            color_ranges=COLOR_RANGES
        )
        
        frame_detections = {}

        for cell_name in boxes.keys():
            info = detections.get(cell_name)

            # if info is None:
            #     frame_detections[cell_name] = None
            #     continue

            color = info.get("color")
            side = info.get("side")

            if color is not None and side is not None:
                frame_detections[cell_name] = board_state_from_colored_stickers(color, side)
            else:
                frame_detections[cell_name] = " "
        
        # Récupérer l'état lissé
        new_sticky_state = update_sticky_state(sticky_state, frame_detections)
        stable_board_state = get_stable_board_state(new_sticky_state)
        
        return stable_board_state

def update_sticky_state(sticky_state, detections):
    """
    detections : dict {cell_name: detected_letter or None}
    """
    for cell, detected in detections.items():
        state = sticky_state[cell]
        stable = state["stable"]
        candidate = state["candidate"]

        # Cas stable confirmé
        if detected == stable:
            state["candidate"] = None
            state["count"] = 0
            continue

        # Ignorer None si stable existe (raté ponctuel)
        if detected == " " and stable != " ":
            state["count"] += 1


        # Nouveau candidat
        if detected != candidate:
            state["candidate"] = detected
            state["count"] = 1
        else:
            state["count"] += 1

        # Validation du changement
        if state["count"] >= MIN_CONFIRM:
            state["stable"] = state["candidate"]
            state["candidate"] = None
            state["count"] = 0
    
    return sticky_state

def get_stable_board_state(sticky_state):
    return {cell: data["stable"] for cell, data in sticky_state.items()}

# def get_smoothed_state(boxes, detection_history):
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
    
    for cell_name in boxes.keys():
        history = detection_history[cell_name]
        
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

def dict_to_board_matrix(state):
    """
    Transforme un dictionnaire {'a1': 'p'} en matrice 8x8
    avec [0,0] = a8 et [7,7] = h1
    """
    board = [[" " for _ in range(8)] for _ in range(8)]

    for square, piece in state.items():
        if square is None:
            continue

        file = ord(square[0]) - ord('a')      # a→0, b→1, ...
        rank = int(square[1])                  # '1' → 1

        row = 8 - rank                         # 8→0, 1→7
        col = file

        board[row][col] = piece if piece is not None else " "

    return board

def matrix_to_fen(matrix, side_to_move):
    fen_rows = []
    for row in matrix:
        empty = 0
        fen_row = ""
        for cell in row:
            if cell == " ":
                empty += 1
            else:
                if empty:
                    fen_row += str(empty)
                    empty = 0
                fen_row += cell
        if empty:
            fen_row += str(empty)
        fen_rows.append(fen_row)

    fen = "/".join(fen_rows)
    fen += f" {'w' if side_to_move == 0 else 'b'} - - 0 1"
    return fen

def capture_and_get_state(cap, boxes, virtual_boxes, M, dims, sticky_state):
    frame_count = 0
    
    while frame_count < FRAME_COUNT:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Impossible de capturer.")
            break
        
        # Traiter la frame
        board_state = process_frame(frame, M, dims, boxes, virtual_boxes, sticky_state)
        frame_count += 1

    board_state_matrix = dict_to_board_matrix(board_state)
    
    return board_state_matrix

# ─────────────────────────────────────────────────────────────────────────────
# ROBOT MOVES FUNCTIONS (pickup, place, capture are encapsulated in play_move)
# ─────────────────────────────────────────────────────────────────────────────

def pickup_with_electromagnet(robot, position, piece_type):
    """
    Pick up a chess piece using the electromagnet at a specified board position.

    Inputs:
        robot (NiryoRobot) – the robot instance controlling movement and electromagnet.
        pick_pose (PoseObject) – target pose above the square where the piece is located.
        piece_type (str) – one-character code for the piece (e.g. 'P', 'n'), used to look up its height.

    Outputs:
        None – the robot moves to the pick_pose, engages the electromagnet, and lifts the piece by a fixed offset.
    """
    h = PIECE_HEIGHTS[piece_type]
    x_rel = position[0]
    y_rel = position[1]
    p = PoseObject(x_rel, y_rel, BOARD_THICKNESS + SHIFT_DIST_EMPTY, 0.0, math.pi/2, 0.0)
    robot.move(p)
    robot.shift_pose(RobotAxis.Z, -(SHIFT_DIST_EMPTY-h))
    robot.pull_air_vacuum_pump()
    robot.activate_electromagnet(ELECTROMAGNET_PIN)
    robot.wait(1)
    robot.shift_pose(RobotAxis.Z, SHIFT_DIST_EMPTY)

def place_with_electromagnet(robot, position, piece_type):
    """
    Place a chess piece at a specified board position using the electromagnet.

    Inputs:
        robot (NiryoRobot) – the robot instance controlling movement and electromagnet.
        target_pose (PoseObject) – target pose above the square where the piece should be placed.
        piece_type (str) – one-character code for the piece (e.g. 'P', 'n'), used to look up its height.

    Outputs:
        None – the robot moves to the target_pose, lowers the piece into place, deactivates the electromagnet, and retracts.
    """
    h = PIECE_HEIGHTS[piece_type]
    x_rel = position[0]
    y_rel = position[1]
    p = PoseObject(x_rel, y_rel, BOARD_THICKNESS + SHIFT_DIST_EMPTY + h, 0.0, math.pi/2, 0.0)
    robot.move(p)
    robot.shift_pose(RobotAxis.Z, -SHIFT_DIST_EMPTY)
    robot.deactivate_electromagnet(ELECTROMAGNET_PIN)
    robot.wait(1)
    robot.shift_pose(RobotAxis.Z,  SHIFT_DIST_EMPTY-h)

def capture_piece(robot, pick_pose, target_pose, piece_type):
    """
    Perform a capture by picking up a piece from one position and placing it at another using the electromagnet.

    Inputs:
        robot (NiryoRobot) – the robot instance controlling movement and electromagnet.
        pick_pose (PoseObject) – pose above the square of the piece to capture.
        target_pose (PoseObject) – pose above the destination square (buffer or opponent square).
        piece_type (str) – one-character code for the piece (e.g. 'P', 'n'), used to look up its height.

    Outputs:
        None – the robot executes a pickup then place sequence to move the captured piece.
    """

    pickup_with_electromagnet(robot, pick_pose, piece_type)
    place_with_electromagnet(robot, target_pose, piece_type)

def play_move(
    robot: NiryoRobot,
    from_sq: tuple,
    to_sq: tuple,
    move_type: str,
    tiles_positions: dict,
    buffer_positions: dict,
    buffer_state: dict
):
    """
    Execute a chess move physically with the robot, handling normal moves, captures (with buffering), and all castling cases.

    Inputs:
        robot (NiryoRobot) – the robot instance controlling motions and electromagnet.
        from_sq (tuple) – (piece_type, origin_square), e.g. ('P', 'e2').
        to_sq (tuple) – (captured_or_moved_piece, destination_square), e.g. ('p', 'd5') or ('P', 'e4').
        move_type (str or None) – None for a quiet move; square name for captures; 'O-O', 'O-O-O', 'o-o', 'o-o-o' for castling.
        tiles_positions (dict[str, PoseObject]) – mapping of board squares to robot poses.
        buffer_positions (dict[str, PoseObject]) – mapping of buffer slots to robot poses.
        buffer_state (dict[str, str or None]) – tracks which buffer slots are occupied by which piece.

    Outputs:
        None – the robot performs the necessary pickup, placement, and buffer operations to realize the move.
    """

    # (1) Traitement du roque (4 cas)
    if move_type == 'O-O':
        # Petit roque blanc
        pickup_with_electromagnet(robot, tiles_positions['h1'], 'r')
        place_with_electromagnet(robot, tiles_positions['f1'], 'r')
        pickup_with_electromagnet(robot, tiles_positions['e1'], 'k')
        place_with_electromagnet(robot, tiles_positions['g1'], 'k')

    elif move_type == 'O-O-O':
        # Grand roque blanc
        pickup_with_electromagnet(robot, tiles_positions['a1'], 'r')
        place_with_electromagnet(robot, tiles_positions['d1'], 'r')
        pickup_with_electromagnet(robot, tiles_positions['e1'], 'k')
        place_with_electromagnet(robot, tiles_positions['c1'], 'k')

    elif move_type == 'o-o':
        # Petit roque noir
        pickup_with_electromagnet(robot, tiles_positions['h8'], 'r')
        place_with_electromagnet(robot, tiles_positions['f8'], 'r')
        pickup_with_electromagnet(robot, tiles_positions['e8'], 'k')
        place_with_electromagnet(robot, tiles_positions['g8'], 'k')

    elif move_type == 'o-o-o':
        # Grand roque noir
        pickup_with_electromagnet(robot, tiles_positions['a8'], 'r')
        place_with_electromagnet(robot, tiles_positions['d8'], 'r')
        pickup_with_electromagnet(robot, tiles_positions['e8'], 'k')
        place_with_electromagnet(robot, tiles_positions['c8'], 'k')

    # (2) Coup normal (move_type None)
    elif move_type is None:
        piece_type_to_move, pick_tile = from_sq
        print(f"OK: {piece_type_to_move}")
        _, place_tile = to_sq
        pickup_with_electromagnet(robot, tiles_positions[pick_tile], piece_type_to_move)
        place_with_electromagnet(robot, tiles_positions[place_tile], piece_type_to_move)

    # (3) Coup de prise (move_type contient 'd5' par ex.)
    else:
        piece_to_move_type, piece_to_move_tile = from_sq
        piece_to_capture_type, piece_to_place_tile = to_sq
        captured_tile = move_type  # par ex. 'd5'

        # 3.1) Trouve la première case libre du buffer
        free_buffer = None
        for buf_tile, occupant in buffer_state.items():
            if occupant is None:
                free_buffer = buf_tile
                break
        if free_buffer is None:
            raise RuntimeError("Buffer plein ! Impossible de stocker la pièce capturée.")

        # 3.2) On capture la pièce vers le buffer
        capture_piece(robot,
                      tiles_positions[captured_tile],
                      buffer_positions[free_buffer],
                      piece_to_capture_type)
        buffer_state[free_buffer] = piece_to_capture_type

        # 3.3) On déplace la pièce captante sur la case désormais libre
        pickup_with_electromagnet(robot,
                                  tiles_positions[piece_to_move_tile],
                                  piece_to_move_type)
        place_with_electromagnet(robot,
                                 tiles_positions[piece_to_place_tile],
                                 piece_to_move_type)


def main():
    # Robot initialisation
    robot = NiryoRobot(ROBOT_IP)
    robot.clear_collision_detected()
    robot.calibrate_auto()
    robot.update_tool()
    robot.setup_electromagnet(ELECTROMAGNET_PIN)

    capture = initialize_camera()
    M = None  # Matrice de transformation homographique
    M_inv = None  # Matrice inverse pour reprojeter vers l'image brute
    dims = None  # Dimensions de l'échiquier warpé

    # AI model loading
    STOCKFISH_PATH = "src/ChessUtils/model_data/stockfish/stockfish-macos-m1-apple-silicon"
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    print("Modèle IA chargé")
   
    # Robot's wait pose object initialisation (SetUp in Niryo Studio)
    wait_pose = PoseObject(0.134, 0.0, 0.150, 0.0, math.pi/2, 0.0)
    robot.move(wait_pose)

    # Calibration instructions
    init_ok = False
    print("[PROCESSING] Détection des marqueurs...")
    while True:
        ret, img_und = capture.read()
        if not ret:
            print("[ERROR] Impossible de capturer.")
            break
        try:
            boxes, virtual_boxes, init_ok, M, dims = detect_board(img_und)
            if init_ok:
                break
        except ValueError:
            print("Impossible de détecter les marqueurs.")
            robot.wait(2) # slowing down capture process
     
    # Player side detection
    input("→ Placez les pièces et appuyez sur Entrée…")
    player_plays_white = True
    # player_plays_white = player_has_white(corners, yolo_res)
    # print(f"Joueur = {'Blancs' if player_plays_white else 'Noirs'}")

    # Board and buffer tiles poses calculation based on NiryoMarkerTL pose
    board_tiles = {}
    for file in 'abcdefgh':
        for rank in '12345678':
            sq = f"{file}{rank}"
            board_tiles[sq] = chess_square_to_rel(sq, player_plays_white)
    buffer_tiles = {f"{c}{r}": None for c in '1234' for r in '1234'}
    buffer_positions = set_buffer_tiles_positions(buffer_tiles)
    # for pose in buffer_positions.values():
    #     robot.move(pose)
    # Empty Buffer on start (you can place a spare queen in it)
    buffer_state = {k: None for k in buffer_positions}

     # DEBUG2
    # boucle sur toutes les colonnes et rangées
    # piece_type = 'k'
    # for file in 'a':
    #     for rank in '2':
    #         sq = f"{file}{rank}"
    #         x_rel, y_rel = chess_square_to_rel(sq)
    #         print(f"Case {sq} → x_rel={x_rel:.4f}, y_rel={y_rel:.4f}")
    #         # x_rel_off, y_rel_off = apply_magnet_offset(x_rel, y_rel)
    #         try:
    #             print(f"→ Test sur la case {sq}")
    #             # on approche, on attrape, puis on repose
    #             pickup_with_electromagnet(robot, x_rel, y_rel, piece_type)
    #             robot.wait(1)                     
    #             place_with_electromagnet(robot, x_rel, y_rel, piece_type)
    #             robot.wait(1)
    #         except Exception as e:
    #             print(f" Erreur sur {sq} : {e}")

    # First board state detection
    input("Appuyez sur Entrée pour commencer la partie")
    if player_plays_white : 
            matrix_current = [['r','n','b','q','k','b','n','r'],
                    ['p','p','p','p','p','p','p','p'],
                    [' ',' ',' ',' ',' ',' ',' ',' '],
                    [' ',' ',' ',' ',' ',' ',' ',' '],
                    [' ',' ',' ',' ',' ',' ',' ',' '],
                    [' ',' ',' ',' ',' ',' ',' ',' '],
                    ['P','P','P','P','P','P','P','P'],
                    ['R','N','B','Q','K','B','N','R']]
    else : 
        matrix_current = [['R','N','B','Q','K','B','N','R'],
                    ['P','P','P','P','P','P','P','P'],
                    [' ',' ',' ',' ',' ',' ',' ',' '],
                    [' ',' ',' ',' ',' ',' ',' ',' '],
                    [' ',' ',' ',' ',' ',' ',' ',' '],
                    [' ',' ',' ',' ',' ',' ',' ',' '],
                    ['p','p','p','p','p','p','p','p'],
                    ['r','n','b','q','k','b','n','r']]
        
    sticky_state = init_sticky_state(boxes)
    capture_fn = lambda: capture_and_get_state(capture, boxes, virtual_boxes, M, dims, sticky_state)
    matrix_depart= capture_fn()
    matrix_depart = np.array(matrix_depart)
    print (matrix_depart)

    max_tries = FRAME_COUNT
    tries = 0

    while not np.array_equal(matrix_depart, matrix_current) : 
        print(matrix_depart)
        print("Si le pion est bien placé mais mal détecté, il y a un problème de luminosité")
        input("Appuyez sur Entrée pour un nouvel essai")
        matrix_depart= capture_fn()
        matrix_depart = np.array(matrix_depart)
        tries += 1

        if tries >= max_tries:
            raise RuntimeError("Luminosité instable : plateau non fiable")
        
    # Softaware's board initialisation
    b = ChessBoard()
    b.current_board = np.array(matrix_current)
    b.player = 0  # Always white first
    

    # Game loop
    while game_result(b) is None:
        is_human_turn = (b.player == 0 and player_plays_white) or \
                        (b.player == 1 and not player_plays_white)

        if is_human_turn:
            # Wait until player plays legal move
            matrix_new, _ = wait_player_move(matrix_current, capture_fn)
            print(np.array(matrix_new))
            from_sq, to_sq, move_type = b.get_move_details(
                np.array(matrix_current), np.array(matrix_new)
            )
            # Updtating board object
            b.move_piece(
                square_to_index(from_sq[1]),
                square_to_index(to_sq[1])
            )
            matrix_current = matrix_new

        else:
            # Robot/AI turn
            old_matrix = b.current_board.copy()
            print(old_matrix)

            # MCTS + net
            fen = matrix_to_fen(matrix_current, b.player)
            board_sf = chess.Board(fen)

            result = engine.play(
                board_sf,
                chess.engine.Limit(time=0.3)  # 300 ms → largement suffisant
            )

            move = result.move
            from_sq_sf, to_sq_sf = stockfish_move_to_robot(move)
            old_matrix = np.array(matrix_current)
            new_matrix = old_matrix.copy()

            r_from, c_from = square_to_index(from_sq_sf)
            r_to, c_to = square_to_index(to_sq_sf)

            new_matrix[r_to][c_to] = new_matrix[r_from][c_from]
            new_matrix[r_from][c_from] = " "
            # Move details gathering
            from_sq, to_sq, move_type = b.get_move_details(old_matrix,new_matrix)

            # Move execution with Ned2
            play_move(
                robot, from_sq, to_sq, move_type,
                board_tiles, buffer_positions, buffer_state
            )
            robot.move(wait_pose)

            b.current_board = new_matrix
            b.player = 1 - b.player
            matrix_current = capture_fn()

    # Game ends
    print(f"### Résultat : {game_result(b)} ###")
    engine.quit()
    robot.move(wait_pose)
    robot.close_connection()

if __name__ == "__main__":
    main()