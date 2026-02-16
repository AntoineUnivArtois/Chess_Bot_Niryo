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
    visualize_real_and_virtual_grids,
    Img_treatment,
    ChessTeacher,
    MarkerValidator,
    ChessFeedbackGenerator,
    COLOR_RANGES

)

# ─────────────────────────────────────────────────────────────────────────────
# II. ROBOT CONFIG
# ─────────────────────────────────────────────────────────────────────────────

ROBOT_IP = '169.254.200.200' # Direct Ethernet
CAMERA_ID = 0
CELL_SIZE = 0.04  # 4 cm
ELECTROMAGNET_PIN = 'DO4'
BOARD_SIZE = CELL_SIZE*8   # m
BOARD_THICKNESS = 0.013 # Épaisseur du plateau en m

COORD_BASE_MM = np.array([
    [436, -148],
    [442, 126],
    [154, -141],
    [158, 124]
], dtype=np.float32)
PTS_IDEAUX = np.array([[0, 0],[7, 0],[0, 7],[7, 7]], dtype=np.float32)
H, _ = cv2.findHomography(PTS_IDEAUX, COORD_BASE_MM)

# Individual pieces heights in m
PIECE_HEIGHTS = {
    'P': 0.042, 'p': 0.042,
    'N': 0.053, 'n': 0.053,
    'B': 0.055, 'b': 0.055,
    'R': 0.047, 'r': 0.047,
    'Q': 0.062, 'q': 0.062,
    'K': 0.074, 'k': 0.074,
}

SHIFT_DIST = 0.015       # Distance de sécurité
SHIFT_DIST_EMPTY  = PIECE_HEIGHTS['k'] + SHIFT_DIST # Distance de déplacement à vide (au dessus de la plus grande pièce)

FRAME_COUNT = 8        
MIN_CONFIRM = int(FRAME_COUNT * 0.6)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def square_to_index(sq: str):
    """Ex. 'e2' → (6,4) (line i, column j in current_board)."""
    file, rank = sq[0], int(sq[1])
    j = ord(file) - ord('a')
    i = 8 - rank
    return i, j

def index_to_square(i, j):
    """Convertit des indices (i:0–7, j:0–7) en case algébrique, ex. (7,4) → 'e1'."""
    file = chr(ord('a') + j)
    rank = str(8 - i)
    return file + rank

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

def invert_matrix(matrix):
    return [row[::-1] for row in matrix[::-1]]

def init_sticky_state(boxes):
    sticky = {}
    for cell in boxes.keys():
        sticky[cell] = {
            "stable": " ",
            "candidate": None,
            "count": 0
        }
    return sticky

def update_sticky_state(sticky_state, detections):
    """Met à jour l'état lissé avec les nouvelles détections"""
    for cell, detected in detections.items():
        state = sticky_state[cell]
        stable = state["stable"]
        candidate = state["candidate"]

        # Cas stable confirmé
        if detected == stable:
            state["candidate"] = None
            state["count"] = 0
            continue

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

def stockfish_move_to_robot(move):
    from_sq = chess.square_name(move.from_square)
    to_sq = chess.square_name(move.to_square)
    return from_sq, to_sq

def board_to_matrix(board: chess.Board):
    """
    Convertit un chess.Board en matrice 8x8 compatible vision
    [0][0] = a8, [7][7] = h1
    """
    matrix = [[" " for _ in range(8)] for _ in range(8)]

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank = chess.square_rank(square)
            file = chess.square_file(square)

            row = 7 - rank
            col = file

            matrix[row][col] = piece.symbol()

    return matrix
# ─────────────────────────────────────────────────────────────────────────────
# VISION -> PYTHON-CHESS
# ─────────────────────────────────────────────────────────────────────────────

def detect_move_from_matrices(
    before: np.ndarray, 
    after: np.ndarray,
    board: chess.Board,
) -> tuple:
    """
    Détecte le coup joué en comparant deux matrices.
    
    Returns:
        (chess.Move, piece_moved, captured_piece) ou (None, None, None)
    """
    diffs = []
    for i in range(8):
        for j in range(8):
            if before[i, j] != after[i, j]:
                diffs.append((i, j))
    
    if len(diffs) == 0:
        return None, None, None
    
    # Détection du roque (4 cases changées)
    if len(diffs) == 4:
        # Petit roque blanc
        if (7,4) in diffs and (7,6) in diffs and before[7,4]=='K' and after[7,6]=='K':
            move = chess.Move.from_uci("e1g1")
            if move in board.legal_moves:
                return move, 'K', None
        
        # Grand roque blanc
        if (7,4) in diffs and (7,2) in diffs and before[7,4]=='K' and after[7,2]=='K':
            move = chess.Move.from_uci("e1c1")
            if move in board.legal_moves:
                return move, 'K', None
        
        # Petit roque noir
        if (0,4) in diffs and (0,6) in diffs and before[0,4]=='k' and after[0,6]=='k':
            move = chess.Move.from_uci("e8g8")
            if move in board.legal_moves:
                return move, 'k', None
        
        # Grand roque noir
        if (0,4) in diffs and (0,2) in diffs and before[0,4]=='k' and after[0,2]=='k':
            move = chess.Move.from_uci("e8c8")
            if move in board.legal_moves:
                return move, 'k', None
    
    # Coup simple : trouver départ et arrivée
    from_pos = None
    to_pos = None
    piece_moved = None
    captured = None
    
    for i, j in diffs:
        # Case qui se vide → départ
        if before[i, j] != ' ' and after[i, j] == ' ':
            from_pos = (i, j)
            piece_moved = before[i, j]
        # Case qui se remplit → arrivée
        elif before[i, j] == ' ' and after[i, j] != ' ':
            to_pos = (i, j)
        # Case qui change de pièce → capture
        elif before[i, j] != ' ' and after[i, j] != ' ' and before[i, j] != after[i, j]:
            # C'est la case d'arrivée
            if after[i, j] != ' ':
                to_pos = (i, j)
                captured = before[i, j]
    
    if from_pos is None or to_pos is None:
        return None, None, None
    
    # Conversion en notation UCI
    from_square = chess.square(from_pos[1], 7 - from_pos[0])
    to_square = chess.square(to_pos[1], 7 - to_pos[0])
    
    # Tester le coup normal
    move = chess.Move(from_square, to_square)
    if move in board.legal_moves:
        return move, piece_moved, captured
    
    # Tester les promotions
    for promotion in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
        move_promo = chess.Move(from_square, to_square, promotion)
        if move_promo in board.legal_moves:
            return move_promo, piece_moved, captured
    
    return None, None, None

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

def estimate_camera_intrinsics(img_shape, fov_deg=90):
    """
    Estime les intrinsics avec un FOV plus réaliste.
    
    Args:
        img_shape: (height, width) de l'image
        fov_deg: champ de vision horizontal estimé (60° par défaut)
    
    Returns:
        camera_matrix, dist_coeffs
    """
    h, w = img_shape[:2]
    
    # Focale basée sur le FOV (plus précis que max(w,h))
    f = w / (2 * np.tan(np.radians(fov_deg) / 2))
    
    # Centre optique (hypothèse : centre de l'image)
    cx = w / 2.0
    cy = h / 2.0
    
    camera_matrix = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Coefficients de distorsion (zéro si caméra non calibrée)
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    
    return camera_matrix, dist_coeffs

def detect_board(frame):
    """
    Détecte l'échiquier une seule fois.
    
    """
    # Extraction des marqueurs
    warp, M, dims, corners, margin_px = extract_img_markers_with_margin(
        frame,
        workspace_ratio=1.0,
        base_size=810,
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
    
    # Coins 3D du plateau réel
    real_corners_3d = np.array([
        [0.0, 0.0, 0.0],       # tl
        [320.0, 0.0, 0.0],     # tr
        [320.0, 320.0, 0.0],   # br
        [0.0, 320.0, 0.0],     # bl
    ], dtype=np.float64)
    
    # Estimation simple des intrinsics
    camera_matrix, dist_coeffs = estimate_camera_intrinsics(warp.shape)
    
    # Créer la grille virtuelle projetée
    virtual_boxes, _ = create_virtual_detection_grid(
        real_corners_3d,
        image_corners,
        camera_matrix,
        dist_coeffs=dist_coeffs,
        grid_size=8,
        plane_height_cm=24.0,
        margin_percent=0.05
    )
    
    board_detected = True
    print("[SUCCESS] Échiquier détecté avec succès ✓\n")

    return warp, boxes, virtual_boxes, board_detected, M, dims

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
        update_sticky_state(sticky_state, frame_detections)

def capture_and_get_state(cap, boxes, virtual_boxes, M, dims, sticky_state):
    """Capture plusieurs frames et retourne l'état lissé"""
    for _ in range(FRAME_COUNT):
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Impossible de capturer.")
            break

        process_frame(frame, M, dims, boxes, virtual_boxes, sticky_state)

    stable_state = get_stable_board_state(sticky_state)
    board_state_matrix = dict_to_board_matrix(stable_state)
    
    return board_state_matrix

def wait_player_move(board, matrix_current, capture_fn, player_has_white):
    """Attend qu'un coup humain valide soit détecté"""
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

                if before != " " and after == " ":
                    removed.append((r, c, before))
                elif before == " " and after != " ":
                    added.append((r, c, after))

                elif before != " " and after != " " and before != after :
                    replaced.append((r, c, before, after))

        if len(removed) == 2 and len(added) == 2:
            removed_squares = {(r, c) for r, c, _ in removed}
            added_squares   = {(r, c) for r, c, _ in added}

            for move in board.legal_moves:
                if board.is_castling(move):
                    from_sq = square_to_index(chess.square_name(move.from_square))
                    to_sq   = square_to_index(chess.square_name(move.to_square))
                    if not player_has_white:
                        from_sq = (7 - from_sq[0], 7 - from_sq[1])
                        to_sq   = (7 - to_sq[0], 7 - to_sq[1])

                    if from_sq in removed_squares and to_sq in added_squares:
                        print("Roque détecté")
                        board.push(move)
                        return board_to_matrix(board)

        # Vérifications de validité
        if not removed and not added and not replaced:
            print("Pas encore bougé…")
            continue

        if len(removed) == 1 and not added and not replaced:
            print("Disparition isolée détectée → nouvelle capture")
            continue
        
        if not removed and len(replaced) == 1:
            print("Changement de casse isolé → ignoré")
            continue

        # Déplacement simple
        if len(removed) == 1 and len(added) == 1 and not replaced:
            r_from, c_from, piece = removed[0]
            r_to, c_to, _ = added[0]

            matrix_new = [row.copy() for row in matrix_current]
            matrix_new[r_from][c_from] = " "
            matrix_new[r_to][c_to] = piece  # casse LOGIQUE conservée

            valid_move = True
            print(f"Coup détecté : {piece}@({r_from},{c_from}) → ({r_to},{c_to})")
            return matrix_new

        # Prise
        if len(removed) == 1 and len(replaced) == 1 and not added:
            r_from, c_from, piece = removed[0]
            r_to, c_to, victim, detected = replaced[0]

            piece_final = detected if detected.lower() == piece.lower() else piece

            matrix_new = [row.copy() for row in matrix_current]
            matrix_new[r_from][c_from] = " "
            matrix_new[r_to][c_to] = piece_final

            valid_move = True
            print(f"Prise détectée : {piece_final} x {victim} sur {index_to_square(r_from,c_from)} ")
            return matrix_new

        print("Coup instable → nouvelle capture")
        print(np.array(matrix_detected))
        continue

def player_has_white(virtual_boxes, capture_fn):
    """
    Determine whether the human player has the white pieces by comparing distances from the top-left marker to detected piece centers.

    Inputs:
        virtual_boxes (dict) – a dictionary mapping virtual box names to their positions.
        capture_fn (function) – a function that returns the current state of the board.

    Outputs:
        bool or None – True if white pieces are closer on average (human plays White), False if black pieces are closer (human plays Black), or None if insufficient detections to decide.
    """
    # Initialisation des distances minimales
    min_dist_white = float('inf')
    min_dist_black = float('inf')

    matrix_detected = capture_fn()
    tl = virtual_boxes['a8'][0]

    for i_line,line in enumerate(matrix_detected):
        for i_piece,piece in enumerate(line):
            if piece.islower():
                couleur = 'black'
            elif piece.isupper():
                couleur = 'white'
            else:
                continue

            box = square_to_index(f"{i_piece}{i_line}")
            box_center = case_center(box[0], box[1])

            # Distance euclidienne entre le marqueur TL et le centre de la box
            dist = np.linalg.norm(tl - box_center)

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

# ─────────────────────────────────────────────────────────────────────────────
# ROBOTS
# ─────────────────────────────────────────────────────────────────────────────

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

def place_with_electromagnet(robot, position, piece_type, if_buffer = False):
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
    if if_buffer:
        robot.shift_pose(RobotAxis.Z,  SHIFT_DIST_EMPTY-h-0.006)
    else :
        robot.shift_pose(RobotAxis.Z,  SHIFT_DIST_EMPTY-h)

def execute_move(
    robot, 
    move: chess.Move, 
    board: chess.Board,
    piece_moved: str,
    captured_piece: str,
    board_tiles: dict,
    buffer_positions: dict,
    buffer_state: dict
):
    """
    Exécute physiquement un coup avec le robot.
    
    Args:
        move: Le coup chess.Move à jouer
        piece_moved: La pièce déplacée ('P', 'n', etc.)
        captured_piece: La pièce capturée ou None
        board: chess.Board pour détecter les roques
    """
    from_square = chess.square_name(move.from_square)
    to_square = chess.square_name(move.to_square)
    
    # Détection du roque
    if board.is_castling(move):
        # Identifier quel roque
        if to_square == 'g1':  # Petit roque blanc
            pickup_with_electromagnet(robot, board_tiles['h1'], 'R')
            place_with_electromagnet(robot, board_tiles['f1'], 'R')
            pickup_with_electromagnet(robot, board_tiles['e1'], 'K')
            place_with_electromagnet(robot, board_tiles['g1'], 'K')
        
        elif to_square == 'c1':  # Grand roque blanc
            pickup_with_electromagnet(robot, board_tiles['a1'], 'R')
            place_with_electromagnet(robot, board_tiles['d1'], 'R')
            pickup_with_electromagnet(robot, board_tiles['e1'], 'K')
            place_with_electromagnet(robot, board_tiles['c1'], 'K')
        
        elif to_square == 'g8':  # Petit roque noir
            pickup_with_electromagnet(robot, board_tiles['h8'], 'r')
            place_with_electromagnet(robot, board_tiles['f8'], 'r')
            pickup_with_electromagnet(robot, board_tiles['e8'], 'k')
            place_with_electromagnet(robot, board_tiles['g8'], 'k')
        
        elif to_square == 'c8':  # Grand roque noir
            pickup_with_electromagnet(robot, board_tiles['a8'], 'r')
            place_with_electromagnet(robot, board_tiles['d8'], 'r')
            pickup_with_electromagnet(robot, board_tiles['e8'], 'k')
            place_with_electromagnet(robot, board_tiles['c8'], 'k')
        
        return
    
    # Gestion des captures
    if captured_piece is not None or board.is_capture(move):
        # Trouver un buffer libre
        free_buffer = None
        for buf_tile, occupant in buffer_state.items():
            if occupant is None:
                free_buffer = buf_tile
                break
        
        if free_buffer is None:
            raise RuntimeError("Buffer plein !")
        
        # Identifier la pièce capturée
        if captured_piece:
            piece_to_capture = captured_piece
        else:
            # En passant ou autre cas
            captured_square = chess.square_name(move.to_square)
            i, j = square_to_index(captured_square)
            piece_to_capture = board.piece_at(move.to_square)
            if piece_to_capture:
                piece_to_capture = piece_to_capture.symbol()
            else:
                piece_to_capture = 'p' if board.turn == chess.WHITE else 'P'
        
        # Déplacer la pièce capturée au buffer
        pickup_with_electromagnet(robot, board_tiles[to_square], piece_to_capture)
        place_with_electromagnet(robot, buffer_positions[free_buffer], piece_to_capture, True)
        buffer_state[free_buffer] = piece_to_capture
    
    # Déplacement normal
    pickup_with_electromagnet(robot, board_tiles[from_square], piece_moved)
    
    # Gestion de la promotion
    if move.promotion:
        promotion_map = {
            chess.QUEEN: 'Q' if board.turn == chess.BLACK else 'q',
            chess.ROOK: 'R' if board.turn == chess.BLACK else 'r',
            chess.BISHOP: 'B' if board.turn == chess.BLACK else 'b',
            chess.KNIGHT: 'N' if board.turn == chess.BLACK else 'n',
        }
        piece_moved = promotion_map.get(move.promotion, piece_moved)
    
    place_with_electromagnet(robot, board_tiles[to_square], piece_moved)

def main():
    # Robot initialisation
    robot = NiryoRobot(ROBOT_IP)
    robot.clear_collision_detected()
    robot.calibrate_auto()
    robot.update_tool()
    robot.set_arm_max_velocity(100)
    robot.setup_electromagnet(ELECTROMAGNET_PIN)

    capture = initialize_camera()

    # Robot's wait pose object initialisation (SetUp in Niryo Studio)
    wait_pose = PoseObject(0.134, 0.0, 0.150, 0.0, math.pi/2, 0.0)
    robot.move(wait_pose)

    # Calibration instructions
    print("[PROCESSING] Détection des marqueurs...")
    boxes, virtual_boxes, M, dims = None, None, None, None
    while True:
        ret, img_und = capture.read()
        if not ret:
            print("[ERROR] Impossible de capturer")
            break

        try:
            warp, boxes, virtual_boxes, init_ok, M, dims = detect_board(img_und)
            if not init_ok:
                continue

            vis = visualize_real_and_virtual_grids(warp, boxes, virtual_boxes)

            # Affiche l'interface Tkinter pour valider
            validator = MarkerValidator(vis)
            if validator.result:
                print("[OK] Marqueurs validés")
                break
            else:
                print("[Retry] Nouvelle tentative")
                continue

        except ValueError as e:
            continue

    # AI model loading
    STOCKFISH_PATH = "src/ChessUtils/model_data/stockfish/stockfish-macos-m1-apple-silicon"
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": 1320})
    teacher = ChessTeacher(engine)
    feedback = ChessFeedbackGenerator()
    print("✓ Stockfish chargé")

    sticky_state = init_sticky_state(boxes)
    capture_fn = lambda: capture_and_get_state(capture, boxes, virtual_boxes, M, dims, sticky_state)
   
    # Player side detection
    input("→ Placez les pièces et appuyez sur Entrée…")
    # player_plays_white = True
    player_plays_white = player_has_white(virtual_boxes, capture_fn)
    print(f"Joueur = {'Blancs' if player_plays_white else 'Noirs'}")

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
        matrix_current = [['R','N','B','K','Q','B','N','R'],
                    ['P','P','P','P','P','P','P','P'],
                    [' ',' ',' ',' ',' ',' ',' ',' '],
                    [' ',' ',' ',' ',' ',' ',' ',' '],
                    [' ',' ',' ',' ',' ',' ',' ',' '],
                    [' ',' ',' ',' ',' ',' ',' ',' '],
                    ['p','p','p','p','p','p','p','p'],
                    ['r','n','b','k','q','b','n','r']]
        
    matrix_depart= capture_fn()
    matrix_depart = np.array(matrix_depart)

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
            raise RuntimeError("Luminosité instable : plateau non fiable. Recommencez l'initialisation.")
        
        
    # Software's board initialisation
    b = chess.Board()
    
    # Game loop
    while not b.is_game_over():
        is_human_turn = (b.turn == chess.WHITE and player_plays_white) or \
                        (b.turn == chess.BLACK and not player_plays_white)

        if is_human_turn:
            # Wait until player plays legal move
            matrix_view = wait_player_move(b, matrix_current, capture_fn, player_plays_white)

            if player_plays_white:
                matrix_calcul = matrix_view
            else:
                matrix_current = invert_matrix(matrix_current)
                matrix_calcul = invert_matrix(matrix_view)

            move, piece_moved, captured_piece = detect_move_from_matrices(
                np.array(matrix_current), np.array(matrix_calcul),b
            )

            # Updtating board object
            if move and move in b.legal_moves:
                # b.push(move)
                b, result = teacher.analyse_move(b, move, player_plays_white)
                print(result["classification"])
                print(result["delta"])
                print(feedback.generate(result["reason"]))
                matrix_current = board_to_matrix(b)

            elif not player_plays_white:
                matrix_current = invert_matrix(matrix_current)

        else:
            # Robot/AI turn
            result = engine.play(b, chess.engine.Limit(time=0.3))
            move = result.move
            
            # Extraire les coordonnées pour le robot
            from_sq = chess.square_name(move.from_square)
            to_sq = chess.square_name(move.to_square)
            r_from, c_from = square_to_index(from_sq)
            r_to, c_to = square_to_index(to_sq)

            piece_moved = matrix_current[r_from][c_from]
            captured_piece = None

            if b.is_capture(move):
                captured_piece = matrix_current[r_to][c_to]
                if captured_piece == " ":  # Cas en-passant
                    ep_row = r_from
                    ep_col = c_to
                    captured_piece = matrix_current[ep_row][ep_col]

            # Move execution with Ned2
            execute_move(
                robot, move, b, piece_moved, captured_piece,
                board_tiles, buffer_positions, buffer_state
            )

            b.push(move)

            matrix_current = board_to_matrix(b)

            if not player_plays_white:
                matrix_current = invert_matrix(matrix_current)
            print(np.array(matrix_current))

            robot.move(wait_pose)

    # Game ends
    print(f"### Résultat : {b.result()} ###")
    engine.quit()
    robot.move(wait_pose)
    robot.close_connection()

if __name__ == "__main__":
    main()