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
import copy
from math import pi
import numpy as np
import cv2

# Niryo modules import for Ned2 movements
from pyniryo import NiryoRobot, PoseObject, RobotAxis, uncompress_image, show_img

# IA imports
import torch

# Utils imports (workspace gathering / image to board transcription funcs etc.)
sys.path.append("ChessUtils")
from ChessUtils.chess_board import board as ChessBoard
from ChessUtils import (
    get_workspace_poses_ssh,
    extract_img_markers_with_margin,
    get_cell_boxes,
    board_state_from_yolo,
    ChessNet,
    UCT_search,
    do_decode_n_move_pieces

)

# ─────────────────────────────────────────────────────────────────────────────
# II. ROBOT CONFIG
# ─────────────────────────────────────────────────────────────────────────────

ROBOT_IP = '169.254.200.200' # Direct Ethernet
MODEL_PATH = "best.pt" # A enlever

NUM_READS = 200
CELL_SIZE = 0.04  # 4 cm
ELECTROMAGNET_PIN = 'DO4'
ELECTROMAGNET_ELEVATION_COEFF = 0.14
SHIFT_DIST  = 0.005

# Individual pieces heights (Change this to your pieces heights if you cloned..
# ..this repo and changed STLs files) in meters
PIECE_HEIGHTS = {
    'P': 0.042, 'p': 0.042,
    'N': 0.053, 'n': 0.053,
    'B': 0.056, 'b': 0.056,
    'R': 0.047, 'r': 0.047,
    'Q': 0.063, 'q': 0.063,
    'K': 0.075, 'k': 0.075,
}

# Niryo API func that allows this was not implemented so I made a..
# ..new one (connects to robot and gets calibration poses file)
CAL1, CAL2, CAL3, CAL4 = get_workspace_poses_ssh()

# ─────────────────────────────────────────────────────────────────────────────
# III. HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def square_to_index(sq: str):
    """Ex. 'e2' → (6,4) (line i, column j in current_board)."""
    file, rank = sq[0], int(sq[1])
    j = ord(file) - ord('a')
    i = 8 - rank
    return i, j

def game_result(b: ChessBoard):
    """'mat', 'pat' or None if game continues."""
    if b.actions():
        return None
    return 'mat' if b.check_status() else 'pat'

# ─────────────────────────────────────────────────────────────────────────────
# X, Y, Z TILES & BUFFERS' POSES CALCULATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def set_board_tiles_positions(board_tiles, player_plays_white, cell_size=CELL_SIZE):
    """
    Compute and assign a PoseObject to each chessboard square, based on the robot workspace and player orientation.

    Inputs:
        board_tiles (dict[str, PoseObject]) – mapping from square names (e.g. "e2") to PoseObject placeholders.
        player_plays_white (bool) – True if the human plays White (standard orientation), False for Black (rotated board).
        cell_size (float) – size of one square in meters, distance between adjacent square centers.

    Outputs:
        dict[str, PoseObject] – the same board_tiles dict, with each entry set to a PoseObject at the correct (x, y, z, roll, pitch, yaw).
    """

    position_A1 = CAL4
    position_A1.x += cell_size * 0.1
    position_A1.z += (ELECTROMAGNET_ELEVATION_COEFF*0.6) * cell_size

    letters = list('abcdefgh')
    numbers = list('12345678')
    for li, l in enumerate(letters):
        for ni, n in enumerate(numbers):
            mapped_l = l if player_plays_white else letters[::-1][li]
            mapped_n = n if player_plays_white else numbers[::-1][ni]
            dx = - ni * cell_size
            dy =   li * cell_size
            board_tiles[f"{mapped_l}{mapped_n}"] = PoseObject(
                position_A1.x + dx,
                position_A1.y + dy,
                position_A1.z,
                0.00, pi/2, pi/4,
            )
    return board_tiles

def set_buffer_tiles_positions(buffer_tiles, cell_size=CELL_SIZE):
    """
    Compute and assign a PoseObject to each buffer slot around the board, based on a reference pose and square size.

    Inputs:
        buffer_tiles (dict[str, PoseObject]) – mapping from buffer slot names (e.g. "1 1", "1 2", ..., "4 4") to PoseObject placeholders.
        cell_size (float) – size of one chessboard square in meters, used to space buffer slots.

    Outputs:
        dict[str, PoseObject] – the same buffer_tiles dict, with each entry set to a PoseObject at the correct (x, y, z, roll, pitch, yaw) for that slot.
    """

    pos = CAL2
    pos.x -= 2 * cell_size
    pos.z += ELECTROMAGNET_ELEVATION_COEFF * cell_size
    for ci, c in enumerate('1234'):
        for ri, r in enumerate('1234'):
            dx = - ri * cell_size
            dy =   ci * cell_size
            buffer_tiles[f"{c}{r}"] = PoseObject(
                pos.x + dx, pos.y + dy, pos.z,
                0.00, pi/2, pi/4,
            )
    return buffer_tiles

# ─────────────────────────────────────────────────────────────────────────────
# ROBOT MOVES FUNCTIONS (pickup, place, capture are encapsulated in play_move)
# ─────────────────────────────────────────────────────────────────────────────

def pickup_with_electromagnet(robot, pick_pose, piece_type):
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
    p = copy.copy(pick_pose)
    p.z += h + 3*CELL_SIZE
    robot.move_pose(p)
    robot.activate_electromagnet(ELECTROMAGNET_PIN)
    robot.shift_pose(RobotAxis.Z, -3*CELL_SIZE)
    robot.shift_pose(RobotAxis.Z,  3*CELL_SIZE)

def place_with_electromagnet(robot, target_pose, piece_type):
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
    p = copy.copy(target_pose)
    p.z += h + 3*CELL_SIZE
    robot.move_pose(p)
    robot.shift_pose(RobotAxis.Z, -3*CELL_SIZE)
    robot.deactivate_electromagnet(ELECTROMAGNET_PIN)
    robot.shift_pose(RobotAxis.Z,  3*CELL_SIZE)

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

# ─────────────────────────────────────────────────────────────────────────────
# BOARD STATE CAPTURE AND PLAYER SIDE DETECTION (can move to Utils in future)
# ─────────────────────────────────────────────────────────────────────────────

def capture_and_get_state(robot, cell_boxes_base, M, W_tot, H_tot, yolo):
    """
    Capture and warp the robot’s camera image, run YOLO detection, and convert to a board-state matrix.

    Inputs:
        robot (NiryoRobot) – robot instance to grab and undistort the camera image.
        cell_boxes_base (dict[str, tuple]) – mapping of square names to pixel boxes.
        M (ndarray) – homography matrix for perspective transform.
        W_tot, H_tot (int, int) – width and height for the warped image.
        yolo (YOLO) – trained YOLO model instance.

    Outputs:
        board_warp_img (ndarray) – warped top-down image of the chessboard.
        matrix (list[list[str]]) – 8×8 list of detected piece codes per square.
        res (Ultralytics result) – raw YOLO inference result for further use.
    """
    img_und = uncompress_image(robot.get_img_compressed())
    board_warp_img = cv2.warpPerspective(img_und, M, (W_tot, H_tot))
    res = yolo(board_warp_img)[0]
    matrix = board_state_from_yolo(res, cell_boxes_base)
    return board_warp_img, matrix, res

def capture_and_average_state(robot, capture_fn, shift_dist=SHIFT_DIST):
    """
    Compute a stabilized board-state by capturing two images with lateral shifts and merging the results.

    Inputs:
        robot (NiryoRobot) – robot instance capable of shifting pose and capturing images.
        capture_fn (callable) – function that returns (image, matrix, yolo_result) on each call.
        shift_dist (float) – distance in meters to shift left and right for dual capture.

    Outputs:
        avg (list[list[str]]) – 8x8 matrix where each cell is the agreed-upon piece code, preferring matching detections.
    """

    # Décalage à gauche
    robot.shift_pose(RobotAxis.Y, -shift_dist)
    robot.wait(0.5)
    _, mat1, _ = capture_fn()

    # Décalage à droite
    robot.shift_pose(RobotAxis.Y, 2 * shift_dist)
    robot.wait(0.5)
    _, mat2, _ = capture_fn()

    # Retour au centre
    robot.shift_pose(RobotAxis.Y, -shift_dist)
    robot.wait(0.5)

    # Consensus case par case
    mat1 = np.array(mat1)
    mat2 = np.array(mat2)
    avg = [[None] * 8 for _ in range(8)]
    for i in range(8):
        for j in range(8):
            if mat1[i, j] == mat2[i, j]:
                avg[i][j] = mat1[i, j]
            else:
                avg[i][j] = mat1[i, j]
    return avg

def wait_player_move(robot: NiryoRobot, matrix_state_current, matrix_state_post, capture_fn):
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

        # Attendre avant nouvelle capture
        robot.wait(1)
        # Capture stabilisée par double décalage
        matrix_state_post = capture_and_average_state(robot, capture_fn)

    print("Nouvelle position jouée")
    return matrix_state_post

def player_has_white(corners, yolo_result):
    """
    Determine whether the human player has the white pieces by comparing distances from the top-left marker to detected piece centers.

    Inputs:
        corners (list[Marker]) – sorted list of 4 detected markers, with corners[0] being the top-left.
        yolo_result (Ultralytics result) – YOLO inference output containing `names`, `boxes.xyxy`, and `boxes.cls`.

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

# ─────────────────────────────────────────────────────────────────────────────
# IV. MAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Robot initialisation
    robot = NiryoRobot(ROBOT_IP)
    robot.clear_collision_detected()
    robot.calibrate_auto()
    robot.setup_electromagnet(ELECTROMAGNET_PIN)

    # # Camera and YOLO model setup  
    # robot.set_brightness(1.0); robot.set_contrast(1.5); robot.set_saturation(1.8) # A en
    # yolo = YOLO(MODEL_PATH)
    #### REMPLACER PAR MA DETECTION

    # AI model loading
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ChessNet().to(device).eval()
    ckpt = torch.load("ChessUtils/model_data/current_net_trained_iter2.pth.tar", map_location=device)
    net.load_state_dict(ckpt['state_dict'])
    print("Modèle IA chargé")
   
    # Robot's wait pose object initialisation (SetUp in Niryo Studio)
    wait_pose = PoseObject(0.1706, 0.0061, 0.3864, 3.066, 1.340, 3.082)
    robot.move(wait_pose)

    # Calibration instructions
    print("Remove towers to make the robot see the Markers")
    init_ok = False
    while not init_ok:
        img_und = uncompress_image(robot.get_img_compressed())
        try:
            board_empty, M, (W_tot, H_tot), corners, margin_px = \
                extract_img_markers_with_margin(img_und, margin_cells=1.5)
            init_ok = True
        except TypeError:
            print("Impossible de détecter les marqueurs.")
            robot.wait(1) # slowing down capture process
    print("Markers détectés OK")

    # DEBUG1
    show_img('empty_wrap', board_empty, 5000)

    # Player side detection
    ''' TODO
        wait till hands detected
        if no hands during 5 secs
        then it means board is set up
        if hand detected, time_elapsed = 0 and redo process while no hands detected
    '''
    input("→ Placez les pièces et appuyez sur Entrée…")
    yolo_res = yolo(board_empty)[0]
    player_plays_white = player_has_white(corners, yolo_res)
    print(f"Joueur = {'Blancs' if player_plays_white else 'Noirs'}")

    # Board and buffer tiles poses calculation based on NiryoMarkerTL pose
    board_tiles = {f"{f}{n}": None for f in 'abcdefgh' for n in '12345678'}
    tiles_positions = set_board_tiles_positions(board_tiles, player_plays_white)
    buffer_tiles = {f"{c}{r}": None for c in '1234' for r in '1234'}
    buffer_positions = set_buffer_tiles_positions(buffer_tiles)
    # Empty Buffer on start (you can place a spare queen in it)
    buffer_state = {k: None for k in buffer_positions}

    # Calcul des coordonées des cases dans l'image pour l'intersection pièce/échiquier
    # Image tiles' coordinates computation for piece/board intersections
    cell_boxes_base = get_cell_boxes(board_empty, margin_px, player_plays_white)
    print("→ Grille repérée")

    # DEBUG2
    # boucle sur toutes les colonnes et rangées
    # piece_type = 'p'
    # for file in 'abcdefgh':
    #     for rank in '12345678':
    #         sq = f"{file}{rank}"
    #         pose = board_tiles[sq]
    #         try:
    #             print(f"→ Test sur la case {sq}")
    #             # on approche, on attrape, puis on repose
    #             pickup_with_electromagnet(robot, pose, piece_type)
    #             robot.wait(1)                      # petit délai pour stabilité
    #             place_with_electromagnet(robot, pose, piece_type)
    #             robot.wait(1)
    #         except Exception as e:
    #             print(f" Erreur sur {sq} : {e}")

    # First board state detection
    capture_fn = lambda: capture_and_get_state(
        robot, cell_boxes_base, M, W_tot, H_tot, yolo
    )
    board_img, matrix_current, _ = capture_fn()
    matrix_current = np.array(matrix_current)
    show_img("Initialisation", board_img, 5000)

    # DEBUG3 : Show computed grid on board image
    img_grid = board_img.copy()
    for (x1, y1, x2, y2) in cell_boxes_base.values():
        cv2.rectangle(img_grid, (x1, y1), (x2, y2), (255, 0, 0), 2)
    show_img("Grille 8x8", img_grid, wait_ms=1000)

    # Softaware's board initialisation
    b = ChessBoard()
    b.current_board = matrix_current
    b.player = 0  # Always white first

    # Game loop
    while game_result(b) is None:
        is_human_turn = (b.player == 0 and player_plays_white) or \
                        (b.player == 1 and not player_plays_white)

        if is_human_turn:
            # Wait until player plays legal move
            _, matrix_post, _ = capture_fn()
            matrix_new = wait_player_move(
                robot, matrix_current, matrix_post, capture_fn
            )
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
            best_idx, _ = UCT_search(b, num_reads=NUM_READS, net=net)
            new_b = do_decode_n_move_pieces(b, best_idx)

            # Move details gathering
            from_sq, to_sq, move_type = b.get_move_details(
                np.array(old_matrix), np.array(new_b.current_board)
            )

            # Move execution with Ned2
            play_move(
                robot, from_sq, to_sq, move_type,
                tiles_positions, buffer_positions, buffer_state
            )
            robot.move(wait_pose)

            # Board update
            b = new_b
            # Refreshing state for next player move
            _, matrix_current, _ = capture_fn()

    # Game ends
    print(f"### Résultat : {game_result(b)} ###")
    robot.move(wait_pose)
    robot.close_connection()

if __name__ == "__main__":
    main()
