# ─────────────────────────────────────────────────────────────────────────────
# ChessUtils/image_to_board.py
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import cv2

# Valeur par défaut, côté « utile » du plateau (avant marge)
IM_EXTRACT_SMALL_SIDE_PIXELS = 1000

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION DES COULEURS (à adapter selon vos gommettes)
# ─────────────────────────────────────────────────────────────────────────────

COLOR_RANGES = {
    "red": [
        ((0, 100, 100), (5, 255, 255)),      
        ((170, 100, 100), (180, 255, 255))    
    ],
    "orange": [
        ((6, 100, 100), (20, 255, 255)),     
    ],
    "yellow": [
        ((25, 100, 100), (41, 255, 255)),   
    ],
    "green": [
        ((45, 10, 40), (99, 255, 255)),     
    ],
    "purple": [
        ((140, 30, 30), (170, 255, 255)),     
    ],
    "blue": [
        ((100, 50, 50), (130, 255, 255)),     
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS : Corrections d'image et masquage du bois
# ─────────────────────────────────────────────────────────────────────────────

def apply_gamma(img, gamma=1.0):
    """
    Correction gamma : brightening ou darkening.
    gamma > 1.0 : assombrit
    gamma < 1.0 : éclaircit
    """
    if gamma == 1.0:
        return img
    
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype(np.uint8)
    return cv2.LUT(img, table)


def gray_world_wb(img):
    """
    White balance simple (gray world hypothesis).
    Normalise les canaux R, G, B indépendamment.
    """
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.mean(result[:, :, 1])
    avg_b = np.mean(result[:, :, 2])
    result[:, :, 1] -= ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.3)
    result[:, :, 2] -= ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.3)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return np.clip(result, 0, 255).astype(np.uint8)


# ==== ANCIENNE VERSION (HSV) ====
# def mask_board_wood(img_hsv):
#     """
#     Masque les pixels du bois du plateau (trop sombres ou peu saturés).
#     Retourne un masque binaire (255 = à conserver, 0 = à ignorer).
#     """
#     h, s, v = cv2.split(img_hsv)
#     
#     # Rejette pixels bois
#     bad_mask = cv2.inRange(h, 12, 25) & cv2.inRange(s, 0, 200) & cv2.inRange(v, 0, 160)
#     good_mask = cv2.bitwise_not(bad_mask)
#     
#     return good_mask
# ==== NOUVELLE VERSION (LAB) ====
def mask_board_wood(img_lab):
    """
    (LAB) Masque les pixels du bois du plateau.
    Entrée : image en espace LAB (L,a,b) uint8.
    Retourne : masque binaire (255 = conserver, 0 = ignorer).
    Notes:
      - On rejette les tons bruns/oranges du bois via seuils sur a/b et L.
      - Les valeurs sont conservatrices ; ajustez si nécessaire.
    """
    L, a, b = cv2.split(img_lab)

    # heuristiques : le bois a souvent a > 135 (tirant vers le rouge) et b > 135 (tirant vers jaune)
    # on rejette plages typiques du bois (low saturation en a/b proches de ces intervalles)
    wood_mask = cv2.inRange(a, 120, 180) & cv2.inRange(b, 120, 200) & cv2.inRange(L, 20, 220)

    # bad_mask == bois => on veut l'inverser (0 = bois)
    good_mask = cv2.bitwise_not(wood_mask)
    return good_mask


def apply_morphology_closing(mask, kernel_size=5):
    """
    Morphological closing : ferme les petits trous.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


# ─────────────────────────────────────────────────────────────────────────────
# DÉTECTION DE GOMMETTES COLORÉES
# ─────────────────────────────────────────────────────────────────────────────

def detect_color_in_mask(hsv_roi, color_ranges_dict, min_area_percent=2):
    """
    Détecte la présence d'une couleur dans une ROI HSV.
    
    Args:
        hsv_roi : région d'intérêt en espace HSV
        color_ranges_dict : dictionnaire {color_name: [(low, high), ...]}
        min_area_percent : pourcentage minimum de pixels pour valider (2% = ~2-3 pixels)
    
    Returns:
        (detected_color, confidence, (cx, cy), area)
    """
    best_color = None
    best_ratio = 0.0
    best_center = None
    best_area = 0

    h, w = hsv_roi.shape[:2]
    total_pixels = h * w
    min_pixels = max(1, int(total_pixels * min_area_percent / 100.0))
    
    for color_name, ranges in color_ranges_dict.items():
        mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
        
        # Union de toutes les sous-plages pour cette couleur
        for low, high in ranges:
            sub_mask = cv2.inRange(hsv_roi, low, high)
            mask = cv2.bitwise_or(mask, sub_mask)
        
        # Closing pour éliminer bruit
        mask = apply_morphology_closing(mask, kernel_size=3)
        
        pixel_count = cv2.countNonZero(mask)

        if pixel_count >= min_pixels:
            ratio = pixel_count / total_pixels
            if ratio > best_ratio:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(cnt)

                    # Moments → centre
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx = cy = None

                    best_center = (cx, cy)
                    best_area = area

                best_ratio = ratio
                best_color = color_name

    return best_color, best_ratio, best_center, best_area



def detect_colored_stickers(
    board_warped,
    boxes,
    color_ranges=None,
    apply_gamma_correction=True,
    gamma=1.0,
    apply_white_balance=False,
    apply_wood_mask=True,
    min_area_percent=2
):
    """
    Détecte les gommettes colorées dans chaque case du plateau warpé.
    
    Args:
        board_warped : image warpée (H, W, 3) en BGR
        boxes : dictionnaire {cell_name: (x1, y1, x2, y2)}
        color_ranges : dictionnaire de plages HSV ou None (utilise COLOR_RANGES par défaut)
        apply_gamma_correction : bool, applique correction gamma avant détection
        gamma : float, facteur gamma (< 1.0 pour éclaircir)
        apply_white_balance : bool, applique white balance gray world
        apply_wood_mask : bool, masque les pixels du bois
        min_area_percent : pourcentage minimal de pixels pour valider une détection
    
    Returns:
        dict {cell_name: detected_color_or_None}
    """
    if color_ranges is None:
        color_ranges = COLOR_RANGES
    
    # 1) Corrections préliminaires
    img_work = board_warped.copy()
    
    if apply_gamma_correction:
        img_work = apply_gamma(img_work, gamma)
    
    if apply_white_balance:
        img_work = gray_world_wb(img_work)
    
    # Conversion en HSV
    img_hsv = cv2.cvtColor(img_work, cv2.COLOR_BGR2HSV)
    
    # 2) Masque du bois (optionnel)
    wood_mask = None
    if apply_wood_mask:
        wood_mask = mask_board_wood(img_hsv)
    
    # 3) Détection par case
    result = {}
    
    for cell_name, (x1, y1, x2, y2) in boxes.items():
        # Extraction ROI
        roi_hsv = img_hsv[y1:y2, x1:x2]
        
        # Application du masque bois si actif
        if wood_mask is not None:
            roi_wood_mask = wood_mask[y1:y2, x1:x2]
            # Masquer les pixels du bois
            roi_hsv = cv2.bitwise_and(roi_hsv, roi_hsv, mask=roi_wood_mask)
        
        # Détection de gomette
        detected_color, confidence, center, _ = detect_color_in_mask(
            roi_hsv,
            color_ranges,
            min_area_percent=min_area_percent
        )

        if center is not None:
            cx_global = center[0]
            cy_global = center[1]
            piece_side = detect_piece_color(roi_hsv, cx_global, cy_global)
        else:
            piece_side = None
        
        
        result[cell_name] = {
            "color": detected_color,
            "side": piece_side,
            "confidence": confidence,
            "center": center
        }
    
    return result


def board_state_from_colored_stickers(detection_dict, color_to_piece={"red": "K", "orange": "Q", "yellow": "F", "green": "T", "purple": "C", "blue": "P"}):
    """
    Convertit le dict de détection en matrice 8×8.
    
    Args:
        detection_dict : dict {cell_name: {color: ..., confidence: ...}}
        color_to_piece : dict optionnel {color_name: piece_char}
                        ex. {"red": "♔", "green": "♚"} ou {"red": "K", "green": "k"}
    
    Returns:
        matrix : liste de 8 listes (rangées 8 à 1)
    """
    if color_to_piece is None:
        color_to_piece = {}
    
    files = 'abcdefgh'
    ranks = '87654321'
    matrix = [[' ' for _ in range(8)] for _ in range(8)]
    
    for cell_name, info in detection_dict.items():
        color = info.get("color")
        side = info.get("side")
        if color is None:
            continue
        
        file_idx = files.index(cell_name[0])
        rank_idx = ranks.index(cell_name[1])
        
        # Conversion color → piece
        if side == "white":
            piece = color_to_piece.get(color, color[0].lower())
        else:
            piece = color_to_piece.get(color, color[0].upper())
        matrix[rank_idx][file_idx] = piece
    
    return matrix

def detect_piece_color(board_warped, x,y, offset_y=4, offset_x=4, threshold=120):
    """
    Détermine si la pièce est blanche ou noire.
    x, y : centre de la gommette
    offset_y : distance sous la gommette (en pixels)
    roi_size : demi-largeur de la zone analysée
    """
   # Extraire ROI sous la gommette
    y1 = max(0, y - offset_y)
    y2 = min(board_warped.shape[0], y + offset_y)
    x1 = max(0, x - offset_x)
    x2 = min(board_warped.shape[1], x + offset_x)
    
    roi = board_warped[y1:y2, x1:x2]

    # Luminosité moyenne
    _, _, v = cv2.split(roi)
    mean_v = np.mean(v)

    return "white" if mean_v > threshold else "black"


# ─────────────────────────────────────────────────────────────────────────────
# 1) Détection des markers & calcul de la matrice d'homographie
# ─────────────────────────────────────────────────────────────────────────────

def euclidean_dist_2_pts(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))


class PotentialMarker:
    def __init__(self, center, radius, cnt):
        self.center = center
        self.x = center[0]
        self.y = center[1]
        self.radius = radius
        self.contour = cnt
        self.is_merged = False

    def get_center(self):
        return self.center


class Marker:
    def __init__(self, potential_marker: PotentialMarker):
        self.list_centers = [potential_marker.get_center()]
        self.list_radius  = [potential_marker.radius]
        self.cx, self.cy  = potential_marker.get_center()
        self.radius       = potential_marker.radius
        self.identifiant  = None

    def add_circle(self, other: PotentialMarker):
        self.list_centers.append(other.get_center())
        self.list_radius.append(other.radius)
        other.is_merged = True
        mx, my = np.mean(self.list_centers, axis=0)
        self.cx, self.cy = int(round(mx)), int(round(my))
        self.radius      = int(round(max(self.list_radius)))

    def nb_circles(self):
        return len(self.list_centers)

    def get_id_from_slice(self, img_thresh):
        x, y, w, h = self.cx - 1, self.cy - 1, 3, 3
        val = np.mean(img_thresh[y:y+h, x:x+w])
        self.identifiant = "A" if val > 200 else "B"
        return self.identifiant

    def get_center(self):
        return (self.cx, self.cy)

    def get_radius(self):
        return self.radius


def find_markers_from_img_thresh(
    img_thresh,
    max_dist_between_centers=3,
    min_radius_circle=4,
    max_radius_circle=35,
    min_radius_marker=7
):
    contours = cv2.findContours(img_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    pots = []
    for cnt in contours:
        (x, y), r = cv2.minEnclosingCircle(cnt)
        if not (min_radius_circle < r < max_radius_circle):
            continue
        pots.append(PotentialMarker((int(round(x)), int(round(y))), int(round(r)), cnt))

    pots = sorted(pots, key=lambda m: m.x)
    markers = []
    for i, p in enumerate(pots):
        if p.is_merged:
            continue
        m = Marker(p)
        cx, cy = m.get_center()
        for q in pots[i+1:]:
            if q.is_merged:
                continue
            if q.x - cx > max_dist_between_centers:
                break
            if euclidean_dist_2_pts((cx, cy), q.get_center()) <= max_dist_between_centers:
                m.add_circle(q)
                cx, cy = m.get_center()
        if m.nb_circles() > 2 and m.get_radius() >= min_radius_marker:
            markers.append(m)
            m.get_id_from_slice(img_thresh)
    return markers


def sort_markers_detection(list_markers):
    ym = sorted(list_markers, key=lambda m: m.cy)
    top1, top2, bot1, bot2 = ym
    tl = top1 if top1.cx < top2.cx else top2
    tr = top2 if tl is top1 else top1
    bl = bot1 if bot1.cx < bot2.cx else bot2
    br = bot2 if bl is bot1 else bot1

    quad = [tl, tr, br, bl]
    ids  = [m.identifiant for m in quad]
    if ids.count("A") == 1:
        n = ids.index("A")
        return quad[n:] + quad[:n]
    if ids.count("B") == 1:
        n = ids.index("B")
        return quad[n:] + quad[:n]
    return quad


def complicated_sort_markers(list_markers, workspace_ratio):
    import itertools

    if workspace_ratio >= 1.0:
        tw = int(round(workspace_ratio * IM_EXTRACT_SMALL_SIDE_PIXELS))
        th = IM_EXTRACT_SMALL_SIDE_PIXELS
    else:
        thr = 1.0 / workspace_ratio
        th = int(round(thr * IM_EXTRACT_SMALL_SIDE_PIXELS))
        tw = IM_EXTRACT_SMALL_SIDE_PIXELS

    ids = [m.identifiant for m in list_markers]
    a_cnt = ids.count("A")
    b_cnt = ids.count("B")
    if a_cnt < 3 and b_cnt < 3:
        return None

    first, second = ("A", "B") if a_cnt >= b_cnt else ("B", "A")
    combs = []
    list1 = [m for m in list_markers if m.identifiant == first]
    list2 = [m for m in list_markers if m.identifiant == second]

    if list1:
        for m1 in list1:
            for trio in itertools.combinations(list2, 3):
                combs.append(sort_markers_detection([m1, *trio]))
    else:
        for quad in itertools.combinations(list2, 4):
            combs.append(sort_markers_detection(list(quad)))

    if not combs:
        return None

    final_pts = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype=np.float32)
    dets = []
    for quad in combs:
        src = np.array([[m.cx, m.cy] for m in quad], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, final_pts)
        dets.append(np.linalg.det(M))

    best = np.argmin(np.abs(np.array(dets) - 1))
    return combs[best]


def extract_img_markers_with_margin(
    img,
    workspace_ratio: float = 1.0,
    base_size: int = IM_EXTRACT_SMALL_SIDE_PIXELS,
    margin_cells: int = 1
):
    """
    Détecte 4 markers au centre des cases a1,a8,h1,h8 et renvoie :
      - warp (BGR) de taille (W_tot,H_tot) = (tw+2*margin_px, th+2*margin_px)
      - matrice M,
      - (W_tot, H_tot),
      - corners triés,
      - margin_px

    base_size    : taille en px du plateau sans marge (ex. 200).
    margin_cells : nb de cases de marge désiré autour (ex. 1 ou 2).
    """
    # 1) seuillage et détection des markers
    gray       = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        15, 25
    )
    marks = find_markers_from_img_thresh(img_thresh)
    if not marks or len(marks) > 6:
        return None, None, None, None, None

    # tri classique ou compliqué si >4 markers
    if len(marks) == 4:
        corners = sort_markers_detection(marks)
    else:
        corners = complicated_sort_markers(marks, workspace_ratio)
        if corners is None:
            return None, None, None, None, None

    # 2) calcul de la taille d'une case et de la marge en pixels
    cell_px   = base_size / 8.0
    margin_px = int(round(cell_px * margin_cells))

    # 3) dimensions du plateau utile avant marge
    if workspace_ratio >= 1.0:
        tw = th = base_size
    else:
        tw = int(round((1.0 / workspace_ratio) * base_size))
        th = tw

    # 4) dimensions totales avec marge
    W_tot = tw + 2 * margin_px
    H_tot = th + 2 * margin_px

    # 5) points source (centres détectés)
    src_pts = np.array([m.get_center() for m in corners], dtype=np.float32)

    # 6) points destination → centres des 4 coins du plateau utile
    half = cell_px / 2.0
    dst_pts = np.array([
        [margin_px + half,          margin_px + half],           # coin inférieur gauche
        [margin_px + tw - half - 1, margin_px + half],           # coin inférieur droit
        [margin_px + tw - half - 1, margin_px + th - half - 1],  # coin supérieur droit
        [margin_px + half,          margin_px + th - half - 1],  # coin supérieur gauche
    ], dtype=np.float32)

    # 7) homographie + warp
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warp = cv2.warpPerspective(img, M, (W_tot, H_tot))

    return warp, M, (W_tot, H_tot), corners, margin_px



# ─────────────────────────────────────────────────────────────────────────────
# 2) Génération des cases 8×8 au sein de l'image warpée (en ignorant la marge)
# ─────────────────────────────────────────────────────────────────────────────

def get_cell_boxes(board_img, margin_px, player_plays_white=True):
    """
    board_img  : image warpée de taille (W_tot, H_tot) = (base_size+2*margin, base_size+2*margin).
    margin_px  : nombre de pixels à ignorer tout autour.
    player_plays_white : booléen indiquant si le joueur a les pièces blanches.
                         Si False, on inverse la numérotation des cases (180°).

    → Retourne dict { 'a8': (x1,y1,x2,y2), …, 'h1': (…) }
      où la zone utile est le carré [margin_px : margin_px+base_size-1] en largeur ET hauteur.
      Si player_plays_white est False, les noms de cases sont renversés :
        - (i, j) → (7-i, 7-j) dans la grille 8×8.
    """
    H_tot, W_tot = board_img.shape[:2]
    # Base_size = W_tot - 2*margin_px
    base_size = W_tot - 2 * margin_px

    cell_w = base_size/ 8.0
    cell_h = base_size / 8.0
    files = 'abcdefgh'
    ranks = '87654321'
    boxes = {}

    for i in range(8):
        for j in range(8):
            x1 = int(round(margin_px + j * cell_w))
            y1 = int(round(margin_px + i * cell_h))
            x2 = int(round(margin_px + (j + 1) * cell_w))
            y2 = int(round(margin_px + (i + 1) * cell_h))

            if player_plays_white:
                # Nom "standard" si le joueur est blanc :
                cell_name = f"{files[j]}{ranks[i]}"
            else:
                # Inversion 180° : (i, j) → (7-i, 7-j)
                flipped_file = files[7 - j]
                flipped_rank = ranks[7 - i]
                cell_name = f"{flipped_file}{flipped_rank}"

            boxes[cell_name] = (x1, y1, x2, y2)

    return boxes


# ---------------------------------------------------------------------
# Grille virtuelle projetée 
# ---------------------------------------------------------------------
def create_virtual_detection_grid(
    real_corners_3d,
    image_corners_2d,
    camera_matrix,
    dist_coeffs=None,
    grid_size=8,
    plane_height_cm=20.0,
    margin_percent=0.0
):
    """
    Crée une grille virtuelle (grid_size x grid_size) parallèle au plateau réel
    mais située à plane_height_mm au-dessus du plan du plateau (déplacement le long
    de la normale du plan). La projection des points 3D vers l'image utilise
    cv2.projectPoints afin d'appliquer la distorsion optique (barillet/pincushion).

    Args:
        real_corners_3d : np.array shape (4,3) -> coins 3D du plateau (tl, tr, br, bl) en mm
        image_corners_2d: np.array shape (4,2) -> points image correspondants (tl, tr, br, bl)
        camera_matrix   : np.array 3x3 intrinsics
        dist_coeffs     : distorsion (None -> zeros)
        grid_size       : int, nombre de cases (8 pour 8x8)
        plane_height_cm : translation le long de la normale (positif = au-dessus)
        margin_percent  : extension relative du plateau (ex. 0.05 = +5%)

    Returns:
        virtual_boxes : dict { 'a8': (x1,y1,x2,y2), ... } (axis-aligned boxes in image pixels)
        proj_pts      : np.array shape (grid_size+1, grid_size+1, 2) des sommets projetés
    """
    # Validation / conversions
    objp = np.asarray(real_corners_3d, dtype=np.float64).reshape(-1, 3)
    imgp = np.asarray(image_corners_2d, dtype=np.float64).reshape(-1, 2)

    if dist_coeffs is None:
        dist = np.zeros((5, 1), dtype=np.float64)
    else:
        dist = np.asarray(dist_coeffs, dtype=np.float64)

    # solvePnP -> estimer rvec/tvec (pose du plan dans le repère objet)
    # Note : on suppose correspondance 1:1 entre objp[i] et imgp[i] (tl,tr,br,bl)
    ok, rvec, tvec = cv2.solvePnP(objp, imgp, camera_matrix, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed to estimate pose for the board.")

    # Récupération coins 3D et vecteurs planaires
    tl = objp[0].astype(np.float64)
    tr = objp[1].astype(np.float64)
    br = objp[2].astype(np.float64)
    bl = objp[3].astype(np.float64)

    # vecteurs u (droite) et v (bas) sur le plan du plateau (en mm)
    u = tr - tl
    v = bl - tl

    # appliquer margin en agrandissant u et v autour du centre
    center = tl + 0.5 * (u + v)
    u_exp = u * (1.0 + margin_percent)
    v_exp = v * (1.0 + margin_percent)
    tl_expanded = center - 0.5 * (u_exp + v_exp)

    # normale unitaire du plan (direction "haut" en 3D)
    normal = np.cross(u, v)
    norm = np.linalg.norm(normal)
    if norm == 0:
        raise RuntimeError("Invalid real_corners_3d: colinear points (normal is zero).")
    normal_unit = normal / norm

    # Convertir normale dans le repère caméra
    R, _ = cv2.Rodrigues(rvec)
    normal_cam = R @ normal_unit.reshape(3, 1)

    # normal_cam[2] = composante Z
    # Si Z > 0 → normal pointe derrière la caméra → l'inverser
    if normal_cam[2, 0] > 0:
        normal_unit = -normal_unit

    # Construire une grille 3D régulière (grid_size+1 x grid_size+1) parallèle au plan,
    # décalée le long de la normale de plane_height_mm.
    pts_3d = np.zeros((grid_size + 1, grid_size + 1, 3), dtype=np.float64)
    for i in range(grid_size + 1):
        fy = i / float(grid_size)
        for j in range(grid_size + 1):
            fx = j / float(grid_size)
            p = tl_expanded + fx * u_exp + fy * v_exp + normal_unit * plane_height_cm
            pts_3d[i, j] = p

    # Aplatir et projeter en image en utilisant projectPoints (qui applique la distorsion)
    pts_3d_flat = pts_3d.reshape(-1, 3)
    proj_pts, _ = cv2.projectPoints(pts_3d_flat, rvec, tvec, camera_matrix, dist)
    proj_pts = proj_pts.reshape((grid_size + 1, grid_size + 1, 2)).astype(np.float32)

    # Construire des bounding boxes axis-aligned pour chaque cellule virtuelle
    files = 'abcdefgh'
    ranks = '87654321'
    virtual_boxes = {}
    for i in range(grid_size):
        for j in range(grid_size):
            p_tl = proj_pts[i, j]
            p_tr = proj_pts[i, j + 1]
            p_br = proj_pts[i + 1, j + 1]
            p_bl = proj_pts[i + 1, j]
            xs = [p_tl[0], p_tr[0], p_br[0], p_bl[0]]
            ys = [p_tl[1], p_tr[1], p_br[1], p_bl[1]]
            x1 = int(np.floor(min(xs)))
            y1 = int(np.floor(min(ys)))
            x2 = int(np.ceil(max(xs)))
            y2 = int(np.ceil(max(ys)))

            cell_name = f"{files[j]}{ranks[i]}"
            virtual_boxes[cell_name] = (x1, y1, x2, y2)

    return virtual_boxes, proj_pts


def visualize_real_and_virtual_grids(
    img,
    real_boxes,
    virtual_boxes,
    virtual_grid_pts=None,
    label_font_scale=0.4,
    show=True
):
    """
    Visualisation superposée :
      - grille réelle (real_boxes) en bleu
      - grille virtuelle (virtual_boxes) en rouge
      - points projetés (virtual_grid_pts) optionnellement

    Args:
        img             : image BGR (warp) sur laquelle dessiner
        real_boxes      : dict {cell: (x1,y1,x2,y2)} (issue de get_cell_boxes)
        virtual_boxes   : dict {cell: (x1,y1,x2,y2)} (issue de create_virtual_detection_grid)
        virtual_grid_pts: np.array (G+1,G+1,2) des sommets projetés (optionnel)
        label_font_scale: échelle texte
        show            : si True, affiche la fenêtre (util pour tests visuels)

    Retourne :
        image annotée (BGR)
    """
    vis = img.copy()
    h, w = vis.shape[:2]
    color_real = (255, 0, 0)   # bleu (BGR)
    color_virtual = (0, 0, 255)  # rouge
    files = 'abcdefgh'

    # Dessiner la grille réelle (rectangles) en bleu
    for cell, (x1, y1, x2, y2) in real_boxes.items():
        # clip aux dimensions de l'image pour éviter erreurs d'affichage
        x1c = int(np.clip(x1, 0, w - 1)); y1c = int(np.clip(y1, 0, h - 1))
        x2c = int(np.clip(x2, 0, w - 1)); y2c = int(np.clip(y2, 0, h - 1))
        cv2.rectangle(vis, (x1c, y1c), (x2c, y2c), color_real, 1)
        cv2.putText(vis, cell, (x1c + 3, y1c + 12), cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, color_real, 1)

    # Dessiner la grille virtuelle (rectangles) en rouge
    for cell, (x1, y1, x2, y2) in virtual_boxes.items():
        x1c = int(np.clip(x1, 0, w - 1)); y1c = int(np.clip(y1, 0, h - 1))
        x2c = int(np.clip(x2, 0, w - 1)); y2c = int(np.clip(y2, 0, h - 1))
        cv2.rectangle(vis, (x1c, y1c), (x2c, y2c), color_virtual, 1)
        cv2.putText(vis, cell, (x1c + 3, y2c - 4), cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, color_virtual, 1)

    # Optionnel : dessiner tous les points projetés (maillage) en rouge
    if virtual_grid_pts is not None:
        G = virtual_grid_pts.shape[0] - 1
        for i in range(G + 1):
            for j in range(G + 1):
                pt = virtual_grid_pts[i, j].astype(int)
                px = int(np.clip(pt[0], 0, w - 1)); py = int(np.clip(pt[1], 0, h - 1))
                cv2.circle(vis, (px, py), 2, color_virtual, -1)
        # Relier les lignes du maillage (optionnel pour meilleure lisibilité)
        for i in range(G + 1):
            row_pts = [tuple(virtual_grid_pts[i, j].astype(int)) for j in range(G + 1)]
            row_pts_clipped = [(int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))) for (x, y) in row_pts]
            for k in range(len(row_pts_clipped) - 1):
                cv2.line(vis, row_pts_clipped[k], row_pts_clipped[k + 1], color_virtual, 1)
        for j in range(G + 1):
            col_pts = [tuple(virtual_grid_pts[i, j].astype(int)) for i in range(G + 1)]
            col_pts_clipped = [(int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))) for (x, y) in col_pts]
            for k in range(len(col_pts_clipped) - 1):
                cv2.line(vis, col_pts_clipped[k], col_pts_clipped[k + 1], color_virtual, 1)

    if show:
        cv2.imshow('Real (blue) vs Virtual (red) grids - Press any key', vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return vis

# ─────────────────────────────────────────────────────────────────────────────
# AMÉLIORATIONS : Détection adaptative de gommettes (robustesse optique)
# ─────────────────────────────────────────────────────────────────────────────

def apply_clahe_lab(img, clip_limit=2.0, tile_size=8):
    """
    Améliore le contraste via CLAHE (Contrast Limited Adaptive Histogram Equalization)
    sur la couche L de LAB. Utile pour les éclairages non-uniformes.
    
    Args:
        img : image BGR
        clip_limit : limite de contraste (2.0 par défaut)
        tile_size : taille des tuiles (8x8 par défaut)
    
    Returns:
        image BGR améliorée
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_clahe = clahe.apply(l_channel)
    lab[:, :, 0] = l_clahe
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def detect_color_adaptive(hsv_roi, color_ranges_dict, min_area_percent=2, debug=False):
    """
    Version améliorée de detect_color_in_mask avec morphologie et contours.
    
    Args:
        hsv_roi : région d'intérêt en espace HSV
        color_ranges_dict : dictionnaire {color_name: [(low, high), ...]}
        min_area_percent : pourcentage minimum de pixels pour valider
        debug : affiche les masques intermédiaires
    
    Returns:
        (detected_color, confidence, center, area, mask)
    """
    best_color = None
    best_ratio = 0.0
    best_center = None
    best_area = 0
    best_mask = None

    h, w = hsv_roi.shape[:2]
    total_pixels = h * w
    min_pixels = max(1, int(total_pixels * min_area_percent / 100.0))
    
    for color_name, ranges in color_ranges_dict.items():
        mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
        
        # Fusion de toutes les sous-plages pour cette couleur
        for low, high in ranges:
            sub_mask = cv2.inRange(hsv_roi, low, high)
            mask = cv2.bitwise_or(mask, sub_mask)
        
        # Morphologie : ouverture (enlever bruit) puis fermeture (solidifier)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        
        if debug:
            cv2.imshow(f'Mask {color_name}', mask)
        
        # Détection de contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Chercher le plus grand contour
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            
            if area > min_pixels:
                # Centre de masse du contour
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Ratio : aire du contour / aire du ROI
                    ratio = area / total_pixels
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_color = color_name
                        best_center = (cx, cy)
                        best_area = area
                        best_mask = mask.copy()
    
    if debug and best_mask is not None:
        cv2.imshow(f'Best mask: {best_color}', best_mask)
        cv2.waitKey(1)
    
    return best_color, best_ratio, best_center, best_area, best_mask


def smooth_hsv_frame_buffer(frame_buffer, max_buffer_size=5):
    """
    Lissage temporel : moyenne des N derniers frames pour stabiliser la détection.
    Utile pour réduire les vacillements dus à la lumière.
    
    Args:
        frame_buffer : list de frames HSV précédents
        max_buffer_size : nombre de frames à moyenner (5 par défaut)
    
    Returns:
        frame moyenné en HSV
    """
    if len(frame_buffer) == 0:
        return None
    
    frame_buffer = frame_buffer[-max_buffer_size:]
    stacked = np.stack(frame_buffer, axis=0)
    return np.mean(stacked, axis=0).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Détection de gommettes AMÉLIORÉE avec pipeline robuste
# ─────────────────────────────────────────────────────────────────────────────

def detect_colored_stickers_robust(
    board_warped,
    boxes,
    color_ranges=None,
    apply_gamma_correction=True,
    gamma=1.5,
    apply_white_balance=False,
    apply_wood_mask=True,
    apply_clahe=True,
    apply_blur=True,
    min_area_percent=2,
    debug=False,
    adaptive_samples=None
):
    """
    Pipeline robuste de détection de gommettes avec corrections optiques multiples.
    
    Args:
        board_warped : image warpée (H, W, 3) en BGR
        boxes : dictionnaire {cell_name: (x1, y1, x2, y2)}
        color_ranges : dictionnaire de plages HSV ou None
        apply_gamma_correction : bool, applique correction gamma
        gamma : float, facteur gamma (< 1.0 pour éclaircir)
        apply_white_balance : bool, applique white balance gray world
        apply_wood_mask : bool, masque les pixels du bois
        apply_clahe : bool, améliore contraste via CLAHE (LAB)
        apply_blur : bool, applique flou gaussien avant HSV
        min_area_percent : pourcentage minimal de pixels pour valider
        debug : affiche les masques intermédiaires
        adaptive_samples : échantillons pour adaptation des plages HSV (dict {color_name: (x1,y1,x2,y2)})
    
    Returns:
        dict {cell_name: {color, confidence, center, area}}
    """
    if color_ranges is None:
        color_ranges = COLOR_RANGES
    
    # ─────── PIPELINE DE PRÉTRAITEMENT ───────
    img_work = board_warped.copy()
    
    # 1) Correction gamma
    if apply_gamma_correction:
        img_work = apply_gamma(img_work, gamma)
    
    # 2) White balance (gray world)
    if apply_white_balance:
        img_work = gray_world_wb(img_work)
    
    # 3) CLAHE (amélioration contraste adaptative, LAB)
    if apply_clahe:
        img_work = apply_clahe_lab(img_work, clip_limit=2.0, tile_size=8)
    
    # 4) Flou gaussien (réduction bruit)
    if apply_blur:
        img_work = cv2.GaussianBlur(img_work, (5, 5), 1.0)
    
    # Conversion en HSV
    img_hsv = cv2.cvtColor(img_work, cv2.COLOR_BGR2HSV)
    
    # Adaptation des plages HSV à partir d'échantillons (si fournis)
    if adaptive_samples is not None:
        color_ranges = adapt_color_ranges_from_samples(img_hsv, adaptive_samples, color_ranges)
    
    # 5) Masque du bois (optionnel)
    wood_mask = None
    if apply_wood_mask:
        wood_mask = mask_board_wood(img_hsv)
    
    # ─────── DÉTECTION PAR CASE ───────
    result = {}
    
    for cell_name, (x1, y1, x2, y2) in boxes.items():
        # Extraction ROI
        roi_hsv = img_hsv[y1:y2, x1:x2]
        
        # Application du masque bois si actif
        if wood_mask is not None:
            roi_wood_mask = wood_mask[y1:y2, x1:x2]
            roi_hsv = cv2.bitwise_and(roi_hsv, roi_hsv, mask=roi_wood_mask)
        
        # Détection adaptative avec morphologie
        detected_color, confidence, center, area, mask = detect_color_adaptive(
            roi_hsv,
            color_ranges,
            min_area_percent=min_area_percent,
            debug=debug
        )
        
        if center is not None:
            cx_global = center[0]
            cy_global = center[1]
            piece_side = detect_piece_color(roi_hsv, cx_global, cy_global)
        else:
            piece_side = None
        
        result[cell_name] = {
            "color": detected_color,
            "side": piece_side,
            "confidence": confidence,
            "center": center,
            "area": area
        }
    
    if debug:
        cv2.destroyAllWindows()
    
    return result


def adapt_color_ranges_from_samples(img_hsv, sample_rois, base_ranges, h_tol=8, s_tol=40, v_tol=40):
    """
    Crée des plages HSV dynamiques à partir d'échantillons.
    Les tolérances s'appliquent AUTOUR des ranges de base COLOR_RANGES.
    Cela permet d'ajuster légèrement pour la luminosité/caméra, sans dévier radicalement.
    
    Args:
        img_hsv : image complète en HSV (np.uint8)
        sample_rois : dict { color_name: (x1,y1,x2,y2) } — petites ROIs contenant la gommette de référence
        base_ranges : COLOR_RANGES d'origine (structure de référence)
        h_tol : tolérance ajoutée/retirée autour des limites H de base
        s_tol : tolérance ajoutée/retirée autour des limites S de base
        v_tol : tolérance ajoutée/retirée autour des limites V de base
    
    Returns:
        new_ranges : dict same format as COLOR_RANGES
    
    Notes:
        - Mesure la médiane HSV dans chaque échantillon.
        - Décale les ranges de base vers la médiane observée (dans les limites de tolérance).
        - Gère le wrap-around pour le rouge.
        - Évite les chevauchements en réduisant les tolérances si deux couleurs sont trop proches.
    """
    new_ranges = {}
    medians = {}
    
    # ─────── ÉTAPE 1 : Mesurer les médianes dans les échantillons ───────
    for color in base_ranges.keys():
        if sample_rois and color in sample_rois:
            x1, y1, x2, y2 = sample_rois[color]
            roi = img_hsv[y1:y2, x1:x2]
            
            if roi.size == 0:
                medians[color] = None
                continue
            
            h = roi[:, :, 0].flatten().astype(int)
            s = roi[:, :, 1].flatten().astype(int)
            v = roi[:, :, 2].flatten().astype(int)
            
            # Filtrer pixels peu saturés (bruit/fond)
            mask_sat = s > max(0, int(np.median(s) * 0.3))
            
            if np.count_nonzero(mask_sat) < 5:
                medians[color] = None
                continue
            
            h_med = int(np.median(h[mask_sat]))
            s_med = int(np.median(s[mask_sat]))
            v_med = int(np.median(v[mask_sat]))
            
            medians[color] = (h_med, s_med, v_med)
        else:
            medians[color] = None
    
    # ─────── ÉTAPE 2 : Adapter les ranges autour de la base ───────
    for color, base_ranges_list in base_ranges.items():
        
        if medians[color] is None:
            # Pas d'échantillon → garder la base
            new_ranges[color] = base_ranges_list
            continue
        
        h_med, s_med, v_med = medians[color]
        
        # Calculer le centre et les limites de la plage de base
        if isinstance(base_ranges_list[0][0], tuple):
            # Liste de tuples : [(low, high), ...]
            base_low, base_high = base_ranges_list[0]
            h_base_low, s_base_low, v_base_low = base_low
            h_base_high, s_base_high, v_base_high = base_high
        else:
            # Format simple
            h_base_low, s_base_low, v_base_low = base_ranges_list[0]
            h_base_high, s_base_high, v_base_high = base_ranges_list[1]
        
        # Déplacer les limites vers la médiane observée (dans les limites de tolérance)
        h_new_low = np.clip(h_med - h_tol, h_base_low - h_tol, h_base_low + h_tol)
        h_new_high = np.clip(h_med + h_tol, h_base_high - h_tol, h_base_high + h_tol)
        
        s_new_low = np.clip(s_med - s_tol, s_base_low - s_tol, s_base_low + s_tol)
        s_new_high = np.clip(s_med + s_tol, s_base_high - s_tol, s_base_high + s_tol)
        
        v_new_low = np.clip(v_med - v_tol, v_base_low - v_tol, v_base_low + v_tol)
        v_new_high = np.clip(v_med + v_tol, v_base_high - v_tol, v_base_high + v_tol)
        
        # Gestion du wrap-around pour le rouge (H ≈ 0 ou H ≈ 180)
        if color == 'red' and len(base_ranges_list) > 1:
            # Deux sous-ranges pour le rouge
            h_new_low = int(np.clip(h_new_low, 0, 180))
            h_new_high = int(np.clip(h_new_high, 0, 180))
            s_new_low = int(np.clip(s_new_low, 0, 255))
            s_new_high = int(np.clip(s_new_high, 0, 255))
            v_new_low = int(np.clip(v_new_low, 0, 255))
            v_new_high = int(np.clip(v_new_high, 0, 255))
            
            new_ranges[color] = [
                ((0, s_new_low, v_new_low), (h_new_high, s_new_high, v_new_high)),
                ((170, s_new_low, v_new_low), (180, s_new_high, v_new_high))
            ]
        else:
            # Une seule plage
            h_new_low = int(np.clip(h_new_low, 0, 180))
            h_new_high = int(np.clip(h_new_high, 0, 180))
            s_new_low = int(np.clip(s_new_low, 0, 255))
            s_new_high = int(np.clip(s_new_high, 0, 255))
            v_new_low = int(np.clip(v_new_low, 0, 255))
            v_new_high = int(np.clip(v_new_high, 0, 255))
            
            new_ranges[color] = [((h_new_low, s_new_low, v_new_low), (h_new_high, s_new_high, v_new_high))]
    
    # ─────── ÉTAPE 3 : Détecter et résoudre les chevauchements ───────
    colors_with_samples = [c for c, med in medians.items() if med is not None]
    
    if len(colors_with_samples) > 1:
        overlaps = {}
        for i, color1 in enumerate(colors_with_samples):
            for color2 in colors_with_samples[i+1:]:
                h1_med, _, _ = medians[color1]
                h2_med, _, _ = medians[color2]
                
                # Distance Hue (circulaire, 0-180)
                h_dist = min(abs(h1_med - h2_med), 180 - abs(h1_med - h2_med))
                
                if h_dist < 15:  # Seuil de chevauchement potentiel
                    overlaps[(color1, color2)] = h_dist
        
        # Réduire les tolérances pour les paires qui chevauchent
        if overlaps:
            for (color1, color2), h_dist in overlaps.items():
                reduction_factor = max(0.5, 1.0 - (15.0 - h_dist) / 30.0)
                
                for color in [color1, color2]:
                    h_med, s_med, v_med = medians[color]
                    if color != 'red':
                        h_base_low, s_base_low, v_base_low = base_ranges[color][0][0]
                        h_base_high, s_base_high, v_base_high = base_ranges[color][0][1] if len(base_ranges[color]) > 1 else base_ranges[color][0][0]
                    
                    h_tol_adj = int(h_tol * reduction_factor)
                    s_tol_adj = int(s_tol * reduction_factor)
                    v_tol_adj = int(v_tol * reduction_factor)
                    
                    h_new_low = int(np.clip(h_med - h_tol_adj, h_base_low - h_tol, h_base_low + h_tol))
                    h_new_high = int(np.clip(h_med + h_tol_adj, h_base_high - h_tol, h_base_high + h_tol))
                    
                    s_new_low = int(np.clip(s_med - s_tol_adj, s_base_low - s_tol, s_base_low + s_tol))
                    s_new_high = int(np.clip(s_med + s_tol_adj, s_base_high - s_tol, s_base_high + s_tol))
                    
                    v_new_low = int(np.clip(v_med - v_tol_adj, v_base_low - v_tol, v_base_low + v_tol))
                    v_new_high = int(np.clip(v_med + v_tol_adj, v_base_high - v_tol, v_base_high + v_tol))
                    
                    if color == 'red' and len(base_ranges[color]) > 1:
                        new_ranges[color] = [
                            ((0, s_new_low, v_new_low), (h_new_high, s_new_high, v_new_high)),
                            ((170, s_new_low, v_new_low), (180, s_new_high, v_new_high))
                        ]
                    else:
                        new_ranges[color] = [((h_new_low, s_new_low, v_new_low), (h_new_high, s_new_high, v_new_high))]
    
    # ─────── ÉTAPE 4 : Affichage des ranges pour debug ───────
    print("\n[ADAPTED COLOR RANGES - Variation autour de la base]")
    for color in base_ranges.keys():
        if medians[color] is not None:
            h_med, s_med, v_med = medians[color]
            base_str = str(base_ranges[color][0])
            adapted_str = str(new_ranges[color][0])
            print(f"  {color:8s} (H={h_med:3d}, S={s_med:3d}, V={v_med:3d})")
            print(f"    Base:    {base_str}")
            print(f"    Adapted: {adapted_str}")
        else:
            print(f"  {color:8s} (pas d'échantillon - base conservée)")
    
    return new_ranges

def detect_colored_stickers(
    board_warped,
    boxes,
    color_ranges=None,
    apply_gamma_correction=True,
    gamma=1.0,
    apply_white_balance=False,
    apply_wood_mask=False,
    min_area_percent=2,
    use_robust=True
):
    """
    Wrapper compatible : utilise la version robuste par défaut.
    Falls back to simple version si use_robust=False.
    """
    if use_robust:
        return detect_colored_stickers_robust(
            board_warped,
            boxes,
            color_ranges=color_ranges,
            apply_gamma_correction=apply_gamma_correction,
            gamma=gamma,
            apply_white_balance=apply_white_balance,
            apply_wood_mask=apply_wood_mask,
            apply_clahe=True,
            apply_blur=True,
            min_area_percent=min_area_percent,
            debug=False
        )
    else:
        # Fallback à la version basique existante
        if color_ranges is None:
            color_ranges = COLOR_RANGES
        
        img_work = board_warped.copy()
        
        if apply_gamma_correction:
            img_work = apply_gamma(img_work, gamma)
        
        if apply_white_balance:
            img_work = gray_world_wb(img_work)
        
        img_hsv = cv2.cvtColor(img_work, cv2.COLOR_BGR2HSV)
        
        wood_mask = None
        if apply_wood_mask:
            wood_mask = mask_board_wood(img_hsv)
        
        result = {}
        
        for cell_name, (x1, y1, x2, y2) in boxes.items():
            roi_hsv = img_hsv[y1:y2, x1:x2]
            
            if wood_mask is not None:
                roi_wood_mask = wood_mask[y1:y2, x1:x2]
                roi_hsv = cv2.bitwise_and(roi_hsv, roi_hsv, mask=roi_wood_mask)
            
            detected_color, confidence, center, _ = detect_color_in_mask(
                roi_hsv,
                color_ranges,
                min_area_percent=min_area_percent
            )
            
            if center is not None:
                cx_global = center[0]
                cy_global = center[1]
                piece_side = detect_piece_color(roi_hsv, cx_global, cy_global)
            else:
                piece_side = None
            
            result[cell_name] = {
                "color": detected_color,
                "side": piece_side,
                "confidence": confidence,
                "center": center
            }
        
        return result

