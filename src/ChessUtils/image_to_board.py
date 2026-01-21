import numpy as np
import cv2

IM_EXTRACT_SMALL_SIDE_PIXELS = 1000

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION DES COULEURS (à adapter selon vos gommettes)
# ─────────────────────────────────────────────────────────────────────────────

COLOR_RANGES = {
    "red": [
        ((0, 50, 50), (5, 255, 255)),      
        ((170, 50, 50), (180, 255, 255))    
    ],
    "orange": [
        ((6, 50, 50), (20, 255, 255)),     
    ],
    "yellow": [
        ((25, 50, 50), (40, 255, 255)),   
    ],
    "green": [
        ((42, 10, 10), (99, 255, 255)),     
    ],
    "purple": [
        ((121, 50, 50), (170, 255, 255)),     
    ],
    "blue": [
        ((100, 10, 10), (120, 255, 255)),     
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# DÉTECTION DE GOMMETTES COLORÉES
# ─────────────────────────────────────────────────────────────────────────────

def detect_color_in_mask(hsv_roi, color_ranges_dict, min_area_percent):
    """
    Détecte la présence d'une couleur dans une ROI HSV.
    
    Args:
        hsv_roi : région d'intérêt en espace HSV
        color_ranges_dict : dictionnaire {color_name: [(low, high), ...]}
        min_area_percent : pourcentage minimum de pixels pour valider
    
    Returns:
        (detected_color, confidence, (cx, cy), area)
    """
    best_color = None
    best_ratio = 0.0
    best_center = None
    best_area = 0  
    for color_name, ranges in color_ranges_dict.items():
        mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
        
        # Union de toutes les sous-plages pour cette couleur
        for low, high in ranges:
            sub_mask = cv2.inRange(hsv_roi, low, high)
            mask = cv2.bitwise_or(mask, sub_mask)
        
        # # Tests de visualisation
        # cv2.imshow("mask_"+color_name, mask)
        # cv2.waitKey(0)
        # cv2.destroyWindow("mask_"+color_name)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area <= 140:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        bbox_area = w * h
        if bbox_area == 0:
            continue

        local_ratio = area / bbox_area

        if local_ratio >= min_area_percent / 100.0 and local_ratio > best_ratio:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = cy = None

            best_color = color_name
            best_ratio = local_ratio
            best_center = (cx, cy)
            best_area = area


    return best_color, best_ratio, best_center, best_area


def detect_colored_stickers(
    board_warped,
    boxes,
    min_area_percent,
    color_ranges=None, 
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
    
    # 3) Détection par case
    result = {}
    
    for cell_name, (x1, y1, x2, y2) in boxes.items():
        # Extraction ROI
        roi = img_work[y1:y2, x1:x2]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
        
        # Détection de gomette
        detected_color, confidence, center, _ = detect_color_in_mask(
            hsv_roi,
            color_ranges,
            min_area_percent
        )

        if center is not None:
            piece_side = detect_piece_color(roi)
            # print(f"[DEBUG] Case {cell_name}: couleur détectée = {detected_color}, confiance = {confidence:.2%}, côté = {piece_side}")  
        else:
            piece_side = None
        
        result[cell_name] = {
            "color": detected_color,
            "side": piece_side,
            "confidence": confidence,
            "center": center
        }
    
    return result


def board_state_from_colored_stickers(color, side, color_to_piece={"red": "K", "orange": "Q", "yellow": "B", "green": "R", "purple": "N", "blue": "P"}):
    """
    Convertit le dict de détection en matrice 8×8.
    
    Args:
        detection_dict : dict {cell_name: {color: ..., confidence: ...}}
        color_to_piece : dict optionnel {color_name: piece_char}
    
    Returns:
        matrix : liste de 8 listes (rangées 8 à 1)
    """
    if color_to_piece is None:
        color_to_piece = {}
    
    piece_letter = color_to_piece.get(color, color[0].upper())
  
    # Conversion color → piece
    if side == "white":
        piece = piece_letter.upper()
    else:
        piece = piece_letter.lower()

    return piece


def detect_piece_color(roi, zoom_factor=2.5, debug=False):
    """
    Détection basée sur la présence du blanc (prioritaire et strict)
    Si blanc absent → noir par défaut
    """
    
    if debug:
        cv2.imshow("0. ROI original", roi)
    
    # === ZOOM ===
    h, w = roi.shape[:2]
    new_size = (int(w * zoom_factor), int(h * zoom_factor))
    zoomed = cv2.resize(roi, new_size, interpolation=cv2.INTER_CUBIC)
    
    if debug:
        cv2.imshow("1. Zoom", zoomed)
    
    # === RÉDUCTION DES REFLETS SPÉCULAIRES ===
    # Filtre bilateral : préserve les bords, réduit les reflets
    deglared = cv2.bilateralFilter(zoomed, 11, 100, 100)
    
    if debug:
        cv2.imshow("2. Reflets attenues", deglared)
    
    # === TRAITEMENT LAB ===
    lab = cv2.cvtColor(deglared, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    
    if debug:
        cv2.imshow("4a. Canal L brut", L)
        cv2.imshow("4b. Canal a brut", a)
        cv2.imshow("4c. Canal b brut", b)
    
    # === CONTRASTE LOCAL (CLAHE) pour faire ressortir le point même avec ombre ===
    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(6, 6))
    L_clahe = clahe.apply(L)
    
    if debug:
        cv2.imshow("5. Canal L CLAHE", L_clahe)
    
    # === NORMALISATION + CONTRASTE ===
    L_norm = cv2.normalize(L_clahe, None, 0, 255, cv2.NORM_MINMAX)
    L_contrast = np.power(L_norm / 255.0, 0.6) * 255
    L_contrast = L_contrast.astype(np.uint8)
    
    if debug:
        cv2.imshow("6. Canal L contraste", L_contrast)
    
    # Masque des zones "neutres" en couleur (blanc potentiel)
    a_neutral = np.abs(a.astype(np.int16) - 128) < 5
    b_neutral = np.abs(b.astype(np.int16) - 128) < 5
    neutral_mask = a_neutral & b_neutral
    
    if debug:
        neutral_debug = np.zeros_like(L_contrast)
        neutral_debug[neutral_mask] = 255
        cv2.imshow("7. Masque zones neutres (blanc)", neutral_debug)
    
    # Accentuer uniquement les zones blanches neutres
    L_enhanced = L_contrast.copy()
    L_enhanced[neutral_mask] = np.clip(L_enhanced[neutral_mask] * 1.3, 0, 255).astype(np.uint8)
    
    if debug:
        cv2.imshow("8. Canal L avec blanc accentue", L_enhanced)
    
    # === RECONSTRUCTION ===
    enhanced_lab = cv2.merge([L_enhanced, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    if debug:
        cv2.imshow("9. Image finale", enhanced)
    
    border_margin = int(10 * zoom_factor)
    
    # === DÉTECTION STRICTE DU BLANC ===
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    h_ch, s, v = cv2.split(hsv)
    
    # CRITÈRES STRICTS pour le blanc
    # 1. Saturation TRÈS basse
    _, very_low_saturation = cv2.threshold(s, 60, 255, cv2.THRESH_BINARY_INV)
    
    # 2. Valeur TRÈS haute
    _, very_high_value = cv2.threshold(v, 230, 255, cv2.THRESH_BINARY)
    
    # 3. Combiner les deux critères
    white_mask = cv2.bitwise_and(very_low_saturation, very_high_value)
    
    # Nettoyage agressif pour éliminer les reflets isolés
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
    
    # Fermeture pour reconnecter les points fragmentés
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_medium)
    
    if debug:
        cv2.imshow("White mask (strict)", white_mask)
    
    # Trouver les contours
    white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrer les contours valides
    valid_white_contours = []
    taille_contour = []
    min_area = 8 * (zoom_factor ** 2)  # Seuil plus élevé
    max_area = 40 * (zoom_factor ** 2)  # Éviter les grandes zones
    
    for contour in white_contours:
       
        area = cv2.contourArea(contour)
        
        # Taille dans la bonne plage
        if min_area < area < max_area:
            # CRITÈRE SUPPLÉMENTAIRE : vérifier le ratio largeur/hauteur
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            aspect_ratio = max(w_rect, h_rect) / max(min(w_rect, h_rect), 1)
            
            # Un point dessiné devrait être relativement circulaire
            if aspect_ratio < 1.5:
                valid_white_contours.append(contour)
                taille_contour.append([w_rect, h_rect])
                
                if debug:
                    print(f"  Contour blanc valide : aire={area:.0f}, ratio={aspect_ratio:.2f}")
    
    if valid_white_contours:
        # Calculer la densité totale de blanc
        density = []
        for i in range(len(valid_white_contours)):
            total_white_pixels = cv2.contourArea(valid_white_contours[i])
            total_pixels = taille_contour[i][0] * taille_contour[i][1]
            density.append(total_white_pixels / total_pixels)
        white_density = max(density)
    else:
        white_density = 0.0
    
    if debug:
        print(f"\nDensité de blanc : {white_density*100:.2f}%")
        print(f"Contours blancs valides : {len(valid_white_contours)}")
        
        # Visualisation
        debug_img = enhanced.copy()
        cv2.drawContours(debug_img, valid_white_contours, -1, (0, 255, 0), 2)
        cv2.imshow("Detection finale", debug_img)
        cv2.waitKey(0)
        cv2.destroyWindow("Image traitee")
        cv2.destroyWindow("White mask (strict)")
        cv2.destroyWindow("Detection finale")
    
    # DÉCISION STRICTE
    # Si on a des contours blancs valides ET densité suffisante
    if len(valid_white_contours) > 0 and white_density > 0.55: 
        return "white"
    else:
        return "black"  # Par défaut
    
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
# 3) Grille virtuelle projetée 
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

