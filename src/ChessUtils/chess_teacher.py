import chess
import chess.engine
from enum import Enum

VALUES = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9
        }
VISION_VALUES = {'p': 1,
                 'n': 3,
                 'b': 3,
                 'r': 5,
                 'q': 9
                 }


class MoveReasonCode(Enum):
    BEST_MOVE = "BEST_MOVE"
    GOOD_MOVE = "GOOD_MOVE"
    HANGING_PIECE = "HANGING_PIECE"
    MATERIAL_LOSS = "MATERIAL_LOSS"
    MATERIAL_GAIN = "MATERIAL_GAIN"
    MISSED_TACTIC = "MISSED_TACTIC"
    IGNORED_THREAT = "IGNORED_THREAT"
    PASSIVE_MOVE = "PASSIVE_MOVE"
    POSITION_WEAKENED = "POSITION_WEAKENED"


class ChessTeacher:
    def __init__(self, engine : chess.engine.SimpleEngine, depth = 15):
        self.engine = engine
        self.depth = depth

    def analyse_move(self, board_before : chess.Board, move : chess.Move, player_color):
        eval_before, pv_before = self.analyse_position(board_before, player_color)

        board_after = board_before.copy()
        board_after.push(move)

        eval_after, _ =self.analyse_position(board_after,player_color)
        delta = eval_after - eval_before

        best_move = pv_before[0] if pv_before else None

        classification = self.classify_move(delta)

        detections = self.detect_reasons(board_before, board_after, move, best_move, delta, player_color)

        reason = self.determine_main_reason(detections)

        return board_after, {
            "delta": delta,
            "classification": classification,
            "reason": reason
        }
    
    def detect_reasons(self, board_before, board_after, move, best_move, delta, player_color):
        return [
            self.detect_material_change(board_before, board_after, player_color),
            self.detect_hanging_piece(board_after, player_color),
            self.detect_ignored_threat(board_before, board_after, move, player_color),
            self.detect_passive_move(delta, best_move, move),
            self.detect_good_move(best_move, move)
        ]

# ─────────────────────────────────────────────────────────────────────────────

    def analyse_position(self, board : chess.Board, player_color):
        info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
        if player_color:
            score = info["score"].pov(chess.WHITE).score(mate_score=10000)
        else:
            score = info["score"].pov(chess.BLACK).score(mate_score=10000)
        pv = info.get("pv", [])
        return score / 100 if score else 0, pv
    

# ─────────────────────────────────────────────────────────────────────────────

    def classify_move(self, delta):
        if delta >= 0.5:
            return "Excellent"
        elif delta >= 0.1:
            return "Bon"
        elif delta >= -0.1:
            return "Correct"
        elif delta >= -0.5:
            return "Imprécis"
        elif delta >= -2.0:
            return "Erreur"
        else:
            return "Grosse erreur"
        
# ─────────────────────────────────────────────────────────────────────────────
# FACTUAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
    # Perte matériel
    def material_balance(self,board):
        score = 0
        for piece, value in VALUES.items():
            score += len(board.pieces(piece, chess.WHITE)) * value
            score -= len(board.pieces(piece, chess.BLACK)) * value
        return score
    
    def analyze_square(self, board, square, color):
        attackers = board.attackers(not color, square)
        defenders = board.attackers(color, square)

        return {
            "attackers": list(attackers),
            "defenders": list(defenders),
            "attack_count": len(attackers),
            "defense_count": len(defenders)
        }
    
    def detect_material_change(self,board_before, board_after, player_color):
        before = self.material_balance(board_before)
        after = self.material_balance(board_after)
        delta = after - before if player_color else before - after

        if delta < 0:
            return {
                "code": MoveReasonCode.MATERIAL_LOSS,
                "data": {"delta": delta}
            }
        elif delta > 0:
            return {
                "code": MoveReasonCode.MATERIAL_GAIN,
                "data": {"delta": delta}
            }
        return None
    
    #Pièce en prise
    def detect_hanging_piece(self, board_after, player_color):
        for square in chess.SquareSet(board_after.occupied_co[player_color]):
            attackers = board_after.attackers(not player_color, square)
            defenders = board_after.attackers(player_color, square)

            if attackers and not defenders:
                piece = board_after.piece_at(square)
                analysis = self.analyze_square(board_after, square, player_color)
                return {
                    "code": MoveReasonCode.HANGING_PIECE,
                    "data": {
                        "piece": piece.symbol(),
                        "square": chess.square_name(square),
                        "attack_count": analysis["attack_count"],
                        "defense_count": analysis["defense_count"],
                        "attackers": self.describe_attackers(board_after, analysis["attackers"])
                    }
                }
        return None
    
    def detect_threatened_pieces(self, board, player_color):
        threatened = []

        for square in chess.SquareSet(board.occupied_co[player_color]):
            analysis = self.analyze_square(board, square, player_color)

            if analysis["attack_count"] > analysis["defense_count"]:
                piece = board.piece_at(square)
                threatened.append({
                    "piece": piece.symbol(),
                    "square": square,
                    "attack_count": analysis["attack_count"],
                    "defense_count": analysis["defense_count"],
                    "attackers": analysis["attackers"]
                })

        return threatened
    
    def is_capture_immediate(self, board, square, player_color):
        for attacker in board.attackers(not player_color, square):
            if board.is_legal(chess.Move(attacker, square)):
                return True
        return False

# ─────────────────────────────────────────────────────────────────────────────
# PEDAGOGICAL REASONS
# ─────────────────────────────────────────────────────────────────────────────
    def detect_passive_move(self, delta, best_move, played_move):
        if played_move != best_move and delta < -0.3:
            return {
                "code": MoveReasonCode.PASSIVE_MOVE,
                "data": {}
            }
        return None
    
    # Meilleur coup 
    def detect_good_move(self, best_move, played_move):
        if played_move == best_move:
            return {
                "code": MoveReasonCode.BEST_MOVE,
                "data": {}
            }
        return None
    
    # Priorité
    def determine_main_reason(self,detections):
        priority = [
            MoveReasonCode.MATERIAL_LOSS,
            MoveReasonCode.HANGING_PIECE,
            MoveReasonCode.IGNORED_THREAT,
            MoveReasonCode.MATERIAL_GAIN,
            MoveReasonCode.PASSIVE_MOVE,
            MoveReasonCode.BEST_MOVE
        ]

        for p in priority:
            for d in detections:
                if d and d["code"] == p:
                    return d

        return {
            "code": MoveReasonCode.GOOD_MOVE,
            "data": {}
        }
    
    def detect_ignored_threat(self, board_before, board_after, move, player_color):
        threatened_before = self.detect_threatened_pieces(board_before, player_color)

        if not threatened_before:
            return None

        for threat in threatened_before:
            square = threat["square"]

            # La pièce a bougé → menace traitée
            if move.from_square == square:
                continue

            # La pièce n'existe plus (capture volontaire ?)
            if board_after.piece_at(square) is None:
                continue

            # Recalcul après le coup
            analysis_after = self.analyze_square(board_after, square, player_color)

            # Menace toujours présente
            if analysis_after["attack_count"] > analysis_after["defense_count"]:
                return {
                    "code": MoveReasonCode.IGNORED_THREAT,
                    "data": {
                        "piece": threat["piece"],
                        "square": chess.square_name(square),
                        "attack_count": analysis_after["attack_count"],
                        "defense_count": analysis_after["defense_count"],
                        "attackers": self.describe_attackers(
                            board_after,
                            analysis_after["attackers"]
                        ),
                        "value":VISION_VALUES[threat["piece"].lower()],
                        "immediate": self.is_capture_immediate(board_after, square, player_color)
                    }
                }

        return None

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
    def describe_attackers(self, board, attackers):
        desc = []
        for sq in attackers:
            piece = board.piece_at(sq)
            desc.append(f"{piece.symbol().upper()}:{chess.square_name(sq)}")
        return desc
    
    def hanging_pieces_detailed(self, board, color):
        results = []
        for square in chess.SquareSet(board.occupied_co[color]):
            analysis = self.analyze_square(board, square, color)
            if analysis["attack_count"] > 0 and analysis["defense_count"] == 0:
                piece = board.piece_at(square)
                results.append({
                    "piece": piece.symbol(),
                    "square": chess.square_name(square),
                    "attackers": self.describe_attackers(board, analysis["attackers"])
                })
        return results
