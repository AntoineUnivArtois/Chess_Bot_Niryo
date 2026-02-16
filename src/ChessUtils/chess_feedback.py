from .chess_teacher import MoveReasonCode, VALUES

class ChessFeedbackGenerator:
    def __init__(self, level = "beginner"):
        self.level = level

    def generate(self, reason, context = None):
        code = reason["code"]
        data = reason["data"]

        if code == MoveReasonCode.IGNORED_THREAT:
            return self.ignored_threat(data, context)

        if code == MoveReasonCode.HANGING_PIECE:
            return self.hanging_piece(data, context)

        if code == MoveReasonCode.MATERIAL_LOSS:
            return self.material_loss(data, context)
        
        if code == MoveReasonCode.MATERIAL_GAIN:
            return self.material_gain(data, context)

        if code == MoveReasonCode.PASSIVE_MOVE:
            return self.passive_move(context)

        if code == MoveReasonCode.BEST_MOVE:
            return "Excellent coup, c’était le meilleur choix ici."

        return "Coup correct."
    
    def ignored_threat(self, data, context):
        piece = data["piece"].upper()
        square = data["square"]
        attack_count = data["attack_count"]
        def_count = data["defense_count"]

        if data["immediate"] and data["value"] >= 5:
            return (
                f"Ton {piece} en {square} était attaqué directement {attack_count} fois et défendu {def_count} fois. "
                f"Cette menace devait être traitée en priorité."
            )
        
        if self.level == "beginner":
            return (
                f"Attention : ta pièce {piece} en {square} était menacée {attack_count} fois et défendu {def_count} fois, "
                f"et tu ne l’as pas protégée."
            )

        elif self.level == "intermediate":
            return (
                f"Tu as ignoré la menace sur ton {piece} en {square}. Il était attaqué {attack_count} fois et défendu {def_count} fois. "
                f"L’adversaire pouvait la capturer au prochain coup."
            )

        return (
            f"La pièce {piece} en {square} restait en prise après ton coup, "
            f"sans compensation tactique. Attaquée {attack_count} fois et défendue {def_count} fois."
        )
    
    def hanging_piece(self, data, context):
        piece = data["piece"].upper()
        square = data["square"]
        attackers = data.get("attackers", [])
        attack_count = data["attack_count"]
        def_count = data["defense_count"]

        attackers_desc = ", ".join(attackers)

        if self.level == "beginner":
            return (
                f"Attention : ton {piece} en {square} est en prise {attack_count} fois et défendu {def_count} fois. "
            )

        if self.level == "intermediate":
            return (
                f"Ton {piece} en {square} n’est pas protégé et peut être capturé "
                f"par {attackers_desc}. Il est attaqué {attack_count} fois et défendu {def_count} fois."
            )

        return (
            f"Le {piece} en {square} est laissé sans défense, "
            f"ce qui permet à l’adversaire de le capturer immédiatement. Il est attaqué {attack_count} fois et défendu {def_count} fois. ")
    
    def material_loss(self, data, context):
        delta = abs(data["delta"])

        if self.level == "beginner":
            return (
                f"Ce coup te fait perdre du matériel "
                f"(environ {delta} point(s))."
            )

        if self.level == "intermediate":
            return (
                f"Tu perds du matériel sans compensation "
                f"(≈ {delta} points)."
            )

        return (
            f"Cette séquence entraîne une perte matérielle nette "
            f"de {delta} points."
        )
    
    def material_gain(self, data, context):
        delta = data["delta"]

        if self.level == "beginner":
            return (
                f"Bien vu ! Tu gagnes du matériel "
                f"(≈ {delta} point(s))."
            )

        if self.level == "intermediate":
            return (
                f"Ce coup te permet de gagner du matériel "
                f"de manière favorable."
            )

        return (
            f"Le coup crée un gain matériel net "
            f"de {delta} points."
        )
    
    def passive_move(self, context):
        if self.level == "beginner":
            return (
                "Ce coup n’améliore pas vraiment ta position "
                "et laisse l’initiative à l’adversaire."
            )

        if self.level == "intermediate":
            return (
                "C’est un coup passif : il n’exerce pas de pression "
                "et ne répond pas aux enjeux de la position."
            )

        return (
            "Le coup manque d’ambition et ne correspond pas "
            "aux besoins dynamiques de la position."
        )