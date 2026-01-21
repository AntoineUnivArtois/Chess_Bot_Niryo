#!/usr/bin/env python3
import torch
from alpha_net import ChessNet
from chess_board import board as Board
from MCTS_chess import UCT_search, do_decode_n_move_pieces
import encoder_decoder as ed


def tuple_to_uci(from_sq, to_sq, prom=None):
    from_str = chr(from_sq[1] + ord('a')) + str(8 - from_sq[0])
    to_str = chr(to_sq[1] + ord('a')) + str(8 - to_sq[0])
    if prom:
        return from_str + to_str + prom
    return from_str + to_str

def human_move(board):
    # Liste des coups légaux au format UCI
    legal_moves = board.actions()
    legal_ucis = [tuple_to_uci(from_sq, to_sq, prom) for (from_sq, to_sq, prom) in legal_moves]

    print("Coups légaux :", legal_ucis)
    uci = input("Ton coup : ").strip()

    if uci not in legal_ucis:
        print("Coup illégal, essaie un autre.")
        return human_move(board)

    # Trouver le tuple correspondant
    idx = legal_ucis.index(uci)
    from_sq, to_sq, prom = legal_moves[idx]

    move_idx = ed.encode_action(board, from_sq, to_sq, prom)
    return do_decode_n_move_pieces(board, move_idx)


def main():
    # 1) Choix de la couleur
    human_player_color = None
    while human_player_color not in ('0', '1'):
        human_player_color = input("Choisis ta couleur : 0 pour Blancs, 1 pour Noirs : ").strip()
    human_player_color = int(human_player_color)

    # 2) Charger le modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ChessNet().to(device).eval()
    ckpt = torch.load("model_data/current_net_trained_iter2.pth.tar", map_location=device)
    net.load_state_dict(ckpt['state_dict'])
    print("Modèle chargé, début de la partie.")

    # 3) Initialiser l’échiquier
    board = Board()

    # 4) Boucle de jeu
    while True:
        print("\nÉchiquier (0=Blancs, 1=Noirs):\n")
        print(board.current_board)

        if board.player == human_player_color:
            # Tour humain
            board = human_move(board)
        else:
            # Tour IA
            best_idx, _ = UCT_search(board, num_reads=200, net=net)
            board = do_decode_n_move_pieces(board, best_idx)
            print(f"IA joue l’action #{best_idx}")

        # Vérifier fin de partie
        if board.check_status() and not board.in_check_possible_moves():
            gagnant = "Noirs" if board.player == 0 else "Blancs"
            print(f"\nÉchec et mat – Gagnant : {gagnant}")
            break
        if board.move_count > 100:
            print("\n100 coups joués → match nul")
            break

if __name__ == "__main__":
    main()
