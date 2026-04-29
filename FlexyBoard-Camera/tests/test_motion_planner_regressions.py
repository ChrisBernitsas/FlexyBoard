from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Software-GUI'))

from chess_state import ChessState
from motor_sequence import MotionPlanner, _pct_ep, generate_chess_p2_sequence


def test_board_to_pct_uses_direct_route_when_clear() -> None:
    planner = MotionPlanner(occupied={(4, 3)})
    dst = _pct_ep(25.15, 93.95)
    planner.move_board_piece_to_pct((4, 3), dst)

    assert planner.lines == ['4,3 -> 25.15%,93.95%']
    assert any(note.startswith('board_to_pct_route=direct:4,3->25.15%,93.95%') for note in planner._capture_slots_used)
    assert planner._direct_route_hits >= 1


def test_game1_move8_no_duplicate_temp_relocation_pairs() -> None:
    game_dir = ROOT / 'FlexyBoard-Camera' / 'debug_output' / 'Games' / 'Game1'
    move_names = ['Move1', 'Move2_Retry2', 'Move3', 'Move4', 'Move5', 'Move6', 'Move7', 'Move8']

    state = ChessState()
    generated = None
    for move_name in move_names:
        move_dir = game_dir / move_name
        p1 = json.loads((move_dir / 'player1_resolved_move.json').read_text())
        p2 = json.loads((move_dir / 'player2_move.json').read_text())
        uci = p1['metadata']['selected_uci']
        state.apply_move_trusted(uci[:2], uci[2:4])
        generated = generate_chess_p2_sequence(state.copy(), p2['from'], p2['to'])
        state.try_apply_p2_move(p2['from'], p2['to'])

    assert generated is not None
    assert generated.lines == [
        '3,1 -> 4,2',
        '4,2 -> 4,3',
        '4,3 -> 5,3',
        '5,3 -> 34.29%,83.70%',
        '34.29%,83.70% -> 25.15%,93.95%',
        '4,3 -> 3,1',
    ]
    assert '5,2 -> 6,2' not in generated.lines
    assert '6,2 -> 5,2' not in generated.lines


def test_game1_move8_debug_metrics_report_single_relocation() -> None:
    game_dir = ROOT / 'FlexyBoard-Camera' / 'debug_output' / 'Games' / 'Game1'
    move_names = ['Move1', 'Move2_Retry2', 'Move3', 'Move4', 'Move5', 'Move6', 'Move7', 'Move8']

    state = ChessState()
    generated = None
    for move_name in move_names:
        move_dir = game_dir / move_name
        p1 = json.loads((move_dir / 'player1_resolved_move.json').read_text())
        p2 = json.loads((move_dir / 'player2_move.json').read_text())
        uci = p1['metadata']['selected_uci']
        state.apply_move_trusted(uci[:2], uci[2:4])
        generated = generate_chess_p2_sequence(state.copy(), p2['from'], p2['to'])
        state.try_apply_p2_move(p2['from'], p2['to'])

    assert generated is not None
    assert generated.debug_metrics['blockers_relocated'] == 0
    assert generated.debug_metrics['blockers_restored'] == 0
    assert generated.debug_metrics['estimated_piece_moves'] == 2
    assert generated.debug_metrics['fallback_used'] is False
    assert generated.debug_metrics['direct_route_rejected_reasons'] == [
        'direct_route_rejected=3,1->25.15%,93.95%:blocked_by:5,2'
    ]
