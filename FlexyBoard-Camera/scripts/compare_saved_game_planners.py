from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SOFT_DIR = ROOT / 'Software-GUI'
GAME_ROOT = ROOT / 'FlexyBoard-Camera' / 'debug_output' / 'Games'
sys.path.insert(0, str(SOFT_DIR))

from chess_state import ChessState


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare saved game planner outputs across planner versions.')
    parser.add_argument('--games', nargs='+', type=int, required=True, help='Game numbers to compare.')
    parser.add_argument('--old-ref', default='HEAD', help='Git ref for old planner source (default: HEAD).')
    parser.add_argument('--out', default=str(ROOT / 'planner_comparison_games.json'), help='Output JSON path.')
    return parser.parse_args()


def parse_endpoint(token: str, current_mod: Any):
    token = token.strip()
    if '%' in token:
        x_str, y_str = token.split(',')
        return current_mod._pct_ep(float(x_str.rstrip('%')), float(y_str.rstrip('%')))
    x_str, y_str = token.split(',')
    return current_mod._board_ep(int(x_str), int(y_str))


def endpoint_to_pct(endpoint: Any, current_mod: Any):
    if isinstance(endpoint, current_mod.PercentEndpoint):
        return endpoint
    return current_mod._board_coord_to_pct_from_mapping((endpoint.x, endpoint.y))


def sequence_distance(lines: list[str], current_mod: Any) -> float:
    total = 0.0
    for line in lines:
        src_token, dst_token = [part.strip() for part in line.split('->')]
        src = parse_endpoint(src_token, current_mod)
        dst = parse_endpoint(dst_token, current_mod)
        total += current_mod._pct_distance(endpoint_to_pct(src, current_mod), endpoint_to_pct(dst, current_mod))
    return total


def infer_piece_moves(lines: list[str]) -> int:
    if not lines:
        return 0
    chains = 1
    prev_dst = None
    for idx, line in enumerate(lines):
        src_token, dst_token = [part.strip() for part in line.split('->')]
        if idx == 0:
            prev_dst = dst_token
            continue
        if src_token != prev_dst:
            chains += 1
        prev_dst = dst_token
    return chains


def load_game_move_dirs(game_dir: Path):
    move_dirs = []
    for p in game_dir.iterdir():
        if not p.is_dir() or not p.name.startswith('Move'):
            continue
        m = re.fullmatch(r'Move(\d+)(?:_Retry(\d+))?', p.name)
        if not m:
            continue
        move_dirs.append((int(m.group(1)), int(m.group(2) or 0), p))
    move_dirs.sort()
    return move_dirs


def compare_game(game_num: int, old_mod: Any, new_mod: Any) -> dict[str, Any]:
    game_dir = GAME_ROOT / f'Game{game_num}'
    if not game_dir.exists():
        return {
            'game': game_num,
            'dir': str(game_dir),
            'missing': True,
            'summary': {},
            'moves': [],
        }
    state_old = ChessState()
    state_new = ChessState()
    result: dict[str, Any] = {
        'game': game_num,
        'dir': str(game_dir),
        'moves': [],
        'summary': {
            'processed_moves': 0,
            'stored_total_segments': 0,
            'old_total_segments': 0,
            'new_total_segments': 0,
            'stored_total_distance_pct': 0.0,
            'old_total_distance_pct': 0.0,
            'new_total_distance_pct': 0.0,
            'stored_total_piece_moves': 0,
            'old_total_piece_moves': 0,
            'new_total_piece_moves': 0,
            'old_total_planning_ms': 0.0,
            'new_total_planning_ms': 0.0,
            'sequence_differences_old_vs_new': 0,
        },
    }

    for num, retry, move_dir in load_game_move_dirs(game_dir):
        p1_file = move_dir / 'player1_resolved_move.json'
        p2_file = move_dir / 'player2_move.json'
        if not (p1_file.exists() and p2_file.exists()):
            continue
        p1 = json.loads(p1_file.read_text())
        p2 = json.loads(p2_file.read_text())
        uci = p1.get('metadata', {}).get('selected_uci')
        p2_from = p2.get('from')
        p2_to = p2.get('to')
        if not (uci and p2_from and p2_to):
            continue

        state_old.apply_move_trusted(uci[:2], uci[2:4])
        state_new.apply_move_trusted(uci[:2], uci[2:4])

        t0 = time.perf_counter()
        old_seq = old_mod.generate_chess_p2_sequence(state_old.copy(), p2_from, p2_to)
        old_ms = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        new_seq = new_mod.generate_chess_p2_sequence(state_new.copy(), p2_from, p2_to)
        new_ms = (time.perf_counter() - t1) * 1000.0

        state_old.try_apply_p2_move(p2_from, p2_to)
        state_new.try_apply_p2_move(p2_from, p2_to)

        stored_lines = p2.get('stm_sequence', [])
        old_lines = list(old_seq.lines)
        new_lines = list(new_seq.lines)

        stored_distance = sequence_distance(stored_lines, new_mod)
        old_distance = sequence_distance(old_lines, new_mod)
        new_distance = sequence_distance(new_lines, new_mod)

        stored_piece_moves = infer_piece_moves(stored_lines)
        old_piece_moves = infer_piece_moves(old_lines)
        new_piece_moves = infer_piece_moves(new_lines)

        move_entry = {
            'move_dir': move_dir.name,
            'turn_number': num,
            'retry': retry,
            'player1_move_uci': uci,
            'player2_move': {'from': p2_from, 'to': p2_to},
            'stored': {
                'lines': stored_lines,
                'segment_count': len(stored_lines),
                'distance_pct': round(stored_distance, 3),
                'distance_board_squares': round(stored_distance / max(new_mod._OFFBOARD_GEOMETRY.board_square_x_pct, new_mod._OFFBOARD_GEOMETRY.board_square_y_pct), 3),
                'piece_moves': stored_piece_moves,
            },
            'old_planner': {
                'lines': old_lines,
                'segment_count': len(old_lines),
                'distance_pct': round(old_distance, 3),
                'distance_board_squares': round(old_distance / max(new_mod._OFFBOARD_GEOMETRY.board_square_x_pct, new_mod._OFFBOARD_GEOMETRY.board_square_y_pct), 3),
                'piece_moves': old_piece_moves,
                'planning_ms': round(old_ms, 3),
                'capture_detected': old_seq.capture_detected,
                'temporary_relocations': old_seq.temporary_relocations,
                'debug_metrics': getattr(old_seq, 'debug_metrics', {}),
            },
            'new_planner': {
                'lines': new_lines,
                'segment_count': len(new_lines),
                'distance_pct': round(new_distance, 3),
                'distance_board_squares': round(new_distance / max(new_mod._OFFBOARD_GEOMETRY.board_square_x_pct, new_mod._OFFBOARD_GEOMETRY.board_square_y_pct), 3),
                'piece_moves': new_piece_moves,
                'planning_ms': round(new_ms, 3),
                'capture_detected': new_seq.capture_detected,
                'temporary_relocations': new_seq.temporary_relocations,
                'debug_metrics': getattr(new_seq, 'debug_metrics', {}),
            },
            'same_sequence_old_vs_new': old_lines == new_lines,
            'old_matches_stored': old_lines == stored_lines,
            'new_matches_stored': new_lines == stored_lines,
        }
        result['moves'].append(move_entry)

        summary = result['summary']
        summary['processed_moves'] += 1
        summary['stored_total_segments'] += len(stored_lines)
        summary['old_total_segments'] += len(old_lines)
        summary['new_total_segments'] += len(new_lines)
        summary['stored_total_distance_pct'] += stored_distance
        summary['old_total_distance_pct'] += old_distance
        summary['new_total_distance_pct'] += new_distance
        summary['stored_total_piece_moves'] += stored_piece_moves
        summary['old_total_piece_moves'] += old_piece_moves
        summary['new_total_piece_moves'] += new_piece_moves
        summary['old_total_planning_ms'] += old_ms
        summary['new_total_planning_ms'] += new_ms
        if old_lines != new_lines:
            summary['sequence_differences_old_vs_new'] += 1

    s = result['summary']
    for key in ('stored_total_distance_pct', 'old_total_distance_pct', 'new_total_distance_pct', 'old_total_planning_ms', 'new_total_planning_ms'):
        s[key] = round(s[key], 3)
    count = max(1, s['processed_moves'])
    s['old_avg_planning_ms'] = round(s['old_total_planning_ms'] / count, 3)
    s['new_avg_planning_ms'] = round(s['new_total_planning_ms'] / count, 3)
    return result


def main() -> int:
    args = parse_args()
    current_mod = load_module('planner_current_compare', SOFT_DIR / 'motor_sequence.py')
    old_src = subprocess.check_output(
        ['git', 'show', f'{args.old_ref}:Software-GUI/motor_sequence.py'],
        cwd=ROOT,
        text=True,
    )
    with tempfile.TemporaryDirectory() as td:
        old_path = Path(td) / 'motor_sequence_old.py'
        old_path.write_text(old_src)
        old_mod = load_module('planner_old_compare', old_path)
        report = {
            'old_ref': args.old_ref,
            'new_source': str(SOFT_DIR / 'motor_sequence.py'),
            'games': [compare_game(game_num, old_mod, current_mod) for game_num in args.games],
        }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(report, indent=2))
    print(out_path)
    for game in report['games']:
        print(json.dumps({'game': game['game'], 'summary': game['summary']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
