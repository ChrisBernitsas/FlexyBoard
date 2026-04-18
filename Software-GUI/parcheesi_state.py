"""Parcheesi board state for the Player 2 GUI.

The model tracks the real Parcheesi concepts the GUI and motor planner need:
- four players, four tokens each
- nests, 68-space main track, seven-space home paths, and home areas
- dice entry on 5, safe squares, blockades, captures, and exact home entry

Location IDs are stable strings used by the UI and motor planner:
- nest_2_1       player 2 token 1 in its nest
- main_17        main track space 17
- home_2_3       player 2 home-path index 3
- homearea_2_1   player 2 token 1 finished
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import random
import sys
from typing import Dict, Iterable, List, Optional, Tuple, cast


class Piece(Enum):
    EMPTY = auto()
    P1_TOKEN_1 = auto()
    P1_TOKEN_2 = auto()
    P1_TOKEN_3 = auto()
    P1_TOKEN_4 = auto()
    P2_TOKEN_1 = auto()
    P2_TOKEN_2 = auto()
    P2_TOKEN_3 = auto()
    P2_TOKEN_4 = auto()
    P3_TOKEN_1 = auto()
    P3_TOKEN_2 = auto()
    P3_TOKEN_3 = auto()
    P3_TOKEN_4 = auto()
    P4_TOKEN_1 = auto()
    P4_TOKEN_2 = auto()
    P4_TOKEN_3 = auto()
    P4_TOKEN_4 = auto()

    def to_player_num(self) -> int:
        if self is Piece.EMPTY:
            return 0
        return int(self.name[1])

    def to_token_num(self) -> int:
        if self is Piece.EMPTY:
            return 0
        return int(self.name.rsplit("_", 1)[1])

    def short_label(self) -> str:
        return f"{self.to_player_num()}.{self.to_token_num()}"


@dataclass(frozen=True)
class ParsedLocation:
    kind: str
    player: int
    pos: int


class ParcheesiState:
    NUM_PLAYERS = 4
    TOKENS_PER_PLAYER = 4
    MAIN_TRACK_LENGTH = 68
    HOME_PATH_LENGTH = 7

    PLAYER_START_SQUARES = {1: 0, 2: 17, 3: 34, 4: 51}
    PLAYER_HOME_ENTRY_SQUARES = {1: 67, 2: 16, 3: 33, 4: 50}
    SAFE_SQUARES = {6, 13, 23, 30, 40, 47, 57, 64}

    # Virtual 19x19 board used by the UI and percent motor projection.
    GRID_MAX = 18
    TRACK_GRID_POSITIONS: tuple[tuple[float, float], ...] = (
        # Clockwise outline of the 19x19 plus-shaped Parcheesi road.
        # The outline has 68 cells, matching the logical main track.
        (8, 0), (9, 0), (10, 0),
        (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (11, 7),
        (12, 8), (13, 8), (14, 8), (15, 8), (16, 8), (17, 8), (18, 8),
        (18, 9), (18, 10),
        (17, 10), (16, 10), (15, 10), (14, 10), (13, 10), (12, 10), (11, 10), (11, 11),
        (10, 12), (10, 13), (10, 14), (10, 15), (10, 16), (10, 17), (10, 18),
        (9, 18), (8, 18),
        (8, 17), (8, 16), (8, 15), (8, 14), (8, 13), (8, 12), (8, 11), (7, 11),
        (6, 10), (5, 10), (4, 10), (3, 10), (2, 10), (1, 10), (0, 10),
        (0, 9), (0, 8),
        (1, 8), (2, 8), (3, 8), (4, 8), (5, 8), (6, 8), (7, 8), (7, 7),
        (8, 6), (8, 5), (8, 4), (8, 3), (8, 2), (8, 1),
    )

    def __init__(self) -> None:
        self.nests: Dict[int, List[Piece]] = {}
        self.main_track: List[List[Piece]] = [[] for _ in range(self.MAIN_TRACK_LENGTH)]
        self.home_paths: Dict[int, List[Optional[Piece]]] = {}
        self.home_areas: Dict[int, List[Piece]] = {}
        self.piece_locations: Dict[Piece, ParsedLocation] = {}
        self.captured_pieces: Dict[int, List[Piece]] = {p: [] for p in range(1, self.NUM_PLAYERS + 1)}
        self.current_player: int = 1
        self.dice_rolls: Tuple[int, int] = (0, 0)
        self.rolls_remaining: List[int] = []
        self.doubles_count: int = 0
        self.last_error: str | None = None
        self._initialize_board()

    def _initialize_board(self) -> None:
        self.nests = {p: [] for p in range(1, self.NUM_PLAYERS + 1)}
        self.main_track = [[] for _ in range(self.MAIN_TRACK_LENGTH)]
        self.home_paths = {p: [None] * self.HOME_PATH_LENGTH for p in range(1, self.NUM_PLAYERS + 1)}
        self.home_areas = {p: [] for p in range(1, self.NUM_PLAYERS + 1)}
        self.piece_locations = {}
        for player in range(1, self.NUM_PLAYERS + 1):
            for token in range(1, self.TOKENS_PER_PLAYER + 1):
                piece = Piece[f"P{player}_TOKEN_{token}"]
                self.nests[player].append(piece)
                self.piece_locations[piece] = ParsedLocation("nest", player, token)

    def reset(self) -> None:
        self._initialize_board()
        self.captured_pieces = {p: [] for p in range(1, self.NUM_PLAYERS + 1)}
        self.current_player = 1
        self.dice_rolls = (0, 0)
        self.rolls_remaining = []
        self.doubles_count = 0
        self.last_error = None

    def copy(self) -> "ParcheesiState":
        nxt = cast(ParcheesiState, ParcheesiState.__new__(ParcheesiState))
        nxt.nests = {p: pieces[:] for p, pieces in self.nests.items()}
        nxt.main_track = [pieces[:] for pieces in self.main_track]
        nxt.home_paths = {p: path[:] for p, path in self.home_paths.items()}
        nxt.home_areas = {p: pieces[:] for p, pieces in self.home_areas.items()}
        nxt.piece_locations = self.piece_locations.copy()
        nxt.captured_pieces = {p: pieces[:] for p, pieces in self.captured_pieces.items()}
        nxt.current_player = self.current_player
        nxt.dice_rolls = self.dice_rolls
        nxt.rolls_remaining = self.rolls_remaining[:]
        nxt.doubles_count = self.doubles_count
        nxt.last_error = self.last_error
        return nxt

    @staticmethod
    def all_pieces_for_player(player: int) -> list[Piece]:
        return [Piece[f"P{player}_TOKEN_{token}"] for token in range(1, ParcheesiState.TOKENS_PER_PLAYER + 1)]

    @staticmethod
    def parse_location_id(location_id: str) -> ParsedLocation:
        parts = location_id.strip().lower().split("_")
        try:
            if parts[0] == "nest" and len(parts) in {2, 3}:
                player = int(parts[1])
                token = int(parts[2]) if len(parts) == 3 else 1
                return ParsedLocation("nest", player, token)
            if parts[0] == "main" and len(parts) == 2:
                return ParsedLocation("main", 0, int(parts[1]))
            if parts[0] == "home" and len(parts) == 3:
                return ParsedLocation("home", int(parts[1]), int(parts[2]))
            if parts[0] == "homearea" and len(parts) in {2, 3}:
                player = int(parts[1])
                token = int(parts[2]) if len(parts) == 3 else 1
                return ParsedLocation("homearea", player, token)
        except ValueError as exc:
            raise ValueError(f"invalid Parcheesi location ID: {location_id!r}") from exc
        raise ValueError(f"invalid Parcheesi location ID: {location_id!r}")

    @staticmethod
    def location_id(kind: str, player: int = 0, pos: int = 0) -> str:
        if kind == "nest":
            return f"nest_{player}_{pos}"
        if kind == "main":
            return f"main_{pos}"
        if kind == "home":
            return f"home_{player}_{pos}"
        if kind == "homearea":
            return f"homearea_{player}_{pos}"
        raise ValueError(f"unknown location kind: {kind}")

    def piece_location_id(self, piece: Piece) -> str:
        loc = self.piece_locations[piece]
        if loc.kind == "main":
            return self.location_id("main", pos=loc.pos)
        return self.location_id(loc.kind, loc.player, loc.pos)

    def piece_at_id(self, location_id: str) -> Piece:
        pieces = self.pieces_at_id(location_id)
        return pieces[0] if pieces else Piece.EMPTY

    def pieces_at_id(self, location_id: str) -> list[Piece]:
        loc = self.parse_location_id(location_id)
        if loc.kind == "nest":
            if loc.pos > 0:
                piece = Piece[f"P{loc.player}_TOKEN_{loc.pos}"]
                return [piece] if piece in self.nests[loc.player] else []
            return self.nests[loc.player][:]
        if loc.kind == "main":
            if 0 <= loc.pos < self.MAIN_TRACK_LENGTH:
                return self.main_track[loc.pos][:]
            return []
        if loc.kind == "home":
            if 0 <= loc.pos < self.HOME_PATH_LENGTH:
                piece = self.home_paths[loc.player][loc.pos]
                return [] if piece is None else [piece]
            return []
        if loc.kind == "homearea":
            if loc.pos > 0:
                piece = Piece[f"P{loc.player}_TOKEN_{loc.pos}"]
                return [piece] if piece in self.home_areas[loc.player] else []
            return self.home_areas[loc.player][:]
        return []

    def roll_dice(self, dice: tuple[int, int] | None = None) -> tuple[int, int]:
        self.dice_rolls = dice if dice is not None else (random.randint(1, 6), random.randint(1, 6))
        d1, d2 = self.dice_rolls
        if d1 == d2:
            self.doubles_count += 1
            self.rolls_remaining = [d1, d1, d1, d1]
        else:
            self.doubles_count = 0
            self.rolls_remaining = [d1, d2]
        self.last_error = None
        return self.dice_rolls

    def clear_dice(self) -> None:
        self.dice_rolls = (0, 0)
        self.rolls_remaining = []

    def end_turn(self) -> None:
        self.current_player = (self.current_player % self.NUM_PLAYERS) + 1
        self.clear_dice()

    def check_win_condition(self, player_num: int) -> bool:
        return len(self.home_areas[player_num]) == self.TOKENS_PER_PLAYER

    def is_blockade(self, position: int) -> bool:
        pieces = self.main_track[position]
        if len(pieces) < 2:
            return False
        owner = pieces[0].to_player_num()
        return all(piece.to_player_num() == owner for piece in pieces)

    def _roll_options(self) -> list[int]:
        if not self.rolls_remaining:
            return []
        out = sorted(set(self.rolls_remaining))
        if len(self.rolls_remaining) >= 2:
            total = sum(self.rolls_remaining)
            if total not in out:
                out.append(total)
        return out

    def _consume_roll_distance(self, distance: int) -> bool:
        if distance in self.rolls_remaining:
            self.rolls_remaining.remove(distance)
            return True
        if self.rolls_remaining and distance == sum(self.rolls_remaining):
            self.rolls_remaining.clear()
            return True
        return False

    def get_possible_moves(self, player_num: int, dice_rolls: Tuple[int, int] | None = None) -> List[Tuple[Piece, str, str]]:
        if dice_rolls is not None and dice_rolls != self.dice_rolls:
            trial = self.copy()
            trial.current_player = player_num
            trial.roll_dice(dice_rolls)
            return trial.get_possible_moves(player_num)

        moves: list[tuple[Piece, str, str]] = []
        for piece in self.all_pieces_for_player(player_num):
            loc = self.piece_locations[piece]
            if loc.kind == "homearea":
                continue
            start_id = self.piece_location_id(piece)
            for distance in self._roll_options():
                end_id = self._destination_for_distance(piece, distance)
                if end_id is not None:
                    moves.append((piece, start_id, end_id))
        return moves

    def move_is_capture(self, piece: Piece, end_id: str) -> bool:
        loc = self.parse_location_id(end_id)
        if loc.kind != "main" or loc.pos in self.SAFE_SQUARES:
            return False
        player = piece.to_player_num()
        return any(p.to_player_num() != player for p in self.main_track[loc.pos])

    def _destination_for_distance(self, piece: Piece, distance: int) -> str | None:
        player = piece.to_player_num()
        loc = self.piece_locations[piece]
        if loc.kind == "nest":
            if distance != 5:
                return None
            start = self.PLAYER_START_SQUARES[player]
            return self.location_id("main", pos=start) if self._can_land_on_main(player, start) else None

        if loc.kind == "main":
            return self._main_destination(player, loc.pos, distance, piece.to_token_num())

        if loc.kind == "home":
            target = loc.pos + distance
            if target < self.HOME_PATH_LENGTH:
                return self.location_id("home", player, target) if self.home_paths[player][target] is None else None
            if target == self.HOME_PATH_LENGTH:
                return self.location_id("homearea", player, piece.to_token_num())
            return None

        return None

    def _main_destination(self, player: int, current_pos: int, distance: int, token: int) -> str | None:
        steps_to_entry = (self.PLAYER_HOME_ENTRY_SQUARES[player] - current_pos) % self.MAIN_TRACK_LENGTH

        for step in range(1, min(distance, steps_to_entry) + 1):
            pos = (current_pos + step) % self.MAIN_TRACK_LENGTH
            if self.is_blockade(pos):
                return None

        if distance <= steps_to_entry:
            dest = (current_pos + distance) % self.MAIN_TRACK_LENGTH
            return self.location_id("main", pos=dest) if self._can_land_on_main(player, dest) else None

        home_index = distance - steps_to_entry - 1
        if home_index < self.HOME_PATH_LENGTH:
            return self.location_id("home", player, home_index) if self.home_paths[player][home_index] is None else None
        if home_index == self.HOME_PATH_LENGTH:
            return self.location_id("homearea", player, token)
        return None

    def _can_land_on_main(self, player: int, pos: int) -> bool:
        pieces = self.main_track[pos]
        if not pieces:
            return True
        owners = {piece.to_player_num() for piece in pieces}
        if owners == {player}:
            return len(pieces) < 2
        if self.is_blockade(pos):
            return False
        if pos in self.SAFE_SQUARES:
            return False
        return True

    def _remove_piece_from_current_location(self, piece: Piece) -> None:
        loc = self.piece_locations[piece]
        if loc.kind == "nest":
            if piece in self.nests[loc.player]:
                self.nests[loc.player].remove(piece)
        elif loc.kind == "main":
            if piece in self.main_track[loc.pos]:
                self.main_track[loc.pos].remove(piece)
        elif loc.kind == "home":
            if self.home_paths[loc.player][loc.pos] == piece:
                self.home_paths[loc.player][loc.pos] = None
        elif loc.kind == "homearea":
            if piece in self.home_areas[loc.player]:
                self.home_areas[loc.player].remove(piece)

    def _send_to_nest(self, piece: Piece) -> None:
        player = piece.to_player_num()
        self._remove_piece_from_current_location(piece)
        if piece not in self.nests[player]:
            self.nests[player].append(piece)
        self.piece_locations[piece] = ParsedLocation("nest", player, piece.to_token_num())

    def _find_piece_for_start(self, start_id: str, player_num: int) -> Piece | None:
        loc = self.parse_location_id(start_id)
        if loc.kind == "nest":
            if loc.pos > 0:
                piece = Piece[f"P{player_num}_TOKEN_{loc.pos}"]
                return piece if piece in self.nests[player_num] else None
            return self.nests[player_num][0] if self.nests[player_num] else None
        for piece in self.pieces_at_id(start_id):
            if piece.to_player_num() == player_num:
                return piece
        return None

    def apply_move(
        self,
        piece: Piece,
        start_id: str,
        end_id: str,
        *,
        consume_roll: bool = True,
        trusted: bool = False,
    ) -> Optional[str]:
        player = piece.to_player_num()
        if piece is Piece.EMPTY:
            return "cannot move EMPTY"
        if self.piece_location_id(piece) != start_id and not start_id.startswith(f"nest_{player}"):
            return f"{piece.name} is not at {start_id}"

        distance = self._distance_for_end(piece, end_id)
        if distance is None:
            return "destination is not reachable from selected piece"
        if not trusted:
            if not self.rolls_remaining:
                return "roll dice first"
            if distance not in self._roll_options():
                return f"move distance {distance} does not match remaining dice {self.rolls_remaining}"

        end_loc = self.parse_location_id(end_id)
        self._remove_piece_from_current_location(piece)

        if end_loc.kind == "main":
            dest_pieces = self.main_track[end_loc.pos]
            if dest_pieces and end_loc.pos not in self.SAFE_SQUARES:
                opponents = [p for p in dest_pieces if p.to_player_num() != player]
                for captured in opponents:
                    self.captured_pieces[captured.to_player_num()].append(captured)
                    self._send_to_nest(captured)
                dest_pieces = self.main_track[end_loc.pos]
            dest_pieces.append(piece)
            self.piece_locations[piece] = ParsedLocation("main", player, end_loc.pos)
        elif end_loc.kind == "home":
            self.home_paths[player][end_loc.pos] = piece
            self.piece_locations[piece] = ParsedLocation("home", player, end_loc.pos)
        elif end_loc.kind == "homearea":
            if piece not in self.home_areas[player]:
                self.home_areas[player].append(piece)
            self.piece_locations[piece] = ParsedLocation("homearea", player, piece.to_token_num())
        else:
            return f"cannot move to {end_id}"

        if consume_roll and not trusted:
            self._consume_roll_distance(distance)
        self.last_error = None
        return None

    def _distance_for_end(self, piece: Piece, end_id: str) -> int | None:
        for distance in range(1, 25):
            if self._destination_for_distance(piece, distance) == end_id:
                return distance
        return None

    def apply_move_trusted(self, start_id: str, end_id: str) -> None:
        try:
            piece = self._find_piece_for_start(start_id, 1)
        except ValueError as exc:
            print(f"p1_move ignored: {exc}", file=sys.stderr)
            return
        if piece is None:
            print(f"p1_move ignored: no P1 piece at {start_id}", file=sys.stderr)
            return
        err = self.apply_move(piece, start_id, end_id, consume_roll=False, trusted=True)
        if err:
            print(f"p1_move ignored: {err}: {start_id} -> {end_id}", file=sys.stderr)
            return
        self.current_player = 2
        self.clear_dice()

    def try_apply_p2_move(self, start_id: str, end_id: str) -> Optional[str]:
        try:
            piece = self._find_piece_for_start(start_id, 2)
        except ValueError as exc:
            return str(exc)
        if piece is None:
            return f"no P2 piece at {start_id}"
        if self.current_player != 2:
            return f"current player is P{self.current_player}, not P2"
        return self.apply_move(piece, start_id, end_id)

    @classmethod
    def track_grid_position(cls, pos: int) -> tuple[float, float]:
        return cls.TRACK_GRID_POSITIONS[pos % cls.MAIN_TRACK_LENGTH]

    @classmethod
    def home_grid_position(cls, player: int, pos: int) -> tuple[float, float]:
        t = min(max(pos, 0), cls.HOME_PATH_LENGTH - 1)
        home_lanes = {
            1: (9.0, float(t + 1)),
            2: (float(17 - t), 9.0),
            3: (9.0, float(17 - t)),
            4: (float(t + 1), 9.0),
        }
        return home_lanes[player]

    @classmethod
    def nest_grid_position(cls, player: int, token: int) -> tuple[float, float]:
        bases = {
            1: (4.0, 4.0),
            2: (14.0, 4.0),
            3: (14.0, 14.0),
            4: (4.0, 14.0),
        }
        offsets = {
            1: (-0.75, -0.75),
            2: (0.75, -0.75),
            3: (-0.75, 0.75),
            4: (0.75, 0.75),
        }
        bx, by = bases[player]
        ox, oy = offsets.get(token, (0.0, 0.0))
        return bx + ox, by + oy

    @classmethod
    def home_area_grid_position(cls, player: int, token: int = 1) -> tuple[float, float]:
        offsets = {
            1: (-0.65, -0.65),
            2: (0.65, -0.65),
            3: (0.65, 0.65),
            4: (-0.65, 0.65),
        }
        token_offsets = {
            1: (-0.22, -0.22),
            2: (0.22, -0.22),
            3: (-0.22, 0.22),
            4: (0.22, 0.22),
        }
        ox, oy = offsets[player]
        tx, ty = token_offsets.get(token, (0.0, 0.0))
        return 9.0 + ox + tx, 9.0 + oy + ty

    @classmethod
    def location_id_to_grid(cls, location_id: str) -> tuple[float, float]:
        loc = cls.parse_location_id(location_id)
        if loc.kind == "main":
            return cls.track_grid_position(loc.pos)
        if loc.kind == "home":
            return cls.home_grid_position(loc.player, loc.pos)
        if loc.kind == "nest":
            return cls.nest_grid_position(loc.player, loc.pos)
        if loc.kind == "homearea":
            return cls.home_area_grid_position(loc.player, loc.pos)
        raise ValueError(f"unsupported Parcheesi location: {location_id}")

    @classmethod
    def iter_draw_locations(cls) -> Iterable[str]:
        for i in range(cls.MAIN_TRACK_LENGTH):
            yield cls.location_id("main", pos=i)
        for player in range(1, cls.NUM_PLAYERS + 1):
            for token in range(1, cls.TOKENS_PER_PLAYER + 1):
                yield cls.location_id("nest", player, token)
                yield cls.location_id("homearea", player, token)
            for pos in range(cls.HOME_PATH_LENGTH):
                yield cls.location_id("home", player, pos)
