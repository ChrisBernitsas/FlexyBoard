import time
import sys
from pathlib import Path

# Add the parent directory of Software-GUI to the Python path
# This is a bit of a hack, but it allows us to import modules from Software-GUI
# without installing it as a package.
sys.path.append(str(Path(__file__).parent.parent.parent / "Software-GUI"))

from ipc.transport_tcp import TcpClientTransport
from ipc.protocol import P2MoveMessage
from coords import Square, format_square

PI_HOST = "flexyboard-pi.local"
BRIDGE_PORT = 8765

def generate_snake_pattern_moves():
    moves = []
    prev_square = Square(0, 0) # Start at A1 (0,0)
    moves.append(prev_square)

    for rank_index in range(8):
        if rank_index % 2 == 0:  # Even ranks (1, 3, 5, 7) go a-h
            for file_index in range(8):
                current_square = Square(file_index, rank_index)
                if current_square != prev_square: # Avoid duplicate if starting at 0,0
                    moves.append(current_square)
                prev_square = current_square
        else:  # Odd ranks (2, 4, 6, 8) go h-a
            for file_index in range(7, -1, -1):
                current_square = Square(file_index, rank_index)
                moves.append(current_square)
                prev_square = current_square
    return moves

def main():
    print(f"Connecting to {PI_HOST}:{BRIDGE_PORT}...")
    client = TcpClientTransport(PI_HOST, BRIDGE_PORT)
    try:
        client.connect(retries=5, retry_delay_sec=1.0)
        print("Connected to Raspberry Pi.")

        # Move to home position (0,0) first
        print("Moving to home position (0,0)...")
        home_sequence = ["0,0 -> 0,0"]
        home_message = P2MoveMessage(frm="home", to="home", stm_sequence=home_sequence)
        client.send_p2_move(home_message)
        time.sleep(2) # Give time for the motor to move

        squares_to_visit = generate_snake_pattern_moves()

        prev_x, prev_y = 0, 0 # Start at (0,0) for the first move

        for i, current_square in enumerate(squares_to_visit):
            curr_x, curr_y = current_square.file_index, current_square.rank_index
            square_id = format_square(curr_x, curr_y)
            print(f"Moving to {square_id} ({curr_x},{curr_y})...")

            stm_sequence = [f"{prev_x},{prev_y} -> {curr_x},{curr_y}"]
            p2_move_msg = P2MoveMessage(
                frm=format_square(prev_x, prev_y),
                to=square_id,
                stm_sequence=stm_sequence
            )
            client.send_p2_move(p2_move_msg)
            time.sleep(2) # Adjust delay as needed for motor movement

            prev_x, prev_y = curr_x, curr_y

        print("All squares visited. Returning to home position (0,0)...")
        home_sequence = [f"{prev_x},{prev_y} -> 0,0"]
        home_message = P2MoveMessage(frm=format_square(prev_x, prev_y), to="home", stm_sequence=home_sequence)
        client.send_p2_move(home_message)
        time.sleep(2)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Closing connection.")
        client.close()

if __name__ == "__main__":
    main()
