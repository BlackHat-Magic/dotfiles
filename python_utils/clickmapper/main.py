#!/usr/bin/env python3
"""
clickmapper - Map a position in one range to noisy keypresses in another range.

Listens for left-click events on a mouse/touchpad device, then sends a keypress
that approximates the given position in the output range with Gaussian noise.
"""

import argparse
import random
import subprocess
import sys
import time

from evdev import InputDevice, ecodes


def parse_range(s: str) -> tuple[int, int]:
    """Parse a range string like '1..13' into (start, end) inclusive."""
    parts = s.split("..")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Range must be in format 'start..end', got '{s}'"
        )
    start, end = int(parts[0]), int(parts[1])
    if start > end:
        raise argparse.ArgumentTypeError(f"Start ({start}) must be <= end ({end})")
    return start, end


# Mapping from output value to Linux input scancodes
# KEY_1=2, KEY_2=3, ..., KEY_9=10, KEY_0=11
DEFAULT_KEY_MAP: dict[int, int] = {
    1: 2,
    2: 3,
    3: 4,
    4: 5,
    5: 6,
    6: 7,
    7: 8,
    8: 9,
    9: 10,
    10: 11,
}


def build_key_map(
    output_range: tuple[int, int], keys: list[str] | None
) -> dict[int, int]:
    """Build a mapping from output values to scancodes."""
    start, end = output_range
    n = end - start + 1

    if keys:
        if len(keys) != n:
            print(
                f"Error: need {n} keys for range {start}..{end}, got {len(keys)}",
                file=sys.stderr,
            )
            sys.exit(1)
        key_map: dict[int, int] = {}
        for i in range(n):
            val = start + i
            try:
                key_map[val] = int(keys[i])
            except ValueError:
                print(
                    f"Error: key '{keys[i]}' is not a valid scancode",
                    file=sys.stderr,
                )
                sys.exit(1)
        return key_map

    key_map = {}
    for i in range(n):
        val = start + i
        key_map[val] = DEFAULT_KEY_MAP.get(val, 0)
    return key_map


def find_mouse_device(device_path: str | None) -> InputDevice:
    """Find the mouse/touchpad device to listen on."""
    if device_path:
        return InputDevice(device_path)

    from evdev import list_devices

    devices = [InputDevice(path) for path in list_devices()]
    for keyword in ["Touchpad", "touchpad", "Mouse", "mouse"]:
        for dev in devices:
            if keyword in dev.name:
                return dev

    for dev in devices:
        if ecodes.BTN_LEFT in dev.capabilities().get(ecodes.EV_KEY, []):
            return dev

    print("Error: no mouse device found. Specify one with --device.", file=sys.stderr)
    sys.exit(1)


def send_key(scancode: int, hold_ms: int = 50) -> None:
    """Send a keypress via ydotool using scancodes."""
    subprocess.run(
        ["ydotool", "key", "-d", str(hold_ms), f"{scancode}:1", f"{scancode}:0"],
        check=False,
        capture_output=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map a position in one range to noisy keypresses in another range."
    )
    parser.add_argument(
        "input_range",
        type=parse_range,
        help="Input range to interpolate between (e.g., 1..13)",
    )
    parser.add_argument(
        "output_range",
        type=parse_range,
        help="Output range to sample from (e.g., 1..10)",
    )
    parser.add_argument(
        "position",
        type=float,
        help="Position in the input range (can be fractional)",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default=None,
        help="Path to the evdev input device (e.g., /dev/input/event7)",
    )
    parser.add_argument(
        "--keys",
        "-k",
        nargs="+",
        default=None,
        help="Keys to send for each output value (in order). Default: 1 2 3 4 5 6 7 8 9 0",
    )
    parser.add_argument(
        "--noise",
        "-n",
        type=float,
        default=0.5,
        help="Standard deviation of Gaussian noise (in output units). Default: 0.5",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.25,
        help="Delay after click before sending keypress, in seconds. Default: 0.25",
    )
    parser.add_argument(
        "--key-hold",
        type=int,
        default=50,
        help="How long to hold the key in ms. Default: 50",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available input devices and exit",
    )

    args = parser.parse_args()

    if args.list_devices:
        from evdev import list_devices

        for path in list_devices():
            dev = InputDevice(path)
            print(f"{path}: {dev.name}")
        return

    in_start, in_end = args.input_range
    out_start, out_end = args.output_range
    position = args.position

    if position < in_start or position > in_end:
        print(
            f"Warning: position {position} is outside input range {in_start}..{in_end}",
            file=sys.stderr,
        )

    key_map = build_key_map(args.output_range, args.keys)

    try:
        dev = find_mouse_device(args.device)
    except PermissionError as e:
        print(f"Error: {e}", file=sys.stderr)
        print(
            "\nFix by running with sudo, or add your user to the input group:",
            file=sys.stderr,
        )
        print(
            "  sudo usermod -aG input $USER  # then log out and back in",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Listening on: {dev.name} ({dev.path})")
    print(f"Input range: {in_start}..{in_end}, position: {position}")
    print(f"Output range: {out_start}..{out_end}, noise std: {args.noise}")
    print(f"Key map: {key_map}")
    print("Press left-click to send a keypress. Ctrl+C to quit.\n")

    t = (position - in_start) / (in_end - in_start) if in_end != in_start else 0.5
    target = out_start + t * (out_end - out_start)

    print(f"Normalized position: {t:.4f}, target output: {target:.4f}\n")

    try:
        for event in dev.read_loop():
            if event.type == ecodes.EV_KEY and event.code == ecodes.BTN_LEFT:
                if event.value == 1:
                    time.sleep(args.delay)

                    noisy = target + random.gauss(0, args.noise)
                    noisy = max(out_start, min(out_end, noisy))
                    key_val = round(noisy)
                    key_val = max(out_start, min(out_end, key_val))

                    scancode = key_map[key_val]
                    print(
                        f"  -> target={target:.2f}, noisy={noisy:.2f}, scancode={scancode}"
                    )
                    send_key(scancode, args.key_hold)
    except KeyboardInterrupt:
        print("\nExiting.")


def cli_entry_point() -> None:
    main()


if __name__ == "__main__":
    main()
