from __future__ import annotations

import time


class TriggerError(RuntimeError):
    pass


def wait_for_software_trigger() -> None:
    input("Press Enter to trigger end-turn cycle... ")


def wait_for_gpio_trigger(pin: int = 17, timeout_sec: float | None = None) -> bool:
    try:
        import RPi.GPIO as GPIO  # type: ignore
    except Exception as exc:
        raise TriggerError("RPi.GPIO is unavailable on this system") from exc

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    start = time.monotonic()
    try:
        while True:
            if GPIO.input(pin) == GPIO.LOW:
                return True
            if timeout_sec is not None and (time.monotonic() - start) >= timeout_sec:
                return False
            time.sleep(0.01)
    finally:
        GPIO.cleanup(pin)
