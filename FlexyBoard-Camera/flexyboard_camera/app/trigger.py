from __future__ import annotations

import sys
import time
import threading


class TriggerError(RuntimeError):
    pass


def wait_for_software_trigger() -> None:
    input("Press Enter to trigger end-turn cycle... ")


def _extend_with_system_python_paths() -> None:
    major = sys.version_info.major
    minor = sys.version_info.minor
    candidates = [
        f"/usr/local/lib/python{major}.{minor}/dist-packages",
        "/usr/lib/python3/dist-packages",
        f"/usr/lib/python{major}.{minor}/dist-packages",
    ]
    for path in candidates:
        if path not in sys.path:
            sys.path.append(path)


class GPIOControlPanel:
    def __init__(self, *, button_pin: int | None = None, led_pin: int | None = None) -> None:
        if button_pin is None and led_pin is None:
            raise TriggerError("GPIOControlPanel requires at least one configured pin")

        self.button_pin = button_pin
        self.led_pin = led_pin
        self._player_ready_on = False
        self._backend = ""
        self._gpio = None
        self._button = None
        self._led = None
        self._button_latched = False
        self._lock = threading.Lock()

        try:
            import RPi.GPIO as GPIO  # type: ignore
        except Exception:
            GPIO = None

        if GPIO is not None:
            self._backend = "rpi_gpio"
            self._gpio = GPIO
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BCM)
            if self.button_pin is not None:
                GPIO.setup(self.button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
                GPIO.add_event_detect(self.button_pin, GPIO.FALLING, callback=self._handle_rpi_gpio_press, bouncetime=60)
            if self.led_pin is not None:
                GPIO.setup(self.led_pin, GPIO.OUT, initial=GPIO.LOW)
            return

        _extend_with_system_python_paths()
        try:
            from gpiozero import Button, Device, LED  # type: ignore
            from gpiozero.pins.lgpio import LGPIOFactory  # type: ignore
        except Exception as exc:
            raise TriggerError("No usable GPIO backend found; install RPi.GPIO or gpiozero+lgpio") from exc

        try:
            Device.pin_factory = LGPIOFactory()
        except Exception:
            # If gpiozero already has a working pin factory, keep it.
            pass

        self._backend = "gpiozero"
        if self.button_pin is not None:
            self._button = Button(self.button_pin, pull_up=True, bounce_time=0.03)
            self._button.when_pressed = self._handle_gpiozero_press
        if self.led_pin is not None:
            self._led = LED(self.led_pin)

    def _latch_button_press(self) -> None:
        with self._lock:
            self._button_latched = True

    def _consume_button_latch(self) -> bool:
        with self._lock:
            if not self._button_latched:
                return False
            self._button_latched = False
            return True

    def _clear_button_latch(self) -> None:
        with self._lock:
            self._button_latched = False

    def _handle_rpi_gpio_press(self, _channel: int) -> None:
        self._latch_button_press()

    def _handle_gpiozero_press(self) -> None:
        self._latch_button_press()

    def _button_is_pressed(self) -> bool:
        if self.button_pin is None:
            return False
        if self._backend == "gpiozero":
            assert self._button is not None
            return bool(self._button.is_pressed)
        assert self._gpio is not None
        return self._gpio.input(self.button_pin) == self._gpio.LOW

    def _wait_until_released(self) -> None:
        while self._button_is_pressed():
            time.sleep(0.01)
        # Drop any extra edge callbacks that may have fired during the same
        # physical press so the next wait requires a new press.
        self._clear_button_latch()

    def set_player_ready(self, on: bool) -> None:
        self._player_ready_on = bool(on)
        if self.led_pin is None:
            return
        if self._backend == "rpi_gpio":
            assert self._gpio is not None
            self._gpio.output(self.led_pin, self._gpio.HIGH if on else self._gpio.LOW)
            return
        if self._led is not None:
            if on:
                self._led.on()
            else:
                self._led.off()

    def wait_for_button(self, timeout_sec: float | None = None) -> bool:
        if self.button_pin is None:
            raise TriggerError("GPIO button pin is not configured")

        start = time.monotonic()
        while True:
            if self._consume_button_latch():
                self._wait_until_released()
                return True
            if self._button_is_pressed():
                time.sleep(0.03)
                self._wait_until_released()
                return True
            if timeout_sec is not None and (time.monotonic() - start) >= timeout_sec:
                return False
            time.sleep(0.01)

    def cleanup(self) -> None:
        if self._backend == "gpiozero":
            if self._led is not None:
                self._led.off()
                self._led.close()
            if self._button is not None:
                self._button.close()
            return

        if self._gpio is None:
            return
        pins: list[int] = []
        if self.led_pin is not None:
            self._gpio.output(self.led_pin, self._gpio.LOW)
            pins.append(self.led_pin)
        if self.button_pin is not None:
            try:
                self._gpio.remove_event_detect(self.button_pin)
            except Exception:
                pass
            pins.append(self.button_pin)
        if pins:
            self._gpio.cleanup(pins)


def wait_for_gpio_trigger(pin: int = 17, timeout_sec: float | None = None) -> bool:
    panel = GPIOControlPanel(button_pin=pin)
    try:
        return panel.wait_for_button(timeout_sec=timeout_sec)
    finally:
        panel.cleanup()
