# motor_simple_code

Standalone teammate-style one-motor test at top level of `Design Code`.

## Pin mapping
- `A0` (`PA0`) -> stepper driver `STEP`
- `A1` (`PA1`) -> stepper driver `DIR`
- optional: `A2` (`PA4`) -> stepper driver `EN` (set `USE_EN_PIN=1` in `main.h`)

## Wiring minimum
- STM32 `A0` -> driver `STEP`
- STM32 `A1` -> driver `DIR`
- STM32 `GND` -> driver logic `GND`
- external motor PSU -> driver `VMOT` and PSU `GND`
- motor coils -> driver coil outputs

## Build and flash
```bash
cd /Users/christopher/Desktop/Design\ Code/motor_simple_code
chmod +x build.sh flash.sh
./build.sh
./flash.sh
```

## Notes
- Do not power motor coils from STM32.
- If driver `EN` is not MCU-controlled, hardwire EN to enabled state.
- In default config (`USE_EN_PIN=0`), `A2` is unused.

## Speed / smoothness tuning
Edit constants in `main.h`:
- `STEP_DELAY_CYCLES`: fixed pulse high/low delay (lower = faster)
- `DIR_SETTLE_DELAY_CYCLES`: small wait after direction change
- `TEST_STEPS`: how many steps per forward/back segment
- `PAUSE_DELAY_CYCLES`: pause between forward and reverse segments

Lower delay values = faster motion.
