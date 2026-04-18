#include "main.h"

void gpio_init(void) {
  /* Enable clock for GPIOA */
  RCC_AHB1ENR |= (1u << 0);

  /* Set STEP, DIR, optional EN as outputs */
  GPIOA_MODER &= ~((3u << (STEP_PIN * 2u)) | (3u << (DIR_PIN * 2u))
#if USE_EN_PIN
                   | (3u << (EN_PIN * 2u))
#endif
                   );

  GPIOA_MODER |= ((1u << (STEP_PIN * 2u)) | (1u << (DIR_PIN * 2u))
#if USE_EN_PIN
                  | (1u << (EN_PIN * 2u))
#endif
                  );

  GPIOA_OTYPER &= ~((1u << STEP_PIN) | (1u << DIR_PIN)
#if USE_EN_PIN
                    | (1u << EN_PIN)
#endif
                    );

  GPIOA_PUPDR &= ~((3u << (STEP_PIN * 2u)) | (3u << (DIR_PIN * 2u))
#if USE_EN_PIN
                   | (3u << (EN_PIN * 2u))
#endif
                   );

  GPIOA_OSPEEDR |= ((2u << (STEP_PIN * 2u)) | (2u << (DIR_PIN * 2u))
#if USE_EN_PIN
                    | (2u << (EN_PIN * 2u))
#endif
                    );
}

void delay_cycles(volatile uint32_t count) {
  while (count-- > 0u) {
    __asm__("nop");
  }
}

void enable_driver(void) {
#if USE_EN_PIN
  GPIOA_ODR &= ~(1u << EN_PIN);
#endif
}

void disable_driver(void) {
#if USE_EN_PIN
  GPIOA_ODR |= (1u << EN_PIN);
#endif
}

void step_motor(uint32_t steps, uint8_t dir) {
  if (dir != 0u) {
    GPIOA_ODR |= (1u << DIR_PIN);
  } else {
    GPIOA_ODR &= ~(1u << DIR_PIN);
  }

  delay_cycles(DIR_SETTLE_DELAY_CYCLES);

  for (uint32_t i = 0; i < steps; ++i) {
    GPIOA_ODR |= (1u << STEP_PIN);
    delay_cycles(STEP_DELAY_CYCLES);

    GPIOA_ODR &= ~(1u << STEP_PIN);
    delay_cycles(STEP_DELAY_CYCLES);
  }
}

int main(void) {
  gpio_init();
  enable_driver();

  while (1) {
    step_motor(TEST_STEPS, 1u);
    delay_cycles(PAUSE_DELAY_CYCLES);

    step_motor(TEST_STEPS, 0u);
    delay_cycles(PAUSE_DELAY_CYCLES);
  }
}
