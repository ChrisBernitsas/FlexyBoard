#ifndef MOTOR_SIMPLE_MAIN_H
#define MOTOR_SIMPLE_MAIN_H

#include <stdint.h>

/* Base addresses */
#define PERIPH_BASE        0x40000000UL
#define AHB1PERIPH_BASE    (PERIPH_BASE + 0x00020000UL)
#define RCC_BASE           (AHB1PERIPH_BASE + 0x3800UL)
#define GPIOA_BASE         (AHB1PERIPH_BASE + 0x0000UL)

/* RCC registers */
#define RCC_AHB1ENR        (*(volatile uint32_t *)(RCC_BASE + 0x30UL))

/* GPIOA registers */
#define GPIOA_MODER        (*(volatile uint32_t *)(GPIOA_BASE + 0x00UL))
#define GPIOA_OTYPER       (*(volatile uint32_t *)(GPIOA_BASE + 0x04UL))
#define GPIOA_OSPEEDR      (*(volatile uint32_t *)(GPIOA_BASE + 0x08UL))
#define GPIOA_PUPDR        (*(volatile uint32_t *)(GPIOA_BASE + 0x0CUL))
#define GPIOA_ODR          (*(volatile uint32_t *)(GPIOA_BASE + 0x14UL))

/*
 * NUCLEO-F446RE simple one-motor test mapping:
 * A0 -> PA0 (STEP)
 * A1 -> PA1 (DIR)
 * A2 -> PA4 (optional EN)
 */
#define STEP_PIN           0u   /* PA0 / A0 */
#define DIR_PIN            1u   /* PA1 / A1 */

/* Set to 1 if firmware should drive EN_PIN. */
#define USE_EN_PIN         0u
#define EN_PIN             4u   /* PA4 / A2, active low for A4988-style enable */

/* Test parameters */
#define TEST_STEPS         2000u
#define STEP_DELAY_CYCLES  5000u // lower is faster
#define DIR_SETTLE_DELAY_CYCLES  2000u
#define PAUSE_DELAY_CYCLES 3000000u

void gpio_init(void);
void delay_cycles(volatile uint32_t count);
void step_motor(uint32_t steps, uint8_t dir);
void enable_driver(void);
void disable_driver(void);

#endif
