#ifndef MAIN_H
#define MAIN_H

#include <stdbool.h>
#include <stdint.h>

/* Base addresses */
#define PERIPH_BASE           0x40000000UL
#define APB1PERIPH_BASE       PERIPH_BASE
#define AHB1PERIPH_BASE       (PERIPH_BASE + 0x00020000UL)
#define RCC_BASE              (AHB1PERIPH_BASE + 0x3800UL)
#define GPIOA_BASE            (AHB1PERIPH_BASE + 0x0000UL)
#define GPIOB_BASE            (AHB1PERIPH_BASE + 0x0400UL)
#define USART2_BASE           (APB1PERIPH_BASE + 0x4400UL)

/* GPIO register layout */
typedef struct
{
    volatile uint32_t MODER;
    volatile uint32_t OTYPER;
    volatile uint32_t OSPEEDR;
    volatile uint32_t PUPDR;
    volatile uint32_t IDR;
    volatile uint32_t ODR;
    volatile uint32_t BSRR;
    volatile uint32_t LCKR;
    volatile uint32_t AFR[2];
} GPIO_TypeDef;

/* USART register layout */
typedef struct
{
    volatile uint32_t SR;
    volatile uint32_t DR;
    volatile uint32_t BRR;
    volatile uint32_t CR1;
    volatile uint32_t CR2;
    volatile uint32_t CR3;
    volatile uint32_t GTPR;
} USART_TypeDef;

/* RCC registers */
#define RCC_AHB1ENR           (*(volatile uint32_t *)(RCC_BASE + 0x30UL))
#define RCC_APB1ENR           (*(volatile uint32_t *)(RCC_BASE + 0x40UL))

/* GPIO ports */
#define GPIOA                 ((GPIO_TypeDef *)GPIOA_BASE)
#define GPIOB                 ((GPIO_TypeDef *)GPIOB_BASE)
#define USART2                ((USART_TypeDef *)USART2_BASE)

/* X axis on PA pins */
#define X_STEP_PORT           GPIOA
#define X_STEP_PIN            0   /* PA0, Arduino A0 */

#define X_DIR_PORT            GPIOA
#define X_DIR_PIN             1   /* PA1, Arduino A1 */

/* Y axis on Arduino D pins */
#define Y_STEP_PORT           GPIOA
#define Y_STEP_PIN            10  /* PA10, board-labeled D2 */

#define Y1_DIR_PORT           GPIOB
#define Y1_DIR_PIN            3   /* PB3, board-labeled D3 */

#define Y2_DIR_PORT           GPIOB
#define Y2_DIR_PIN            5   /* PB5, board-labeled D4 */

/* Z axis on Arduino D pins */
#define Z_STEP_PORT           GPIOB
#define Z_STEP_PIN            4   /* PB4, board-labeled D5 */

#define Z_DIR_PORT            GPIOB
#define Z_DIR_PIN             10  /* PB10, board-labeled D6 */

/* UART2 on NUCLEO VCP (ST-LINK virtual COM): PA2=TX, PA3=RX, AF7 */
#define UART_TX_PORT          GPIOA
#define UART_TX_PIN           2
#define UART_RX_PORT          GPIOA
#define UART_RX_PIN           3
#define UART_AF_INDEX         7
#define UART_BAUDRATE         115200U
#define CORE_CLOCK_HZ         16000000U /* default HSI if no PLL init */

/* Board coordinate calibration (steps from machine origin) */
#define BOARD_GRID_MAX_INDEX  7

/* Measured corners (board coords):
 * Corner 0 -> (x=0,y=0) -> CORNER_00
 * Corner 1 -> (x=0,y=7) -> CORNER_07
 * Corner 2 -> (x=7,y=7) -> CORNER_77
 * Corner 3 -> (x=7,y=0) -> CORNER_70
 */
#define CORNER_00_X_STEPS     321
#define CORNER_00_Y_STEPS     333
#define CORNER_70_X_STEPS     1670
#define CORNER_70_Y_STEPS     333
#define CORNER_07_X_STEPS     321
#define CORNER_07_Y_STEPS     1671
#define CORNER_77_X_STEPS     1670
#define CORNER_77_Y_STEPS     1671

/* Full reachable workspace (green-grid / gantry area) in motor steps.
 * Used for off-board staging moves (captured pieces, detours).
 */
#define WORKSPACE_MIN_X_STEPS 0
#define WORKSPACE_MAX_X_STEPS 2055
#define WORKSPACE_MIN_Y_STEPS 0
#define WORKSPACE_MAX_Y_STEPS 2280

/* Percentage command scale:
 * 0   -> 0%
 * 100 -> 100%
 */
#define WORKSPACE_PERCENT_SCALE 100

/* Step pulse timing (higher delay = slower speed). */
#define STEP_DELAY_CYCLES     1400
#define STEP_DELAY_CYCLES_Z   3000
#define MOVE_PAUSE_CYCLES     1200000

/* Z actuation profile for pickup/release around a piece move.
 * TODO: flip *_DIR values if your physical Z direction is reversed.
 */
#define Z_PICKUP_STEPS        30
#define Z_RELEASE_STEPS       30
#define Z_PICKUP_DIR          1U
#define Z_RELEASE_DIR         0U

void gpio_init(void);
void delay_cycles(volatile uint32_t count);
void uart_init(void);

void set_x_dir(uint8_t dir);
void set_y_dir(uint8_t dir);
void set_z_dir(uint8_t dir);

void pulse_x_step(void);
void pulse_y_step(void);
void pulse_z_step(void);

void step_x(uint32_t steps, uint8_t dir);
void step_y(uint32_t steps, uint8_t dir);
void step_z(uint32_t steps, uint8_t dir);
void move_xy(uint32_t x_steps, uint8_t x_dir, uint32_t y_steps, uint8_t y_dir);
void move_to_steps(int32_t target_x_steps, int32_t target_y_steps);
bool board_coord_to_steps(int32_t board_x, int32_t board_y, int32_t *out_x_steps, int32_t *out_y_steps);

#endif
