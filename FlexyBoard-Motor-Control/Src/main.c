#include "main.h"

#include <ctype.h>
#include <stdio.h>
#include <string.h>

static int32_t g_current_x_steps = 0;
static int32_t g_current_y_steps = 0;
static int32_t g_current_y_left_steps = 0;
static int32_t g_current_z_steps = 0;
static int32_t g_prev_moveheld_dx_steps = 0;
static int32_t g_prev_moveheld_dy_steps = 0;
static uint8_t g_has_prev_moveheld_direction = 0U;

typedef struct
{
    uint32_t x_step_delay_us;
    uint32_t y_step_delay_us;
    uint32_t xy_start_delay_us;
    uint32_t xy_ramp_steps;
    uint32_t z_step_delay_us;
    uint32_t step_pulse_high_us;
    uint32_t z_pre_pickup_pause_cycles;
    uint32_t moveheld_settle_cycles;
    uint32_t z_post_release_pause_cycles;
    uint32_t return_start_seat_steps;
} motion_runtime_config_t;

#define NVIC_ISER0            (*(volatile uint32_t *)0xE000E100UL)
#define NVIC_ISER1            (*(volatile uint32_t *)0xE000E104UL)

#define RCC_APB1ENR_TIM4EN    (1U << 2)
#define TIM_CR1_CEN           (1U << 0)
#define TIM_DIER_UIE          (1U << 0)
#define TIM_SR_UIF            (1U << 0)
#define TIM_EGR_UG            (1U << 0)
#define TIM4_IRQ_NUM          30U

typedef enum
{
    MOTION_KIND_IDLE = 0,
    MOTION_KIND_XY = 1,
    MOTION_KIND_Z = 2,
} motion_kind_t;

typedef struct
{
    volatile uint8_t active;
    volatile uint8_t pending_x;
    volatile uint8_t pending_y_right;
    volatile uint8_t pending_y_left;
    volatile uint8_t kind;
    volatile uint32_t step_index;
    volatile uint32_t total_steps;
    volatile uint32_t x_steps;
    volatile uint32_t y_right_steps;
    volatile uint32_t y_left_steps;
    volatile uint32_t x_acc;
    volatile uint32_t y_right_acc;
    volatile uint32_t y_left_acc;
    volatile uint32_t cruise_delay_us;
    volatile uint32_t z_delay_us;
} motion_state_t;

static volatile motion_state_t g_motion = {0};
static motion_runtime_config_t g_motion_cfg = {
    .x_step_delay_us = X_STEP_DELAY_CYCLES,
    .y_step_delay_us = Y_STEP_DELAY_CYCLES,
    .xy_start_delay_us = XY_START_DELAY_CYCLES,
    .xy_ramp_steps = XY_RAMP_STEPS,
    .z_step_delay_us = STEP_DELAY_CYCLES_Z,
    .step_pulse_high_us = STEP_PULSE_HIGH_US,
    .z_pre_pickup_pause_cycles = Z_PRE_PICKUP_PAUSE_CYCLES,
    .moveheld_settle_cycles = MOVEHELD_SETTLE_CYCLES,
    .z_post_release_pause_cycles = Z_POST_RELEASE_PAUSE_CYCLES,
    .return_start_seat_steps = RETURN_START_SEAT_STEPS,
};

static void move_to_steps_common(int32_t target_x_steps, int32_t target_y_steps);

/* Called from startup before main(). Provide a local implementation so
 * bare-metal builds do not jump to an undefined weak symbol.
 * Also enable FPU access because this firmware uses float math.
 */
void SystemInit(void)
{
    volatile uint32_t *cpacr = (volatile uint32_t *)0xE000ED88UL;
    *cpacr |= (0xFU << 20);
}

static void gpio_pin_output_init(GPIO_TypeDef *port, uint8_t pin)
{
    port->MODER &= ~(3U << (pin * 2));
    port->MODER |= (1U << (pin * 2));

    port->OTYPER &= ~(1U << pin);

    port->PUPDR &= ~(3U << (pin * 2));

    port->OSPEEDR &= ~(3U << (pin * 2));
    port->OSPEEDR |= (2U << (pin * 2));
}

static void gpio_pin_alt_init(GPIO_TypeDef *port, uint8_t pin, uint8_t af_index)
{
    uint32_t afr_slot = (pin >= 8U) ? 1U : 0U;
    uint32_t afr_shift = (pin % 8U) * 4U;

    port->MODER &= ~(3U << (pin * 2));
    port->MODER |= (2U << (pin * 2)); /* alternate function */

    port->OTYPER &= ~(1U << pin);

    port->PUPDR &= ~(3U << (pin * 2));

    port->OSPEEDR &= ~(3U << (pin * 2));
    port->OSPEEDR |= (2U << (pin * 2));

    port->AFR[afr_slot] &= ~(0xFU << afr_shift);
    port->AFR[afr_slot] |= ((uint32_t)af_index << afr_shift);
}

static void gpio_set_pin(GPIO_TypeDef *port, uint8_t pin)
{
    port->ODR |= (1U << pin);
}

static void gpio_clear_pin(GPIO_TypeDef *port, uint8_t pin)
{
    port->ODR &= ~(1U << pin);
}

static void pulse_step_pin(GPIO_TypeDef *port, uint8_t pin)
{
    uint32_t loops_per_us = (CORE_CLOCK_HZ / 1000000U) / 4U;
    if (loops_per_us == 0U)
    {
        loops_per_us = 1U;
    }
    gpio_set_pin(port, pin);
    delay_cycles(loops_per_us * STEP_PULSE_HIGH_US);
    gpio_clear_pin(port, pin);
}

static void nvic_enable_irq(uint8_t irq_num)
{
    if (irq_num < 32U)
    {
        NVIC_ISER0 = (1UL << irq_num);
    }
    else
    {
        NVIC_ISER1 = (1UL << (irq_num - 32U));
    }
}

static void nvic_set_priority(uint8_t irq_num, uint8_t priority)
{
    volatile uint8_t *nvic_ipr = (volatile uint8_t *)0xE000E400UL;
    nvic_ipr[irq_num] = (priority << 4);
}

static void clear_all_step_pins(void)
{
    gpio_clear_pin(X_STEP_PORT, X_STEP_PIN);
    gpio_clear_pin(Y_RIGHT_STEP_PORT, Y_RIGHT_STEP_PIN);
    gpio_clear_pin(Y_LEFT_STEP_PORT, Y_LEFT_STEP_PIN);
    gpio_clear_pin(Z_STEP_PORT, Z_STEP_PIN);
}

static uint32_t clamp_step_period_us(uint32_t period_us)
{
    uint32_t pulse_high_us = g_motion_cfg.step_pulse_high_us;

    if (pulse_high_us == 0U)
    {
        pulse_high_us = 1U;
    }

    if (period_us <= pulse_high_us)
    {
        return pulse_high_us + 1U;
    }
    return period_us;
}

static uint32_t pulse_high_delay_count(void)
{
    uint32_t loops_per_us = (CORE_CLOCK_HZ / 1000000U) / 4U;
    uint32_t pulse_high_us = g_motion_cfg.step_pulse_high_us;
    if (loops_per_us == 0U)
    {
        loops_per_us = 1U;
    }
    if (pulse_high_us == 0U)
    {
        pulse_high_us = 1U;
    }
    return loops_per_us * pulse_high_us;
}

static void motion_timer_set_interval_us(uint32_t interval_us)
{
    interval_us = clamp_step_period_us(interval_us);

    TIM4->ARR = interval_us - 1U;
}

static void motion_timer_stop(void)
{
    TIM4->CR1 &= ~TIM_CR1_CEN;
    TIM4->DIER &= ~TIM_DIER_UIE;
    TIM4->CNT = 0U;
    TIM4->SR = 0U;
    clear_all_step_pins();

    g_motion.active = 0U;
    g_motion.pending_x = 0U;
    g_motion.pending_y_right = 0U;
    g_motion.pending_y_left = 0U;
    g_motion.kind = MOTION_KIND_IDLE;
}

static void motion_timer_prime(uint32_t interval_us)
{
    motion_timer_set_interval_us(interval_us);
    TIM4->CNT = 0U;
    TIM4->EGR = TIM_EGR_UG;
    TIM4->SR = 0U;
    TIM4->DIER |= TIM_DIER_UIE;
    TIM4->CR1 |= TIM_CR1_CEN;
}

static void motion_timer_init(void)
{
    uint32_t prescaler;

    RCC_APB1ENR |= RCC_APB1ENR_TIM4EN;

    TIM4->CR1 = 0U;
    TIM4->CR2 = 0U;
    TIM4->SMCR = 0U;
    TIM4->DIER = 0U;
    TIM4->CCMR1 = 0U;
    TIM4->CCMR2 = 0U;
    TIM4->CCER = 0U;

    prescaler = (CORE_CLOCK_HZ / MOTION_TIMER_HZ);
    if (prescaler == 0U)
    {
        prescaler = 1U;
    }

    TIM4->PSC = prescaler - 1U;
    TIM4->ARR = 1000U - 1U;
    TIM4->CNT = 0U;
    TIM4->EGR = TIM_EGR_UG;
    TIM4->SR = 0U;

    clear_all_step_pins();
    nvic_set_priority(TIM4_IRQ_NUM, 0); /* Highest priority for motor timing */
    nvic_enable_irq(TIM4_IRQ_NUM);
}

void gpio_init(void)
{
    /* Enable GPIOA clock (bit 0) and GPIOB clock (bit 1) */
    RCC_AHB1ENR |= (1U << 0);
    RCC_AHB1ENR |= (1U << 1);

    gpio_pin_output_init(X_STEP_PORT, X_STEP_PIN);
    gpio_pin_output_init(X_DIR_PORT, X_DIR_PIN);

    gpio_pin_output_init(Y_RIGHT_STEP_PORT, Y_RIGHT_STEP_PIN);
    gpio_pin_output_init(Y_LEFT_STEP_PORT, Y_LEFT_STEP_PIN);
    gpio_pin_output_init(Y1_DIR_PORT, Y1_DIR_PIN);
    gpio_pin_output_init(Y2_DIR_PORT, Y2_DIR_PIN);

    gpio_pin_output_init(Z_STEP_PORT, Z_STEP_PIN);
    gpio_pin_output_init(Z_DIR_PORT, Z_DIR_PIN);

    gpio_pin_alt_init(UART_TX_PORT, UART_TX_PIN, UART_AF_INDEX);
    gpio_pin_alt_init(UART_RX_PORT, UART_RX_PIN, UART_AF_INDEX);
}

void uart_init(void)
{
    /* USART2 clock enable on APB1 (bit 17) */
    RCC_APB1ENR |= (1U << 17);

    USART2->CR1 = 0U;
    USART2->CR2 = 0U;
    USART2->CR3 = 0U;

    /* 16x oversampling, BRR = fck / baud */
    USART2->BRR = (CORE_CLOCK_HZ + (UART_BAUDRATE / 2U)) / UART_BAUDRATE;

    /* UE + TE + RE */
    USART2->CR1 = (1U << 13) | (1U << 3) | (1U << 2);

    nvic_set_priority(38U, 1); /* USART2 IRQ priority 1 */
}

static void uart_write_char(char c)
{
    while ((USART2->SR & (1U << 7)) == 0U)
    {
    }
    USART2->DR = (uint32_t)c;
}

static void uart_write_line(const char *s)
{
    while (*s != '\0')
    {
        uart_write_char(*s);
        s++;
    }
    uart_write_char('\n');
}

static char uart_read_char_blocking(void)
{
    while ((USART2->SR & (1U << 5)) == 0U)
    {
    }
    return (char)(USART2->DR & 0xFFU);
}

static bool uart_read_line(char *out, uint32_t out_size)
{
    uint32_t idx = 0U;

    if (out == NULL || out_size < 2U)
    {
        return false;
    }

    while (1)
    {
        char c = uart_read_char_blocking();

        if (c == '\0')
        {
            /* Ignore occasional null bytes from line startup noise. */
            continue;
        }

        if (c == '\r')
        {
            continue;
        }

        if (c == '\n')
        {
            if (idx == 0U)
            {
                continue;
            }
            out[idx] = '\0';
            return true;
        }

        if (idx < (out_size - 1U))
        {
            out[idx++] = c;
        }
    }
}

void delay_cycles(volatile uint32_t count)
{
    while (count--)
    {
        __asm__("nop");
    }
}

void set_x_dir(uint8_t dir)
{
    if (dir)
    {
        gpio_set_pin(X_DIR_PORT, X_DIR_PIN);
    }
    else
    {
        gpio_clear_pin(X_DIR_PORT, X_DIR_PIN);
    }
}

void set_y_dir(uint8_t dir)
{
    /* Y motors rotate in opposite directions */
    if (dir)
    {
        gpio_set_pin(Y1_DIR_PORT, Y1_DIR_PIN);
        gpio_clear_pin(Y2_DIR_PORT, Y2_DIR_PIN);
    }
    else
    {
        gpio_clear_pin(Y1_DIR_PORT, Y1_DIR_PIN);
        gpio_set_pin(Y2_DIR_PORT, Y2_DIR_PIN);
    }
}

static void step_y_side_only(GPIO_TypeDef *step_port, uint8_t step_pin, uint32_t steps, uint8_t dir)
{
    uint32_t pulse_high_us = g_motion_cfg.step_pulse_high_us;
    uint32_t step_delay_us = clamp_step_period_us(g_motion_cfg.y_step_delay_us);
    uint32_t low_delay_us = step_delay_us - pulse_high_us;
    uint32_t low_delay_cycles = ((CORE_CLOCK_HZ / 1000000U) / 4U) * low_delay_us;

    set_y_dir(dir);

    for (uint32_t i = 0; i < steps; ++i)
    {
        pulse_step_pin(step_port, step_pin);
        delay_cycles(low_delay_cycles);
    }
}

void set_z_dir(uint8_t dir)
{
    if (dir)
    {
        gpio_set_pin(Z_DIR_PORT, Z_DIR_PIN);
    }
    else
    {
        gpio_clear_pin(Z_DIR_PORT, Z_DIR_PIN);
    }
}

static uint32_t max_u32(uint32_t a, uint32_t b)
{
    return (a > b) ? a : b;
}

static int32_t scale_logical_y_to_left_steps(int32_t logical_y_steps)
{
    return logical_y_steps;
}

static uint32_t get_xy_cruise_delay(uint32_t x_steps, uint32_t y_steps)
{
    if (x_steps == 0U)
    {
        return g_motion_cfg.y_step_delay_us;
    }

    if (y_steps == 0U)
    {
        return g_motion_cfg.x_step_delay_us;
    }

    return max_u32(g_motion_cfg.x_step_delay_us, g_motion_cfg.y_step_delay_us);
}

static uint32_t get_xy_step_delay(uint32_t step_index, uint32_t total_steps, uint32_t cruise_delay)
{
    uint32_t start_delay = max_u32(g_motion_cfg.xy_start_delay_us, cruise_delay);
    if (start_delay <= cruise_delay || g_motion_cfg.xy_ramp_steps == 0U || total_steps <= 1U)
    {
        return cruise_delay;
    }

    uint32_t ramp_steps = g_motion_cfg.xy_ramp_steps;
    uint32_t half_steps = total_steps / 2U;
    if (half_steps == 0U)
    {
        half_steps = 1U;
    }
    if (ramp_steps > half_steps)
    {
        ramp_steps = half_steps;
    }
    if (ramp_steps == 0U)
    {
        return cruise_delay;
    }

    uint32_t delta = start_delay - cruise_delay;
    if (step_index < ramp_steps)
    {
        return start_delay - ((delta * step_index) / ramp_steps);
    }

    uint32_t remaining_steps = total_steps - step_index - 1U;
    if (remaining_steps < ramp_steps)
    {
        return start_delay - ((delta * remaining_steps) / ramp_steps);
    }

    return cruise_delay;
}

static void wait_for_motion_complete(void)
{
    while (g_motion.active != 0U)
    {
        __asm__ volatile("wfi");
    }
}

static void motion_begin_xy(uint32_t x_steps, uint8_t x_dir, uint32_t y_right_steps, uint32_t y_left_steps, uint8_t y_dir)
{
    uint32_t max_steps = max_u32(max_u32(x_steps, y_right_steps), y_left_steps);

    if (max_steps == 0U)
    {
        return;
    }

    set_x_dir(x_dir);
    set_y_dir(y_dir);

    g_motion.active = 1U;
    g_motion.pending_x = 0U;
    g_motion.pending_y_right = 0U;
    g_motion.pending_y_left = 0U;
    g_motion.kind = MOTION_KIND_XY;
    g_motion.step_index = 0U;
    g_motion.total_steps = max_steps;
    g_motion.x_steps = x_steps;
    g_motion.y_right_steps = y_right_steps;
    g_motion.y_left_steps = y_left_steps;
    g_motion.x_acc = 0U;
    g_motion.y_right_acc = 0U;
    g_motion.y_left_acc = 0U;
    g_motion.cruise_delay_us = get_xy_cruise_delay(x_steps, y_right_steps);
    g_motion.z_delay_us = 0U;

    motion_timer_prime(get_xy_step_delay(0U, max_steps, g_motion.cruise_delay_us));
}

static void motion_begin_z(uint32_t steps, uint8_t dir)
{
    if (steps == 0U)
    {
        return;
    }

    set_z_dir(dir);

    g_motion.active = 1U;
    g_motion.pending_x = 0U;
    g_motion.pending_y_right = 0U;
    g_motion.pending_y_left = 0U;
    g_motion.kind = MOTION_KIND_Z;
    g_motion.step_index = 0U;
    g_motion.total_steps = steps;
    g_motion.x_steps = 0U;
    g_motion.y_right_steps = 0U;
    g_motion.y_left_steps = 0U;
    g_motion.x_acc = 0U;
    g_motion.y_right_acc = 0U;
    g_motion.y_left_acc = 0U;
    g_motion.cruise_delay_us = 0U;
    g_motion.z_delay_us = g_motion_cfg.z_step_delay_us;

    motion_timer_prime(g_motion.z_delay_us);
}

static void motion_handle_xy_irq(void)
{
    g_motion.pending_x = 0U;
    g_motion.pending_y_right = 0U;
    g_motion.pending_y_left = 0U;

    g_motion.x_acc += g_motion.x_steps;
    g_motion.y_right_acc += g_motion.y_right_steps;
    g_motion.y_left_acc += g_motion.y_left_steps;

    if (g_motion.x_acc >= g_motion.total_steps)
    {
        g_motion.x_acc -= g_motion.total_steps;
        g_motion.pending_x = 1U;
        gpio_set_pin(X_STEP_PORT, X_STEP_PIN);
    }

    if (g_motion.y_right_acc >= g_motion.total_steps)
    {
        g_motion.y_right_acc -= g_motion.total_steps;
        g_motion.pending_y_right = 1U;
        gpio_set_pin(Y_RIGHT_STEP_PORT, Y_RIGHT_STEP_PIN);
    }

    if (g_motion.y_left_acc >= g_motion.total_steps)
    {
        g_motion.y_left_acc -= g_motion.total_steps;
        g_motion.pending_y_left = 1U;
        gpio_set_pin(Y_LEFT_STEP_PORT, Y_LEFT_STEP_PIN);
    }

    delay_cycles(pulse_high_delay_count());

    if (g_motion.pending_x != 0U)
    {
        gpio_clear_pin(X_STEP_PORT, X_STEP_PIN);
    }

    if (g_motion.pending_y_right != 0U)
    {
        gpio_clear_pin(Y_RIGHT_STEP_PORT, Y_RIGHT_STEP_PIN);
    }

    if (g_motion.pending_y_left != 0U)
    {
        gpio_clear_pin(Y_LEFT_STEP_PORT, Y_LEFT_STEP_PIN);
    }

    g_motion.step_index++;
    if (g_motion.step_index >= g_motion.total_steps)
    {
        motion_timer_stop();
        return;
    }

    motion_timer_set_interval_us(get_xy_step_delay(g_motion.step_index, g_motion.total_steps, g_motion.cruise_delay_us));
}

static void motion_handle_z_irq(void)
{
    gpio_set_pin(Z_STEP_PORT, Z_STEP_PIN);
    delay_cycles(pulse_high_delay_count());
    gpio_clear_pin(Z_STEP_PORT, Z_STEP_PIN);

    g_motion.step_index++;
    if (g_motion.step_index >= g_motion.total_steps)
    {
        motion_timer_stop();
        return;
    }

    motion_timer_set_interval_us(g_motion.z_delay_us);
}

void TIM4_IRQHandler(void)
{
    if ((TIM4->SR & TIM_SR_UIF) == 0U)
    {
        return;
    }

    TIM4->SR &= ~TIM_SR_UIF;

    if (g_motion.active == 0U)
    {
        return;
    }

    switch (g_motion.kind)
    {
        case MOTION_KIND_XY:
            motion_handle_xy_irq();
            break;
        case MOTION_KIND_Z:
            motion_handle_z_irq();
            break;
        default:
            motion_timer_stop();
            break;
    }
}

void step_x(uint32_t steps, uint8_t dir)
{
    motion_begin_xy(steps, dir, 0U, 0U, 0U);
    wait_for_motion_complete();
}

void step_y(uint32_t steps, uint8_t dir)
{
    uint32_t left_steps = (uint32_t)scale_logical_y_to_left_steps((int32_t)steps);
    motion_begin_xy(0U, 0U, steps, left_steps, dir);
    wait_for_motion_complete();
}

void step_z(uint32_t steps, uint8_t dir)
{
    motion_begin_z(steps, dir);
    wait_for_motion_complete();

    if (dir != 0U)
    {
        g_current_z_steps += (int32_t)steps;
    }
    else
    {
        g_current_z_steps -= (int32_t)steps;
    }
}

void move_xy(uint32_t x_steps, uint8_t x_dir, uint32_t y_steps, uint8_t y_dir)
{
    uint32_t left_steps = (uint32_t)scale_logical_y_to_left_steps((int32_t)y_steps);
    motion_begin_xy(x_steps, x_dir, y_steps, left_steps, y_dir);
    wait_for_motion_complete();
}

static void move_to_steps_unladen(int32_t target_x_steps, int32_t target_y_steps)
{
    move_to_steps_common(target_x_steps, target_y_steps);
}

void move_to_steps(int32_t target_x_steps, int32_t target_y_steps)
{
    move_to_steps_common(target_x_steps, target_y_steps);
}

bool board_coord_to_steps(int32_t board_x, int32_t board_y, int32_t *out_x_steps, int32_t *out_y_steps)
{
    if (board_x < 0 || board_x > BOARD_GRID_MAX_INDEX || board_y < 0 || board_y > BOARD_GRID_MAX_INDEX)
    {
        return false;
    }

    if (out_x_steps == NULL || out_y_steps == NULL)
    {
        return false;
    }

    {
        float u = (float)board_x / (float)BOARD_GRID_MAX_INDEX;
        float v = (float)board_y / (float)BOARD_GRID_MAX_INDEX;

        float x_interp =
            (1.0f - u) * (1.0f - v) * (float)CORNER_00_X_STEPS +
            u * (1.0f - v) * (float)CORNER_70_X_STEPS +
            (1.0f - u) * v * (float)CORNER_07_X_STEPS +
            u * v * (float)CORNER_77_X_STEPS;

        float y_interp =
            (1.0f - u) * (1.0f - v) * (float)CORNER_00_Y_STEPS +
            u * (1.0f - v) * (float)CORNER_70_Y_STEPS +
            (1.0f - u) * v * (float)CORNER_07_Y_STEPS +
            u * v * (float)CORNER_77_Y_STEPS;

        *out_x_steps = (int32_t)(x_interp + ((x_interp >= 0.0f) ? 0.5f : -0.5f));
        *out_y_steps = (int32_t)(y_interp + ((y_interp >= 0.0f) ? 0.5f : -0.5f));
    }

    return true;
}

static bool workspace_steps_in_range(int32_t x_steps, int32_t y_steps)
{
    if (x_steps < WORKSPACE_MIN_X_STEPS || x_steps > WORKSPACE_MAX_X_STEPS)
    {
        return false;
    }
    if (y_steps < WORKSPACE_MIN_Y_STEPS || y_steps > WORKSPACE_MAX_Y_STEPS)
    {
        return false;
    }
    return true;
}

static bool workspace_percent_to_steps(int32_t pct_x, int32_t pct_y, int32_t *out_x_steps, int32_t *out_y_steps)
{
    int64_t x_range;
    int64_t y_range;
    int64_t x_scaled;
    int64_t y_scaled;

    if (out_x_steps == NULL || out_y_steps == NULL)
    {
        return false;
    }

    if (pct_x < 0 || pct_x > WORKSPACE_PERCENT_SCALE || pct_y < 0 || pct_y > WORKSPACE_PERCENT_SCALE)
    {
        return false;
    }

    x_range = (int64_t)WORKSPACE_MAX_X_STEPS - (int64_t)WORKSPACE_MIN_X_STEPS;
    y_range = (int64_t)WORKSPACE_MAX_Y_STEPS - (int64_t)WORKSPACE_MIN_Y_STEPS;

    x_scaled = x_range * (int64_t)pct_x;
    y_scaled = y_range * (int64_t)pct_y;

    *out_x_steps = WORKSPACE_MIN_X_STEPS +
                   (int32_t)((x_scaled + (WORKSPACE_PERCENT_SCALE / 2)) / WORKSPACE_PERCENT_SCALE);
    *out_y_steps = WORKSPACE_MIN_Y_STEPS +
                   (int32_t)((y_scaled + (WORKSPACE_PERCENT_SCALE / 2)) / WORKSPACE_PERCENT_SCALE);

    return workspace_steps_in_range(*out_x_steps, *out_y_steps);
}

static void send_status(void)
{
    char line[128];
    (void)snprintf(line, sizeof(line), "STATUS cur_x=%ld cur_y=%ld cur_z=%ld", (long)g_current_x_steps,
                   (long)g_current_y_steps, (long)g_current_z_steps);
    uart_write_line(line);
}

static void z_pickup(void)
{
    step_z(Z_PICKUP_STEPS, Z_PICKUP_DIR);
}

static void z_pickup_pulse(uint32_t steps)
{
    if (steps == 0U)
    {
        return;
    }
    motion_begin_z(steps, Z_PICKUP_DIR);
    wait_for_motion_complete();
}

static void z_release(void)
{
    step_z(Z_RELEASE_STEPS, Z_RELEASE_DIR);
}

static void reset_moveheld_direction_tracking(void)
{
    g_prev_moveheld_dx_steps = 0;
    g_prev_moveheld_dy_steps = 0;
    g_has_prev_moveheld_direction = 0U;
}

static uint8_t moveheld_direction_changed(int32_t dx_steps, int32_t dy_steps)
{
    int64_t cross;
    int64_t dot;

    if (g_has_prev_moveheld_direction == 0U)
    {
        return 0U;
    }

    cross = ((int64_t)g_prev_moveheld_dx_steps * (int64_t)dy_steps) -
            ((int64_t)g_prev_moveheld_dy_steps * (int64_t)dx_steps);
    dot = ((int64_t)g_prev_moveheld_dx_steps * (int64_t)dx_steps) +
          ((int64_t)g_prev_moveheld_dy_steps * (int64_t)dy_steps);

    if ((cross == 0) && (dot > 0))
    {
        return 0U;
    }

    return 1U;
}

static void reset_motion_runtime_config(void)
{
    g_motion_cfg.x_step_delay_us = X_STEP_DELAY_CYCLES;
    g_motion_cfg.y_step_delay_us = Y_STEP_DELAY_CYCLES;
    g_motion_cfg.xy_start_delay_us = XY_START_DELAY_CYCLES;
    g_motion_cfg.xy_ramp_steps = XY_RAMP_STEPS;
    g_motion_cfg.z_step_delay_us = STEP_DELAY_CYCLES_Z;
    g_motion_cfg.step_pulse_high_us = STEP_PULSE_HIGH_US;
    g_motion_cfg.z_pre_pickup_pause_cycles = Z_PRE_PICKUP_PAUSE_CYCLES;
    g_motion_cfg.moveheld_settle_cycles = MOVEHELD_SETTLE_CYCLES;
    g_motion_cfg.z_post_release_pause_cycles = Z_POST_RELEASE_PAUSE_CYCLES;
    g_motion_cfg.return_start_seat_steps = RETURN_START_SEAT_STEPS;
}

static void send_motion_config(void)
{
    char line[256];
    (void)snprintf(
        line,
        sizeof(line),
        "MOTIONCFG x_delay=%lu y_delay=%lu xy_start=%lu xy_ramp=%lu z_delay=%lu pulse_high=%lu pre_pickup=%lu moveheld_settle=%lu post_release=%lu return_seat=%lu",
        (unsigned long)g_motion_cfg.x_step_delay_us,
        (unsigned long)g_motion_cfg.y_step_delay_us,
        (unsigned long)g_motion_cfg.xy_start_delay_us,
        (unsigned long)g_motion_cfg.xy_ramp_steps,
        (unsigned long)g_motion_cfg.z_step_delay_us,
        (unsigned long)g_motion_cfg.step_pulse_high_us,
        (unsigned long)g_motion_cfg.z_pre_pickup_pause_cycles,
        (unsigned long)g_motion_cfg.moveheld_settle_cycles,
        (unsigned long)g_motion_cfg.z_post_release_pause_cycles,
        (unsigned long)g_motion_cfg.return_start_seat_steps
    );
    uart_write_line(line);
}

static uint8_t motion_cfg_valid_delay(uint32_t delay_us)
{
    return (delay_us > 0U) ? 1U : 0U;
}

static void move_to_steps_common(int32_t target_x_steps, int32_t target_y_steps)
{
    int32_t dx = target_x_steps - g_current_x_steps;
    int32_t target_y_left_steps = scale_logical_y_to_left_steps(target_y_steps);
    int32_t dy = target_y_steps - g_current_y_steps;
    int32_t dy_left = target_y_left_steps - g_current_y_left_steps;

    uint8_t x_dir = (dx >= 0) ? 1U : 0U;
    uint8_t y_dir = (dy >= 0) ? 1U : 0U;
    uint32_t x_steps = (dx >= 0) ? (uint32_t)dx : (uint32_t)(-dx);
    uint32_t y_steps = (dy >= 0) ? (uint32_t)dy : (uint32_t)(-dy);
    uint32_t y_left_steps = (dy_left >= 0) ? (uint32_t)dy_left : (uint32_t)(-dy_left);

    motion_begin_xy(x_steps, x_dir, y_steps, y_left_steps, y_dir);
    wait_for_motion_complete();

    g_current_x_steps = target_x_steps;
    g_current_y_steps = target_y_steps;
    g_current_y_left_steps = target_y_left_steps;
}

static void execute_pickup_steps(int32_t src_x_steps, int32_t src_y_steps)
{
    reset_moveheld_direction_tracking();
    move_to_steps_unladen(src_x_steps, src_y_steps);
    delay_cycles(g_motion_cfg.z_pre_pickup_pause_cycles);
    z_pickup();
}

static void execute_moveheld_steps(int32_t dst_x_steps, int32_t dst_y_steps)
{
    int32_t dx_steps = dst_x_steps - g_current_x_steps;
    int32_t dy_steps = dst_y_steps - g_current_y_steps;

    move_to_steps(dst_x_steps, dst_y_steps);
    if (moveheld_direction_changed(dx_steps, dy_steps) != 0U)
    {
        delay_cycles(g_motion_cfg.moveheld_settle_cycles);
    }
    if (MOVEHELD_REGRIP_STEPS > 0U)
    {
        z_pickup_pulse(MOVEHELD_REGRIP_STEPS);
    }
    g_prev_moveheld_dx_steps = dx_steps;
    g_prev_moveheld_dy_steps = dy_steps;
    g_has_prev_moveheld_direction = 1U;
}

static void execute_release_steps(int32_t dst_x_steps, int32_t dst_y_steps)
{
    move_to_steps(dst_x_steps, dst_y_steps);
    z_release();
    reset_moveheld_direction_tracking();
    delay_cycles(g_motion_cfg.z_post_release_pause_cycles);
}

static void execute_pick_and_place_steps(int32_t src_x_steps, int32_t src_y_steps, int32_t dst_x_steps,
                                         int32_t dst_y_steps)
{
    execute_pickup_steps(src_x_steps, src_y_steps);
    execute_release_steps(dst_x_steps, dst_y_steps);
}

static void process_command(const char *cmd)
{
    char sanitized[96];
    size_t len;
    const char *start;
    int gx;
    int gy;
    int sx;
    int sy;
    int dx;
    int dy;
    int spx;
    int spy;
    int dpx;
    int dpy;
    int32_t goto_x_steps;
    int32_t goto_y_steps;
    int32_t src_x_steps;
    int32_t src_y_steps;
    int32_t dst_x_steps;
    int32_t dst_y_steps;
    unsigned long ux0;
    unsigned long ux1;
    unsigned long ux2;
    unsigned long ux3;

    if (cmd == NULL)
    {
        uart_write_line("ERR CMD");
        return;
    }

    start = cmd;
    while (*start != '\0' && isspace((unsigned char)*start))
    {
        start++;
    }

    (void)strncpy(sanitized, start, sizeof(sanitized) - 1U);
    sanitized[sizeof(sanitized) - 1U] = '\0';

    len = strlen(sanitized);
    while (len > 0U && isspace((unsigned char)sanitized[len - 1U]))
    {
        sanitized[len - 1U] = '\0';
        len--;
    }

    if (strcmp(sanitized, "PING") == 0)
    {
        uart_write_line("PONG");
        return;
    }

    if (strcmp(sanitized, "ZERO") == 0)
    {
        g_current_x_steps = 0;
        g_current_y_steps = 0;
        g_current_y_left_steps = 0;
        g_current_z_steps = 0;
        reset_moveheld_direction_tracking();
        uart_write_line("OK ZERO");
        return;
    }

    if (strcmp(sanitized, "STATUS") == 0)
    {
        send_status();
        return;
    }

    if (strcmp(sanitized, "MOTIONCFG") == 0)
    {
        send_motion_config();
        return;
    }

    if (strcmp(sanitized, "RESET_MOTIONCFG") == 0)
    {
        reset_motion_runtime_config();
        uart_write_line("OK RESET_MOTIONCFG");
        return;
    }

    if (sscanf(sanitized, "SET_XY_MOTION %lu %lu %lu %lu", &ux0, &ux1, &ux2, &ux3) == 4)
    {
        if (motion_cfg_valid_delay((uint32_t)ux0) == 0U || motion_cfg_valid_delay((uint32_t)ux1) == 0U || motion_cfg_valid_delay((uint32_t)ux2) == 0U)
        {
            uart_write_line("ERR MOTIONCFG_RANGE");
            return;
        }
        g_motion_cfg.x_step_delay_us = (uint32_t)ux0;
        g_motion_cfg.y_step_delay_us = (uint32_t)ux1;
        g_motion_cfg.xy_start_delay_us = (uint32_t)ux2;
        g_motion_cfg.xy_ramp_steps = (uint32_t)ux3;
        uart_write_line("OK SET_XY_MOTION");
        return;
    }

    if (sscanf(sanitized, "SET_Z_MOTION %lu %lu", &ux0, &ux1) == 2)
    {
        if (motion_cfg_valid_delay((uint32_t)ux0) == 0U || motion_cfg_valid_delay((uint32_t)ux1) == 0U)
        {
            uart_write_line("ERR MOTIONCFG_RANGE");
            return;
        }
        g_motion_cfg.z_step_delay_us = (uint32_t)ux0;
        g_motion_cfg.step_pulse_high_us = (uint32_t)ux1;
        uart_write_line("OK SET_Z_MOTION");
        return;
    }

    if (sscanf(sanitized, "SET_SETTLE %lu %lu %lu %lu", &ux0, &ux1, &ux2, &ux3) == 4)
    {
        g_motion_cfg.z_pre_pickup_pause_cycles = (uint32_t)ux0;
        g_motion_cfg.moveheld_settle_cycles = (uint32_t)ux1;
        g_motion_cfg.z_post_release_pause_cycles = (uint32_t)ux2;
        g_motion_cfg.return_start_seat_steps = (uint32_t)ux3;
        uart_write_line("OK SET_SETTLE");
        return;
    }

    if (strcmp(sanitized, "RETURN_START") == 0)
    {
        move_to_steps_unladen(0, 0);
        if (g_motion_cfg.return_start_seat_steps > 0U)
        {
            move_xy(g_motion_cfg.return_start_seat_steps, 0U, g_motion_cfg.return_start_seat_steps, 0U);
            g_current_x_steps = 0;
            g_current_y_steps = 0;
            g_current_y_left_steps = 0;
        }
        reset_moveheld_direction_tracking();
        uart_write_line("OK RETURN_START");
        return;
    }

    if (sscanf(sanitized, "GOTO_STEPS %d %d", &gx, &gy) == 2)
    {
        move_to_steps_unladen((int32_t)gx, (int32_t)gy);
        uart_write_line("OK GOTO_STEPS");
        return;
    }

    if (sscanf(sanitized, "JOG_STEPS %d %d", &gx, &gy) == 2)
    {
        int32_t target_x = g_current_x_steps + (int32_t)gx;
        int32_t target_y = g_current_y_steps + (int32_t)gy;

        move_to_steps_unladen(target_x, target_y);
        uart_write_line("OK JOG_STEPS");
        return;
    }

    if (sscanf(sanitized, "JOG_Y_RIGHT %d", &gx) == 1)
    {
        int32_t delta = (int32_t)gx;
        if (delta >= 0)
        {
            step_y_side_only(Y_RIGHT_STEP_PORT, Y_RIGHT_STEP_PIN, (uint32_t)delta, 1U);
        }
        else
        {
            step_y_side_only(Y_RIGHT_STEP_PORT, Y_RIGHT_STEP_PIN, (uint32_t)(-delta), 0U);
        }

        uart_write_line("OK JOG_Y_RIGHT");
        return;
    }

    if (sscanf(sanitized, "JOG_Y_LEFT %d", &gx) == 1)
    {
        int32_t delta = (int32_t)gx;
        if (delta >= 0)
        {
            step_y_side_only(Y_LEFT_STEP_PORT, Y_LEFT_STEP_PIN, (uint32_t)delta, 1U);
        }
        else
        {
            step_y_side_only(Y_LEFT_STEP_PORT, Y_LEFT_STEP_PIN, (uint32_t)(-delta), 0U);
        }

        uart_write_line("OK JOG_Y_LEFT");
        return;
    }

    if (sscanf(sanitized, "JOG_Z %d", &gx) == 1)
    {
        int32_t delta_z = (int32_t)gx;

        if (delta_z >= 0)
        {
            step_z((uint32_t)delta_z, 1U);
        }
        else
        {
            step_z((uint32_t)(-delta_z), 0U);
        }

        uart_write_line("OK JOG_Z");
        return;
    }

    if (sscanf(sanitized, "GOTOPCT %d %d", &gx, &gy) == 2)
    {
        if (!workspace_percent_to_steps((int32_t)gx, (int32_t)gy, &goto_x_steps, &goto_y_steps))
        {
            uart_write_line("ERR PCT_RANGE");
            return;
        }

        move_to_steps_unladen(goto_x_steps, goto_y_steps);
        uart_write_line("OK GOTOPCT");
        return;
    }

    if (sscanf(sanitized, "GOTO %d %d", &gx, &gy) == 2)
    {
        if (!board_coord_to_steps(gx, gy, &goto_x_steps, &goto_y_steps))
        {
            uart_write_line("ERR GOTO_RANGE");
            return;
        }

        move_to_steps_unladen(goto_x_steps, goto_y_steps);
        uart_write_line("OK GOTO");
        return;
    }

    if (sscanf(sanitized, "MOVE_STEPS %d %d %d %d", &sx, &sy, &dx, &dy) == 4)
    {
        src_x_steps = (int32_t)sx;
        src_y_steps = (int32_t)sy;
        dst_x_steps = (int32_t)dx;
        dst_y_steps = (int32_t)dy;

        if (!workspace_steps_in_range(src_x_steps, src_y_steps))
        {
            uart_write_line("ERR SOURCE_STEP_RANGE");
            return;
        }
        if (!workspace_steps_in_range(dst_x_steps, dst_y_steps))
        {
            uart_write_line("ERR DEST_STEP_RANGE");
            return;
        }

        execute_pick_and_place_steps(src_x_steps, src_y_steps, dst_x_steps, dst_y_steps);
        uart_write_line("OK MOVE_STEPS");
        return;
    }

    if (sscanf(sanitized, "PICKUP_STEPS %d %d", &sx, &sy) == 2)
    {
        src_x_steps = (int32_t)sx;
        src_y_steps = (int32_t)sy;

        if (!workspace_steps_in_range(src_x_steps, src_y_steps))
        {
            uart_write_line("ERR SOURCE_STEP_RANGE");
            return;
        }

        execute_pickup_steps(src_x_steps, src_y_steps);
        uart_write_line("OK PICKUP_STEPS");
        return;
    }

    if (sscanf(sanitized, "MOVEHELD_STEPS %d %d", &dx, &dy) == 2)
    {
        dst_x_steps = (int32_t)dx;
        dst_y_steps = (int32_t)dy;

        if (!workspace_steps_in_range(dst_x_steps, dst_y_steps))
        {
            uart_write_line("ERR DEST_STEP_RANGE");
            return;
        }

        execute_moveheld_steps(dst_x_steps, dst_y_steps);
        uart_write_line("OK MOVEHELD_STEPS");
        return;
    }

    if (sscanf(sanitized, "RELEASE_STEPS %d %d", &dx, &dy) == 2)
    {
        dst_x_steps = (int32_t)dx;
        dst_y_steps = (int32_t)dy;

        if (!workspace_steps_in_range(dst_x_steps, dst_y_steps))
        {
            uart_write_line("ERR DEST_STEP_RANGE");
            return;
        }

        execute_release_steps(dst_x_steps, dst_y_steps);
        uart_write_line("OK RELEASE_STEPS");
        return;
    }

    if (sscanf(sanitized, "MOVEPCT %d %d %d %d", &spx, &spy, &dpx, &dpy) == 4)
    {
        if (!workspace_percent_to_steps((int32_t)spx, (int32_t)spy, &src_x_steps, &src_y_steps))
        {
            uart_write_line("ERR SOURCE_PCT_RANGE");
            return;
        }
        if (!workspace_percent_to_steps((int32_t)dpx, (int32_t)dpy, &dst_x_steps, &dst_y_steps))
        {
            uart_write_line("ERR DEST_PCT_RANGE");
            return;
        }

        execute_pick_and_place_steps(src_x_steps, src_y_steps, dst_x_steps, dst_y_steps);
        uart_write_line("OK MOVEPCT");
        return;
    }

    if (sscanf(sanitized, "MOVE %d %d %d %d", &sx, &sy, &dx, &dy) == 4)
    {
        if (!board_coord_to_steps(sx, sy, &src_x_steps, &src_y_steps))
        {
            uart_write_line("ERR SOURCE_RANGE");
            return;
        }
        if (!board_coord_to_steps(dx, dy, &dst_x_steps, &dst_y_steps))
        {
            uart_write_line("ERR DEST_RANGE");
            return;
        }

        execute_pick_and_place_steps(src_x_steps, src_y_steps, dst_x_steps, dst_y_steps);
        uart_write_line("OK MOVE");
        return;
    }

    uart_write_line("ERR CMD");
}

int main(void)
{
    char cmd_line[96];

    gpio_init();
    motion_timer_init();
    uart_init();
    __asm__ volatile("cpsie i");

    uart_write_line("READY");

    while (1)
    {
        if (uart_read_line(cmd_line, sizeof(cmd_line)))
        {
            process_command(cmd_line);
        }
    }
}
