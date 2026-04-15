#include "main.h"

#include <ctype.h>
#include <stdio.h>
#include <string.h>

static int32_t g_current_x_steps = 0;
static int32_t g_current_y_steps = 0;

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

void gpio_init(void)
{
    /* Enable GPIOA clock (bit 0) and GPIOB clock (bit 1) */
    RCC_AHB1ENR |= (1U << 0);
    RCC_AHB1ENR |= (1U << 1);

    gpio_pin_output_init(X_STEP_PORT, X_STEP_PIN);
    gpio_pin_output_init(X_DIR_PORT, X_DIR_PIN);

    gpio_pin_output_init(Y_STEP_PORT, Y_STEP_PIN);
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

void pulse_x_step(void)
{
    gpio_set_pin(X_STEP_PORT, X_STEP_PIN);
    delay_cycles(STEP_DELAY_CYCLES);

    gpio_clear_pin(X_STEP_PORT, X_STEP_PIN);
    delay_cycles(STEP_DELAY_CYCLES);
}

void pulse_y_step(void)
{
    /* Shared STEP signal for both Y drivers */
    gpio_set_pin(Y_STEP_PORT, Y_STEP_PIN);
    delay_cycles(STEP_DELAY_CYCLES);

    gpio_clear_pin(Y_STEP_PORT, Y_STEP_PIN);
    delay_cycles(STEP_DELAY_CYCLES);
}

void pulse_z_step(void)
{
    gpio_set_pin(Z_STEP_PORT, Z_STEP_PIN);
    delay_cycles(STEP_DELAY_CYCLES_Z);

    gpio_clear_pin(Z_STEP_PORT, Z_STEP_PIN);
    delay_cycles(STEP_DELAY_CYCLES_Z);
}

void step_x(uint32_t steps, uint8_t dir)
{
    set_x_dir(dir);
    delay_cycles(STEP_DELAY_CYCLES);

    for (uint32_t i = 0; i < steps; i++)
    {
        pulse_x_step();
    }
}

void step_y(uint32_t steps, uint8_t dir)
{
    set_y_dir(dir);
    delay_cycles(STEP_DELAY_CYCLES);

    for (uint32_t i = 0; i < steps; i++)
    {
        pulse_y_step();
    }
}

void step_z(uint32_t steps, uint8_t dir)
{
    set_z_dir(dir);
    delay_cycles(STEP_DELAY_CYCLES_Z);

    for (uint32_t i = 0; i < steps; i++)
    {
        pulse_z_step();
    }
}

void move_xy(uint32_t x_steps, uint8_t x_dir, uint32_t y_steps, uint8_t y_dir)
{
    uint32_t i;
    uint32_t max_steps;
    uint32_t x_acc = 0U;
    uint32_t y_acc = 0U;

    set_x_dir(x_dir);
    set_y_dir(y_dir);
    delay_cycles(STEP_DELAY_CYCLES);

    max_steps = (x_steps > y_steps) ? x_steps : y_steps;

    if (max_steps == 0U)
    {
        return;
    }

    for (i = 0U; i < max_steps; i++)
    {
        uint8_t do_x = 0U;
        uint8_t do_y = 0U;

        x_acc += x_steps;
        y_acc += y_steps;

        if (x_acc >= max_steps)
        {
            x_acc -= max_steps;
            do_x = 1U;
        }

        if (y_acc >= max_steps)
        {
            y_acc -= max_steps;
            do_y = 1U;
        }

        if (do_x)
        {
            gpio_set_pin(X_STEP_PORT, X_STEP_PIN);
        }

        if (do_y)
        {
            gpio_set_pin(Y_STEP_PORT, Y_STEP_PIN);
        }

        delay_cycles(STEP_DELAY_CYCLES);

        if (do_x)
        {
            gpio_clear_pin(X_STEP_PORT, X_STEP_PIN);
        }

        if (do_y)
        {
            gpio_clear_pin(Y_STEP_PORT, Y_STEP_PIN);
        }

        delay_cycles(STEP_DELAY_CYCLES);
    }
}

void move_to_steps(int32_t target_x_steps, int32_t target_y_steps)
{
    int32_t dx = target_x_steps - g_current_x_steps;
    int32_t dy = target_y_steps - g_current_y_steps;

    uint8_t x_dir = (dx >= 0) ? 1U : 0U;
    uint8_t y_dir = (dy >= 0) ? 1U : 0U;
    uint32_t x_steps = (dx >= 0) ? (uint32_t)dx : (uint32_t)(-dx);
    uint32_t y_steps = (dy >= 0) ? (uint32_t)dy : (uint32_t)(-dy);

    move_xy(x_steps, x_dir, y_steps, y_dir);

    g_current_x_steps = target_x_steps;
    g_current_y_steps = target_y_steps;
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

static void send_status(void)
{
    char line[96];
    (void)snprintf(line, sizeof(line), "STATUS cur_x=%ld cur_y=%ld", (long)g_current_x_steps,
                   (long)g_current_y_steps);
    uart_write_line(line);
}

static void z_pickup(void)
{
    step_z(Z_PICKUP_STEPS, Z_PICKUP_DIR);
}

static void z_release(void)
{
    step_z(Z_RELEASE_STEPS, Z_RELEASE_DIR);
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
    int32_t goto_x_steps;
    int32_t goto_y_steps;
    int32_t src_x_steps;
    int32_t src_y_steps;
    int32_t dst_x_steps;
    int32_t dst_y_steps;

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
        uart_write_line("OK ZERO");
        return;
    }

    if (strcmp(sanitized, "STATUS") == 0)
    {
        send_status();
        return;
    }

    if (strcmp(sanitized, "RETURN_START") == 0)
    {
        move_to_steps(0, 0);
        uart_write_line("OK RETURN_START");
        return;
    }

    if (sscanf(sanitized, "GOTO %d %d", &gx, &gy) == 2)
    {
        if (!board_coord_to_steps(gx, gy, &goto_x_steps, &goto_y_steps))
        {
            uart_write_line("ERR GOTO_RANGE");
            return;
        }

        move_to_steps(goto_x_steps, goto_y_steps);
        uart_write_line("OK GOTO");
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

        move_to_steps(src_x_steps, src_y_steps);
        delay_cycles(MOVE_PAUSE_CYCLES);
        z_pickup();
        delay_cycles(MOVE_PAUSE_CYCLES);
        move_to_steps(dst_x_steps, dst_y_steps);
        delay_cycles(MOVE_PAUSE_CYCLES);
        z_release();
        uart_write_line("OK MOVE");
        return;
    }

    uart_write_line("ERR CMD");
}

int main(void)
{
    char cmd_line[96];

    gpio_init();
    uart_init();

    uart_write_line("READY");

    while (1)
    {
        if (uart_read_line(cmd_line, sizeof(cmd_line)))
        {
            process_command(cmd_line);
        }
    }
}
