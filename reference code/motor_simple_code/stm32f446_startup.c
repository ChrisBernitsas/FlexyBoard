#include <stdint.h>

/* Linker-provided symbols */
extern uint32_t _estack;
extern uint32_t _sidata;
extern uint32_t _sdata;
extern uint32_t _edata;
extern uint32_t _sbss;
extern uint32_t _ebss;

int main(void);

void Reset_Handler(void);
void Default_Handler(void);

void NMI_Handler(void) __attribute__((weak, alias("Default_Handler")));
void HardFault_Handler(void) __attribute__((weak, alias("Default_Handler")));
void MemManage_Handler(void) __attribute__((weak, alias("Default_Handler")));
void BusFault_Handler(void) __attribute__((weak, alias("Default_Handler")));
void UsageFault_Handler(void) __attribute__((weak, alias("Default_Handler")));
void SVC_Handler(void) __attribute__((weak, alias("Default_Handler")));
void DebugMon_Handler(void) __attribute__((weak, alias("Default_Handler")));
void PendSV_Handler(void) __attribute__((weak, alias("Default_Handler")));
void SysTick_Handler(void) __attribute__((weak, alias("Default_Handler")));

void WWDG_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void PVD_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void TAMP_STAMP_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void RTC_WKUP_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void FLASH_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void RCC_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void EXTI0_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void EXTI1_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void EXTI2_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void EXTI3_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void EXTI4_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void DMA1_Stream0_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void DMA1_Stream1_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void DMA1_Stream2_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void DMA1_Stream3_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void DMA1_Stream4_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void DMA1_Stream5_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void DMA1_Stream6_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void ADC_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void CAN1_TX_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void CAN1_RX0_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void CAN1_RX1_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void CAN1_SCE_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void EXTI9_5_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void TIM1_BRK_TIM9_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void TIM1_UP_TIM10_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void TIM1_TRG_COM_TIM11_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void TIM1_CC_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void TIM2_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void TIM3_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void TIM4_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void I2C1_EV_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void I2C1_ER_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void I2C2_EV_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void I2C2_ER_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void SPI1_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void SPI2_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void USART1_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void USART2_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void USART3_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void EXTI15_10_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void RTC_Alarm_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void OTG_FS_WKUP_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void DMA1_Stream7_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void SDIO_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void TIM5_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void SPI3_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void DMA2_Stream0_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void DMA2_Stream1_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void DMA2_Stream2_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void DMA2_Stream3_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void DMA2_Stream4_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void OTG_FS_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void DMA2_Stream5_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void DMA2_Stream6_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void DMA2_Stream7_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void USART6_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void I2C3_EV_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void I2C3_ER_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void FPU_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void SPI4_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void SAI1_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void SAI2_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void QUADSPI_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void HDMI_CEC_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void SPDIF_RX_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void FMPI2C1_EV_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));
void FMPI2C1_ER_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));

__attribute__((section(".isr_vector")))
const uintptr_t g_pfn_vectors[] = {
    (uintptr_t)(&_estack), /* Initial stack pointer */
    (uintptr_t)Reset_Handler,
    (uintptr_t)NMI_Handler,
    (uintptr_t)HardFault_Handler,
    (uintptr_t)MemManage_Handler,
    (uintptr_t)BusFault_Handler,
    (uintptr_t)UsageFault_Handler,
    0,
    0,
    0,
    0,
    (uintptr_t)SVC_Handler,
    (uintptr_t)DebugMon_Handler,
    0,
    (uintptr_t)PendSV_Handler,
    (uintptr_t)SysTick_Handler,
    (uintptr_t)WWDG_IRQHandler,
    (uintptr_t)PVD_IRQHandler,
    (uintptr_t)TAMP_STAMP_IRQHandler,
    (uintptr_t)RTC_WKUP_IRQHandler,
    (uintptr_t)FLASH_IRQHandler,
    (uintptr_t)RCC_IRQHandler,
    (uintptr_t)EXTI0_IRQHandler,
    (uintptr_t)EXTI1_IRQHandler,
    (uintptr_t)EXTI2_IRQHandler,
    (uintptr_t)EXTI3_IRQHandler,
    (uintptr_t)EXTI4_IRQHandler,
    (uintptr_t)DMA1_Stream0_IRQHandler,
    (uintptr_t)DMA1_Stream1_IRQHandler,
    (uintptr_t)DMA1_Stream2_IRQHandler,
    (uintptr_t)DMA1_Stream3_IRQHandler,
    (uintptr_t)DMA1_Stream4_IRQHandler,
    (uintptr_t)DMA1_Stream5_IRQHandler,
    (uintptr_t)DMA1_Stream6_IRQHandler,
    (uintptr_t)ADC_IRQHandler,
    (uintptr_t)CAN1_TX_IRQHandler,
    (uintptr_t)CAN1_RX0_IRQHandler,
    (uintptr_t)CAN1_RX1_IRQHandler,
    (uintptr_t)CAN1_SCE_IRQHandler,
    (uintptr_t)EXTI9_5_IRQHandler,
    (uintptr_t)TIM1_BRK_TIM9_IRQHandler,
    (uintptr_t)TIM1_UP_TIM10_IRQHandler,
    (uintptr_t)TIM1_TRG_COM_TIM11_IRQHandler,
    (uintptr_t)TIM1_CC_IRQHandler,
    (uintptr_t)TIM2_IRQHandler,
    (uintptr_t)TIM3_IRQHandler,
    (uintptr_t)TIM4_IRQHandler,
    (uintptr_t)I2C1_EV_IRQHandler,
    (uintptr_t)I2C1_ER_IRQHandler,
    (uintptr_t)I2C2_EV_IRQHandler,
    (uintptr_t)I2C2_ER_IRQHandler,
    (uintptr_t)SPI1_IRQHandler,
    (uintptr_t)SPI2_IRQHandler,
    (uintptr_t)USART1_IRQHandler,
    (uintptr_t)USART2_IRQHandler,
    (uintptr_t)USART3_IRQHandler,
    (uintptr_t)EXTI15_10_IRQHandler,
    (uintptr_t)RTC_Alarm_IRQHandler,
    (uintptr_t)OTG_FS_WKUP_IRQHandler,
    (uintptr_t)DMA1_Stream7_IRQHandler,
    (uintptr_t)SDIO_IRQHandler,
    (uintptr_t)TIM5_IRQHandler,
    (uintptr_t)SPI3_IRQHandler,
    (uintptr_t)DMA2_Stream0_IRQHandler,
    (uintptr_t)DMA2_Stream1_IRQHandler,
    (uintptr_t)DMA2_Stream2_IRQHandler,
    (uintptr_t)DMA2_Stream3_IRQHandler,
    (uintptr_t)DMA2_Stream4_IRQHandler,
    0,
    0,
    0,
    0,
    (uintptr_t)OTG_FS_IRQHandler,
    (uintptr_t)DMA2_Stream5_IRQHandler,
    (uintptr_t)DMA2_Stream6_IRQHandler,
    (uintptr_t)DMA2_Stream7_IRQHandler,
    (uintptr_t)USART6_IRQHandler,
    (uintptr_t)I2C3_EV_IRQHandler,
    (uintptr_t)I2C3_ER_IRQHandler,
    0,
    0,
    0,
    0,
    0,
    0,
    (uintptr_t)FPU_IRQHandler,
    0,
    0,
    (uintptr_t)SPI4_IRQHandler,
    0,
    0,
    (uintptr_t)SAI1_IRQHandler,
    0,
    0,
    0,
    (uintptr_t)SAI2_IRQHandler,
    (uintptr_t)QUADSPI_IRQHandler,
    (uintptr_t)HDMI_CEC_IRQHandler,
    (uintptr_t)SPDIF_RX_IRQHandler,
    (uintptr_t)FMPI2C1_EV_IRQHandler,
    (uintptr_t)FMPI2C1_ER_IRQHandler,
};

void SystemInit(void) __attribute__((weak));
void SystemInit(void) {}

void Reset_Handler(void) {
  uint32_t *src = &_sidata;
  uint32_t *dst = &_sdata;

  while (dst < &_edata) {
    *dst++ = *src++;
  }

  dst = &_sbss;
  while (dst < &_ebss) {
    *dst++ = 0u;
  }

  SystemInit();
  (void)main();

  while (1) {
  }
}

void Default_Handler(void) {
  while (1) {
  }
}
