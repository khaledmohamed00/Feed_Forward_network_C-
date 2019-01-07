/* Board Support Package (BSP) for the EK-TM4C123GXL board */
#include <stdint.h>  /* Standard integers. WG14/N843 C99 Standard */

#include "bsp.h"
#include "miros.h"
#include "TM4C123GH6PM.h" /* the TM4C MCU Peripheral Access Layer (TI) */
#include "tm4c123gh6pm.h"
/* on-board LEDs */
#define LED_RED   (1U << 1)
#define LED_BLUE  (1U << 2)
#define LED_GREEN (1U << 3)
#define TEST_PIN  (1U << 4)

static uint32_t volatile l_tickCtr;
__attribute__((naked)) void assert_failed (char const *file, int line) {
    /* TBD: damage control */
    NVIC_SystemReset(); /* reset the system */
}

void SysTick_Handler(void) {
    GPIOF_AHB->DATA_Bits[TEST_PIN] = TEST_PIN;

    OS_tick();

    __disable_irq();
    OS_sched();
    __enable_irq();

    GPIOF_AHB->DATA_Bits[TEST_PIN] = 0U;
}

void BSP_init(void) {
    SYSCTL->RCGCGPIO  |= (1U << 5); /* enable Run mode for GPIOF */
    SYSCTL->GPIOHBCTL |= (1U << 5); /* enable AHB for GPIOF */
    GPIOF_AHB->DIR |= (LED_RED | LED_BLUE | LED_GREEN | TEST_PIN);
    GPIOF_AHB->DEN |= (LED_RED | LED_BLUE | LED_GREEN | TEST_PIN);
}

void BSP_ledRedOn(void) {
    GPIOF_AHB->DATA_Bits[LED_RED] = LED_RED;
}

void BSP_ledRedOff(void) {
    GPIOF_AHB->DATA_Bits[LED_RED] = 0U;
}

void BSP_ledBlueOn(void) {
    GPIOF_AHB->DATA_Bits[LED_BLUE] = LED_BLUE;
}

void BSP_ledBlueOff(void) {
    GPIOF_AHB->DATA_Bits[LED_BLUE] = 0U;
}

void BSP_ledGreenOn(void) {
    GPIOF_AHB->DATA_Bits[LED_GREEN] = LED_GREEN;
}

void BSP_ledGreenOff(void) {
    GPIOF_AHB->DATA_Bits[LED_GREEN] = 0U;
}

void OS_onStartup(void) {
    //SystemCoreClockUpdate();
    //SysTick_Config(SystemCoreClock / BSP_TICKS_PER_SEC);

    /* set the SysTick interrupt priority (highest) */
    //NVIC_SetPriority(SysTick_IRQn, 0U);
    NVIC_ST_CTRL_R = 0;         // disable SysTick during setup

      NVIC_ST_RELOAD_R = 80000000 / BSP_TICKS_PER_SEC;// reload value

      NVIC_ST_CURRENT_R = 0;      // any write to current clears it

      NVIC_SYS_PRI3_R = (NVIC_SYS_PRI3_R&0x00FFFFFF)|0x00000000; // priority 0

      NVIC_ST_CTRL_R = 0x0007; // enable SysTick with core clock and interrupts
}

void OS_onIdle(void) {
    GPIOF_AHB->DATA_Bits[LED_RED] = LED_RED;
    GPIOF_AHB->DATA_Bits[LED_RED] = 0U;
    //__WFI(); /* stop the CPU and Wait for Interrupt */
}

//void Q_onAssert(char const *module, int loc) {
    /* TBD: damage control */
  //  (void)module; /* avoid the "unused parameter" compiler warning */
    //(void)loc;    /* avoid the "unused parameter" compiler warning */
    //NVIC_SystemReset();
//}
