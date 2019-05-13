/*
 * BSP.c
 *
 *  Created on: Apr 16, 2019
 *      Author: khaled
 */
#include "TM4C123GH6PM.h"
#include "tm4c123gh6pm.h"
#include "BSP_interface.h"
__attribute__((naked)) void assert_failed (char const *file, int line) {
    /* TBD: damage control */
    NVIC_SystemReset(); /* reset the system */
}

void init_SYSTICK(void){
        SysTick->CTRL=0;
        SysTick->LOAD =System_Clock/OS_TICKS_PER_SEC ;
        SysTick->VAL  = 0U;
        NVIC_SYS_PRI3_R = (NVIC_SYS_PRI3_R&0x00FFFFFF)|0x00000000; // priority 0
        SysTick->CTRL = (1U << 2) | (1U << 1) | 1U;

}
void init_pendsv(void){
    /* set the PendSV interrupt priority to the lowest level 0xFF */
       NVIC_SYS_PRI3_R |= (0xFFU << 16);


}
void SysTick_Handler(void) {

    OS_tick();

    __disable_irq();
    OS_Scheduler();
    __enable_irq();

}

