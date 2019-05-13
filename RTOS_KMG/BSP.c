/*
 * BSP.c
 *
 *  Created on: Apr 17, 2019
 *      Author: khaled
 */
#include "TM4C123GH6PM.h"
#include "tm4c123gh6pm.h"
#include "BSP_interface.h"

__attribute__((naked)) void assert_failed (char const *file, int line) {
    /* TBD: damage control */
    NVIC_SystemReset(); /* reset the system */
}
void Timer0A_Init(unsigned short period){
  SYSCTL_RCGC1_R |= SYSCTL_RCGC1_TIMER0; // 0) activate timer0
 // PeriodicTask = task;             // user function
  TIMER0_CTL_R &= ~0x00000001;     // 1) disable timer0A during setup
  TIMER0_CFG_R = 0x00000004;       // 2) configure for 16-bit timer mode
  TIMER0_TAMR_R = 0x00000002;      // 3) configure for periodic mode
  TIMER0_TAILR_R = period;         // 4) reload value
  TIMER0_TAPR_R = 49;              // 5) 1us timer0A
  TIMER0_ICR_R = 0x00000001;       // 6) clear timer0A timeout flag
  TIMER0_IMR_R |= 0x00000001;      // 7) arm timeout interrupt
  NVIC_PRI4_R = (NVIC_PRI4_R&0x00FFFFFF)|0x40000000; // 8) priority 2
  NVIC_EN0_R |= NVIC_EN0_INT19;    // 9) enable interrupt 19 in NVIC
  TIMER0_CTL_R |= 0x00000001;      // 10) enable timer0A
  __enable_irq();
}

void SYS_TICK_Init(uint32_t theTimeSlice){

    SysTick->CTRL=0;
    SysTick->LOAD = theTimeSlice-1;
    SysTick->VAL  = 0U;
    NVIC_SYS_PRI3_R =(NVIC_SYS_PRI3_R&0x00FFFFFF)|0xE0000000;
    SysTick->CTRL = (1U << 2) | (1U << 1) | 1U;


}
