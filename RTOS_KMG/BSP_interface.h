/*
 * BSP_interface.h
 *
 *  Created on: Apr 17, 2019
 *      Author: khaled
 */

#ifndef BSP_INTERFACE_H_
#define BSP_INTERFACE_H_

#define NVIC_EN0_INT19          0x00080000  // Interrupt 19 enable
#define TIMER_ICR_TATOCINT      0x00000001  // GPTM TimerA Time-Out Raw

void Timer0A_Init(unsigned short period);
void SYS_TICK_Init(uint32_t theTimeSlice);
#endif /* BSP_INTERFACE_H_ */
