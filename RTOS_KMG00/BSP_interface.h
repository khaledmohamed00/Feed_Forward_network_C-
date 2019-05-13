/*
 * BSP.h
 *
 *  Created on: Apr 16, 2019
 *      Author: khaled
 */

#ifndef BSP_INTERFACE_H_
#define BSP_INTERFACE_H_

#define System_Clock 16000000ul
#define OS_TICKS_PER_SEC 1000ul

void init_SYSTICK(void);
void init_pendsv(void);


#endif /* BSP_INTERFACE_H_ */
