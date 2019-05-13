/*
 * OS_interface.h
 *
 *  Created on: Apr 16, 2019
 *      Author: khaled
 */

#ifndef OS_INTERFACE_H_
#define OS_INTERFACE_H_

typedef struct {
 uint32_t * sp;
 uint32_t time_out;
 uint32_t prio;
}TCB_Type;

typedef void (*Thread_fun_Type)(void);


void main_idleThread() ;
void Create_Thread(TCB_Type *thread_TCB ,Thread_fun_Type thread_function,uint32_t prio,void *stack_ptr,uint32_t stack_size);
void OS_INIT(void *stack_ptr,uint32_t stack_size);
void OS_launch();
void OS_Scheduler(void);
void OS_tick(void);
void OS_delay(uint32_t ticks) ;




#endif /* OS_INTERFACE_H_ */
