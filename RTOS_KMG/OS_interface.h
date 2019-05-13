/*
 * OS_interface.h
 *
 *  Created on: Apr 17, 2019
 *      Author: khaled
 */

#ifndef OS_INTERFACE_H_
#define OS_INTERFACE_H_

#define NO_OF_Threads 3
#define STACKSIZE 100 // number of 32-bit words in stack
#define no_of_register_in_stack 16
typedef void (*Thread_fun_Type)(void);


struct TCB_Type{
 uint32_t * sp;
 struct TCB_Type *next;
 int32_t *blocked;
 uint32_t sleep;
};
typedef struct TCB_Type TCB_Type;
TCB_Type TCBS[NO_OF_Threads];
uint32_t Stacks[NO_OF_Threads][STACKSIZE];
TCB_Type *RunPt;

void Initialize_Stack(uint8_t thread_no);
uint8_t create_OS_Thread(uint8_t Thread_NO, Thread_fun_Type Thread_fun );
void Establish_Chain(void);
void OS_launch(uint32_t theTimeSlice);
void Start_OS(void);
void OS_Suspend(void);
void OS_Signal(int32_t *s);
void OS_Sleep(uint32_t sleep_time);
void Scheduler(void);


#endif /* OS_INTERFACE_H_ */
