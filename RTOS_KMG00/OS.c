/*
 * OS.c
 *
 *  Created on: Apr 16, 2019
 *      Author: khaled
 */
#include "TM4C123GH6PM.h"
#include "tm4c123gh6pm.h"
#include "OS_interface.h"
#include "BSP_interface.h"

TCB_Type *OS_thread[32 + 1]; /* array of threads started so far */
uint32_t OS_readySet=0;
uint32_t OS_delayedSet;
TCB_Type idleThread;
TCB_Type *volatile current_thread;
TCB_Type *volatile next_thread;


#define LOG2(x) (32U - __clz(x))

static __inline int __clz(int x)
{
    int numZeros;
    __asm__ ("clz %0, %1" : "=r" (numZeros) : "r" (x) : "cc");
    return numZeros;
}

void main_idleThread(void) {
    while (1) {
       // OS_onIdle();
        int count=0;
        count++;
        GPIOF_AHB->DATA_Bits[1] = 1;
        GPIOF_AHB->DATA_Bits[2] = 0U;
    }
}

void Create_Thread(TCB_Type *thread_TCB ,Thread_fun_Type thread_function,uint32_t prio,void *stack_ptr,uint32_t stack_size){

    uint32_t *sp = (uint32_t *)((((uint32_t)stack_ptr + stack_size) / 8) * 8);
    uint32_t* stack_limit = (uint32_t *)(((((uint32_t)stack_ptr - 1U) / 8) + 1U) * 8);

    *(--sp) = (1U << 24);  /* xPSR */
    *(--sp) = (uint32_t)thread_function; /* PC */
    *(--sp) = 0x0000000EU; /* LR  */
    *(--sp) = 0x0000000CU; /* R12 */
    *(--sp) = 0x00000003U; /* R3  */
    *(--sp) = 0x00000002U; /* R2  */
    *(--sp) = 0x00000001U; /* R1  */
    *(--sp) = 0x00000000U; /* R0  */
    /* additionally, fake registers R4-R11 */
    *(--sp) = 0x0000000BU; /* R11 */
    *(--sp) = 0x0000000AU; /* R10 */
    *(--sp) = 0x00000009U; /* R9 */
    *(--sp) = 0x00000008U; /* R8 */
    *(--sp) = 0x00000007U; /* R7 */
    *(--sp) = 0x00000006U; /* R6 */
    *(--sp) = 0x00000005U; /* R5 */
    *(--sp) = 0x00000004U; /* R4 */

    thread_TCB->sp=sp;

    for (sp = sp - 1U; sp >= stack_limit; --sp) {
                    *sp = 0xDEADBEEFU;
                }
    thread_TCB->prio=prio;
    OS_thread[prio]=thread_TCB;

    if (prio > 0U) {
            OS_readySet |= (1U << (prio - 1U));
        }

}

void OS_INIT(void *stack_ptr,uint32_t stack_size){
    NVIC_SYS_PRI3_R|= (0xFFU << 16);//pendSV
    //*(uint32_t volatile *)0xE000ED20 |= (0xFFU << 16);//NVIC_SYS_PRI3_R

    Create_Thread(&idleThread,&main_idleThread,0,stack_ptr,stack_size);

}

void OS_launch(){
    init_SYSTICK();
    __disable_irq();
    OS_Scheduler();
    __enable_irq();

}

void OS_Scheduler(void){

    if(OS_readySet==0){
    next_thread=OS_thread[0];
   }
    else {
        next_thread = OS_thread[LOG2(OS_readySet)];
        }

        /* trigger PendSV, if needed */
        if (next_thread != current_thread) {
            //NVIC_INT_CTRL_R = (1U << 28);//Trigger pendSV
            *(uint32_t volatile *)0xE000ED04 = (1U << 28);//NVIC_INT_CTRL_R

        }

}

void PendSV_Handler(void) {
__asm(
  //  "IMPORT  current_thread \n" /* extern variable */
   // "IMPORT  next_thread\n " /* extern variable */

    /* __disable_irq(); */
    "CPSID         I\n"

    /* if (current_thread != (OSThread *)0) { */
    "LDR           r1,=current_thread\n"
    "LDR           r1,[r1,#0x00]\n"
    "CBZ           r1,PendSV_restore\n"

    /*     push registers r4-r11 on the stack */
    "PUSH          {r4-r11}\n"

    /*     current_thread->sp = sp; */
    "LDR           r1,=current_thread\n"
    "LDR           r1,[r1,#0x00]\n"
    "STR           sp,[r1,#0x00]\n"
    /* } */

"PendSV_restore:\n"
    /* sp = next_thread->sp; */
    "LDR           r1,=next_thread\n"
    "LDR           r1,[r1,#0x00]\n"
    "LDR           sp,[r1,#0x00]\n"

    /* current_thread = next_thread; */
    "LDR           r1,=next_thread\n"
    "LDR           r1,[r1,#0x00]\n"
    "LDR           r2,=current_thread\n"
    "STR           r1,[r2,#0x00]\n"

    /* pop registers r4-r11 */
    "POP           {r4-r11}\n"

    /* __enable_irq(); */
    "CPSIE         I\n"

    /* return to the next thread */
    "BX            lr\n");
}

void OS_tick(void){
    uint32_t workingSet = OS_delayedSet;
        while (workingSet != 0U) {
            TCB_Type *t = OS_thread[LOG2(workingSet)];
            uint32_t bit;

            bit = (1U << (t->prio - 1U));
            --t->time_out;
            if (t->time_out == 0U) {
                OS_readySet   |= bit;  /* insert to set */
                OS_delayedSet &= ~bit; /* remove from set */
            }
            workingSet &= ~bit; /* remove from working set */
        }
}

void OS_delay(uint32_t ticks) {
    uint32_t bit;
    __disable_irq();


    current_thread->time_out = ticks;
    bit = (1U << (current_thread->prio - 1U));
    OS_readySet &= ~bit;
    OS_delayedSet |= bit;
    OS_Scheduler();
    __enable_irq();
}
