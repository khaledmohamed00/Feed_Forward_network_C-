/*
 * OS.c
 *
 *  Created on: Apr 17, 2019
 *      Author: khaled
 */


#include "TM4C123GH6PM.h"
#include "tm4c123gh6pm.h"
#include "OS_interface.h"
#include "BSP_interface.h"

void Initialize_Stack(uint8_t thread_no){
    TCBS[thread_no].sp=&Stacks[thread_no][STACKSIZE-no_of_register_in_stack];
   // TCBS[thread_no].blocked=0;
    Stacks[thread_no][STACKSIZE-1] = 0x01000000; // Thumb bit
    Stacks[thread_no][STACKSIZE-3] = 0x14141414; // R14
    Stacks[thread_no][STACKSIZE-4] = 0x12121212; // R12
    Stacks[thread_no][STACKSIZE-5] = 0x03030303; // R3
    Stacks[thread_no][STACKSIZE-6] = 0x02020202; // R2
    Stacks[thread_no][STACKSIZE-7] = 0x01010101; // R1
    Stacks[thread_no][STACKSIZE-8] = 0x00000000; // R0
    Stacks[thread_no][STACKSIZE-9] = 0x11111111; // R11
    Stacks[thread_no][STACKSIZE-10] = 0x10101010; // R10
    Stacks[thread_no][STACKSIZE-11] = 0x09090909; // R9
    Stacks[thread_no][STACKSIZE-12] = 0x08080808; // R8
    Stacks[thread_no][STACKSIZE-13] = 0x07070707; // R7
    Stacks[thread_no][STACKSIZE-14] = 0x06060606; // R6
    Stacks[thread_no][STACKSIZE-15] = 0x05050505; // R5
    Stacks[thread_no][STACKSIZE-16] = 0x04040404; // R4
}

uint8_t create_OS_Thread(uint8_t Thread_NO, Thread_fun_Type Thread_fun ){
    __disable_irq();
    TCBS[Thread_NO].blocked=0;
    TCBS[Thread_NO].sleep=0;

    Initialize_Stack(Thread_NO);
    Stacks[Thread_NO][STACKSIZE-2] =(uint32_t) Thread_fun;

    __enable_irq();

    return 1;
}

void Establish_Chain(void){

    __disable_irq();
      TCBS[0].next = &TCBS[1]; // 0 points to 1
      TCBS[1].next = &TCBS[2]; // 1 points to 2
      TCBS[2].next = &TCBS[0]; // 2 points to 0
      RunPt = &TCBS[0];
      __enable_irq();

}

void OS_launch(uint32_t theTimeSlice){

        SYS_TICK_Init(theTimeSlice);
        Establish_Chain();
        Timer0A_Init(160000);
        Start_OS();
}
void Start_OS(void){

    __asm(
        "CPSID I\n"
        "LDR r0 ,=RunPt\n"
        "LDR r1 ,[r0] \n"
        "LDR sp ,[r1]\n"
        "POP {r4-r11}\n"
        "POP {r0-r3}\n"
        "POP {r12}\n"
        "ADD sp,sp,#4\n"
        "POP {lr}\n"
        "ADD sp,sp,#4\n"
        "CPSIE I\n"
        "BX lr\n"



    );


}

void OS_Suspend(void){
    __enable_irq();

    NVIC_ST_CURRENT_R = 0;        // reset counter
    NVIC_INT_CTRL_R = 0x04000000; // trigger SysTick
}
void OS_Wait(int32_t *s){
    __disable_irq();
  (*s) = (*s) - 1;
  if((*s) < 0){
    RunPt->blocked = s; // reason it is blocked
    __enable_irq();
    OS_Suspend();       // run thread switcher
  }
  __enable_irq();
}
void OS_Signal(int32_t *s){
    TCB_Type *pt;
  __disable_irq();
  (*s) = (*s) + 1;
  if((*s) <= 0){
    pt = RunPt->next;   // search for a thread blocked on this semaphore
    while(pt->blocked != s){
      pt = pt->next;
    }
    pt->blocked = 0;    // wakeup this one
  }
  __enable_irq();
}


void OS_Sleep(uint32_t sleep_time){
    RunPt->sleep=sleep_time;
    OS_Suspend();
}

void Scheduler(void){
  RunPt = RunPt->next; // skip at least one
  while((RunPt->sleep)||(RunPt-> blocked)){
    RunPt = RunPt->next; // find one not sleeping and not blocked
  }
}
void SysTick_Handler(void){
    __asm(
          "CPSID    I\n"
          "PUSH     {r4-r11}\n"
          "LDR      r0 ,=RunPt\n"
          "LDR      r1 ,[r0]\n"
          "STR      sp ,[r1]\n"
          "PUSH     {r0,lr}\n"
          "BL       Scheduler\n"
          "POP      {r0,lr}\n"
          "LDR      r1,[r0]\n"
          "LDR      sp ,[r1]\n"
          "POP      {r4-r11}\n"
          "CPSIE    I\n"
          "BX       lr\n"


    );
}

void Timer0A_IRQHandler(void){
  TIMER0_ICR_R = TIMER_ICR_TATOCINT;// acknowledge timer0A timeout
 // (*PeriodicTask)();                // execute user task
 for(int i=0;i<NO_OF_Threads;i++){
     if(TCBS[i].sleep){
         TCBS[i].sleep--;
     }

 }

}
