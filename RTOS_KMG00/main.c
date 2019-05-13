#include "TM4C123GH6PM.h"
#include "tm4c123gh6pm.h"
#include "OS_interface.h"

/**
 * main.c
 */


#define TOG_BIT(REG,BIT_NO) REG^=(1<<BIT_NO);
#define NO_OF_Threads 3
#define STACKSIZE 100 // number of 32-bit words in stack
#define no_of_register_in_stack 16
typedef void (*Thread_fun_Type)(void);


uint64_t Count0=0,Count1=0,Count2=0;

uint32_t stack_blinky1[40];
TCB_Type blinky1;

void main_blinky1() {
    while (1) {
        uint32_t volatile i;
        for (i = 1500U; i != 0U; --i) {
            TOG_BIT(GPIOF->DATA,1); // toggle bit

        }
        OS_delay(1U); /* block for 1 tick */
    }
}

uint32_t stack_blinky2[40];
TCB_Type blinky2;
void main_blinky2() {
    while (1) {
        uint32_t volatile i;
        for (i = 3*1500U; i != 0U; --i) {
            TOG_BIT(GPIOF->DATA,2); // toggle bit

        }
        OS_delay(50U); /* block for 50 ticks */
    }
}


uint32_t stack_blinky3[40];
TCB_Type blinky3;
void main_blinky3() {
    while (1) {
        TOG_BIT(GPIOF->DATA,3); // toggle bit

        OS_delay(1000 * 3U / 5U);
    }
}


uint32_t stack_idleThread[40];

int main(void)
{
    SYSCTL->RCGCGPIO |= (1<<5); // enable clock on PortF
    GPIOF->DIR = (1<<1)|(1<<2)|(1<<3);  // make LED pins (PF1, PF2, and PF3) outputs
    GPIOF->DEN = (1<<1)|(1<<2)|(1<<3); // enable digital function on LED pins
    GPIOF->DATA &= ~((1<<1)|(1<<2)|(1<<3)); // turn off leds
    OS_INIT(stack_idleThread, sizeof(stack_idleThread));

    /* start blinky1 thread */
    Create_Thread(&blinky1,
                   &main_blinky1,
                   5U, /* priority */
                   stack_blinky1, sizeof(stack_blinky1));

    /* start blinky2 thread */
    Create_Thread(&blinky2,
                   &main_blinky2,
                   2U, /* priority */

                   stack_blinky2, sizeof(stack_blinky2));

    /* start blinky3 thread */
    Create_Thread(&blinky3,
                   &main_blinky3,
                   1U, /* priority */
                   stack_blinky3, sizeof(stack_blinky3));

    /* transfer control to the RTOS to run the threads */
    OS_launch();


while(1){

}
	return 0;
}
