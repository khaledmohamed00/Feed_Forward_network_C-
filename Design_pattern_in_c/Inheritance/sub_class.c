#include "super_class.h"
#include "sub_class.h"
#include <stdlib.h>


void initialize_SubClass(SubClass_ptr me,int x,int y){
initialize_SuperClass(me->super,x,y);

}

static int sum(void * me){
SubClass_ptr p=(SubClass_ptr)me;
int sum=p->super->x + p->super->y;

return sum;
}

int mul(SubClass_ptr me){

int mul=me->super->x * me->super->y;

return mul;
}

SubClass_ptr creat_SubClass(void){
SuperClass_ptr super= create_SuperClass();
SubClass_ptr S=(SubClass_ptr)malloc(sizeof(struct SubClass));
S->super=super;
S->mul=&mul;
S->sum=&sum;
return S;
}

