#include "super_class.h"
#include <stdlib.h>

static int sum(void * me){
SuperClass_ptr p=(SuperClass_ptr)me;
int sum=p->x + p->y;

return sum;
}
void initialize_SuperClass(SuperClass_ptr me,int x,int y){
me->x=x;
me->y=y;

}
SuperClass_ptr create_SuperClass(void ){

 SuperClass_ptr S=(SuperClass_ptr)malloc(sizeof(struct SuperClass));
 S->sum=&sum;

return S;
}




