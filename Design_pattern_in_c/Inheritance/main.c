#include <stdio.h>
#include <stdlib.h>
#include "super_class.h"
#include "sub_class.h"
int main()
{   SubClass_ptr S= creat_SubClass();
    initialize_SubClass(S,3,5);

    printf("%d %d\n",S->sum(S),S->mul(S));
    return 0;
}
