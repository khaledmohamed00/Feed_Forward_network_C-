#include <stdio.h>
#include <stdlib.h>
#include"Pizza.h"
#include"Margherita.h"
#include"ChickenFiesta.h"
#include"Barbeque.h"
#include"FreshTomato.h"
int main()
{ Pizza_ptr pizza=create_ChickenFiesta();
  Pizza_ptr pizza1=create_Barbeque(pizza);
  Pizza_ptr pizza2=create_FreshTomato(pizza1);

    printf("%d\n",pizza2->get_cost(pizza2));
    printf("%s\n",pizza2->get_description(pizza2));

    return 0;
}
