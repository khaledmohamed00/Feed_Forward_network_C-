#include "customer.h"
#include <stdlib.h>
struct Customer
 {
 const char* name;
 int address;
 //size_t noOfOrders;
// Order orders[42];
 };
 CustomerPtr createCustomer(const char* name, const int address)
 {
 CustomerPtr customer = malloc(sizeof * customer);
 if(customer)
 {customer->name=name;
  customer->address=address;
 /* Initialize each field in the customer... */
 }
 return customer;
 }
 void destroyCustomer(CustomerPtr customer)
 {
 /* Perform clean-up of the customer internals, if necessary. */
 free(customer);
 }
