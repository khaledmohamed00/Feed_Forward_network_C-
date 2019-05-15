#ifndef SUB_CLASS_H
#define SUB_CLASS_H
typedef struct SubClass * SubClass_ptr;
typedef int (*fun) (SubClass_ptr);

struct SubClass{
SuperClass_ptr super;
fun mul;
virtual_fun sum;
};
void initialize_SubClass(SubClass_ptr me,int x,int y);
SubClass_ptr creat_SubClass(void);


#endif // SUB_CLASS_H

