#ifndef SUPER_CLASS_H
#define SUPER_CLASS_H

typedef struct SuperClass *  SuperClass_ptr;
SuperClass_ptr create_SuperClass(void );

typedef int (*virtual_fun) (void *);

/*typedef SuperClass_ptr (*fun_SuperClass_ptr) (void);
typedef void (*fun_SuperClass_ptr_void) (SuperClass_ptr,int ,int );
*/
struct SuperClass{
int x;
int y;
virtual_fun sum;

};

void initialize_SuperClass(SuperClass_ptr me,int x,int y);
SuperClass_ptr create_SuperClass(void );
void initialize_SuperClass(SuperClass_ptr me,int x,int y);
#endif
