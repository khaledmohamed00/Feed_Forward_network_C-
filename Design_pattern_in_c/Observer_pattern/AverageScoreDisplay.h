#ifndef AVG_H
typedef struct AverageScoreDisplay * AverageScoreDisplay_ptr;
Observer_ptr AverageScoreDisplay_ptr_to_Observer_ptr(AverageScoreDisplay_ptr me);
AverageScoreDisplay_ptr create_AverageScoreDisplay(void);
#endif
