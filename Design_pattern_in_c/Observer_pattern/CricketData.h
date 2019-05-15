#ifndef CRICKETDATA_H
typedef struct CricketData * CricketData_ptr;
CricketData_ptr create_CricketData(int runs,int wickets, float overs);
void dataChanged(CricketData_ptr me);


#endif
