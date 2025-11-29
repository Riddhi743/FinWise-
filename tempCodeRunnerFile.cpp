#include <iostream>
using namespace std;
int main(){  
int num[] = {2,8,7,6,0};
int array [100];
for (int i=0;i<5;i++){
    array[i]=num[i];
}
for (int i=3;i<5;i++){
    array[i+1]=num[i];
}
array[3]=1;
for (int i=0;i<6;i++){
    cout<<array[i]<<" ";
}
return 0;
} 
