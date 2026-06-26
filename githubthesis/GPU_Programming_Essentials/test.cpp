#include<bits/stdc++.h>
using namespace std;
int main(){
    double lamda =0.1;
    double R[5][5]={5,3,0,1,0,
    3,0,1,1,0,
    0,1,0,1,0,
    1,1,1,0,0,
    0,0,0,0,1
    };
    double x[5][2]={0.5,0.4,
    1.3,0.9,
    0.8,0.1,
    0.1,0.8,
    0.6,0.3};    
    double y[5][2]={0.1,0.5,
    0.8,0.2,
    0.3,0.9,
    0.4,0.4,
    0.7,0.1};
    double smat[2][2]={0};
    double svec[2]={0};
    double r_predic[5][5]={0};
   for (int iter =1;iter<=5;iter++){
    for(int u=0;u<5;u++){
        smat[0][0]=0; smat[0][1]=0; smat[1][0]=0; smat[1][1]=0;
        svec[0]=0; svec[1]=0;
        for(int i=0;i<5;i++){
           if (R[u][i] > 0) {
               smat[0][0]+=y[i][0]*y[i][0];
               smat[0][1]+=y[i][0]*y[i][1];
               smat[1][0]+=y[i][1]*y[i][0];
               smat[1][1]+=y[i][1]*y[i][1];
               
               svec[0]+=y[i][0]*R[u][i];
               svec[1]+=y[i][1]*R[u][i];
           }
        }
        smat[0][0]+=lamda;
        smat[1][1]+=lamda;
        double det = smat[0][0]*smat[1][1]-smat[0][1]*smat[1][0];
        x[u][0]=(svec[0]*smat[1][1]-svec[1]*smat[0][1])/det;
        x[u][1]=(svec[1]*smat[0][0]-svec[0]*smat[1][0])/det;
    }
    for(int i=0;i<5;i++){
     smat[0][0]=0; smat[0][1]=0; smat[1][0]=0; smat[1][1]=0;
     svec[0]=0; svec[1]=0;
     for(int u=0;u<5;u++){
        if (R[u][i] > 0) {
            smat[0][0]+=x[u][0]*x[u][0];
            smat[0][1]+=x[u][0]*x[u][1];
            smat[1][0]+=x[u][1]*x[u][0];
            smat[1][1]+=x[u][1]*x[u][1];
            
            svec[0]+=x[u][0]*R[u][i];
            svec[1]+=x[u][1]*R[u][i];
        }
     }
     smat[0][0]+=lamda;
     smat[1][1]+=lamda;
     double det = smat[0][0]*smat[1][1]-smat[0][1]*smat[1][0];
     y[i][0]=(svec[0]*smat[1][1]-svec[1]*smat[0][1])/det;
     y[i][1]=(svec[1]*smat[0][0]-svec[0]*smat[1][0])/det;
    }
   } 
   for(int i=0;i<5;i++){
    for(int j=0;j<5;j++){
        r_predic[i][j]= x[i][0]*y[j][0]+x[i][1]*y[j][1];
        cout<<r_predic[i][j]<<" ";
    }
    cout<<endl;
   }
}
