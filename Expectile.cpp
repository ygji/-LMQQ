#include <math.h>
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include<cstdlib>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_permutation.h>
#define random(x)(rand()%x)
#include<time.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_eigen.h>
int main (void)
{


double var_v(const gsl_vector *vect);
double cor_v(const gsl_vector *vect1,const gsl_vector *vect2,const char method[6]);
double cor_m(const gsl_matrix *matr,const int m,const int n,const int k,const int h1,const int h2);
double min_v(const gsl_vector *vect);
//int vect_maxtrix1c(const gsl_vector *vec, const int length, const int dim, gsl_matrix *lowermx);
int vect_maxtrix1r(const gsl_vector *vec, const int length, const int dim, gsl_matrix *lowermx);
int rmvnorm(const gsl_rng *r,const gsl_vector *mean, const gsl_matrix *var, gsl_vector *result);
int rmvnorm_svd(const gsl_rng *r,const gsl_vector *mean, const gsl_matrix *cvar, gsl_vector *result);
int rmvnorm_eigen(const gsl_rng *r,const gsl_vector *mean, const gsl_matrix *var, gsl_vector *result);
int two_vecmul(const gsl_vector *twovec1,const int len1,const gsl_vector *twovec2,const int len2,gsl_matrix *maxtwo);
int apply(const gsl_matrix *A,int rowORcol,char method[6],gsl_vector *result);
int save_datam(const gsl_matrix *needsavedm,const int rowm,const int colm,char method[4], char Filename1[4],const double quantile,char examplebumber[10], const int iter,char distribution[7]);
int submatrix(const gsl_matrix *initialmatrix,const int startrow, const int endrow,const int startcol, const int endcol,gsl_matrix *submaxtrix);
int length_vec(gsl_vector * a);
int numrows_mat(gsl_matrix * A);
int numcols_mat(gsl_matrix * B);
int subvec(const gsl_vector *initialvec,const int start,const int end, gsl_vector *subvec);
int mixed_adlassoEx(const int examplebumber,char method[4],const gsl_rng *ccr,const gsl_vector *y, const int N_train, const gsl_matrix *x, const int p, const int q, const int *tabsubject,const double tau, const int n_sampler,const int simk,char distribution[7],double const tau_expertile);

int simk, i, j;//simk--第simk个数据examplebumber 注意修改
const int N_train=50,N_new=51,Ni=5,p1=8,q1=8,K=100,sd=1,n1_sampler=20000,p=9,q=9;
double tau=0.0;//the tau quantile
	///////////////////误差分布
	char distribution[]={"norm"};
//	char distribution[]={"t3"};
//	char distribution[]={"laplace"};
	//////////////////////////////
char FileName[100];
int tabsubject_train[N_train];
for(i=0;i<N_train;i++){tabsubject_train[i]=Ni;}//注意修改4
clock_t start, finish;  
double  duration;  
start = clock();     

gsl_vector *tauall=gsl_vector_calloc(5);
gsl_vector_set(tauall,0,0.1);
gsl_vector_set(tauall,1,0.3);
gsl_vector_set(tauall,2,0.5);
gsl_vector_set(tauall,3,0.7);
gsl_vector_set(tauall,4,0.9);
//////////////// expertile New add
char norm[]={"norm"};
char t3[]={"t3"};
char laplace[]={"laplace"};
double tau_expertile;
gsl_vector *tauEXall=gsl_vector_calloc(3);
if(strcmp(distribution,norm)==0){
	gsl_vector_set(tauEXall,0,0.03440043);
    gsl_vector_set(tauEXall,1,0.21032251);
    gsl_vector_set(tauEXall,2,0.5);
                                }

if(strcmp(distribution,t3)==0){
	gsl_vector_set(tauEXall,0,0.0672744);
    gsl_vector_set(tauEXall,1,0.2612239);
    gsl_vector_set(tauEXall,2,0.5);
                                }

if(strcmp(distribution,laplace)==0){
	gsl_vector_set(tauEXall,0,0.05526578);
    gsl_vector_set(tauEXall,1,0.27006939);
    gsl_vector_set(tauEXall,2,0.5);
                                }
	 ////////////////////////  end

int M_train=0;
for(i=0;i<N_train;i++){M_train=M_train+tabsubject_train[i];}
 double *x1_tempnew = new double[M_train*p1]; 
 double *y_tempnew = new double[M_train]; 
gsl_matrix *x1_temp=gsl_matrix_calloc(M_train,p1);//initial x
gsl_matrix *x_temp=gsl_matrix_calloc(M_train,p);//cbind(1,x1)

gsl_vector *xcol_temp=gsl_vector_calloc(M_train);
gsl_matrix_set_all(x_temp,1.0);
gsl_vector *y_temp = gsl_vector_calloc(M_train);
gsl_vector *suby=gsl_vector_calloc(Ni);;
int *addsubject=new int[N_new];
    int sub_temp;
    addsubject[0]=0;
	 addsubject[N_train]=M_train;
    for(i=1;i<N_train;i++){
	sub_temp=0;
	for(j=0;j<i;j++){sub_temp=sub_temp+tabsubject_train[j];}
	                addsubject[i]=sub_temp;
                      }	
	
FILE *fp;
FILE *fp2;
long int utim;// 时间随机数
gsl_rng * ccr1; 
ccr1 = gsl_rng_alloc (gsl_rng_taus);         /* generate beta,gamma*/
//printf("ccr1=%d",*ccr1);
//printf("\n");
int itau=0,examplebumber=1;
for(examplebumber=4;examplebumber<=4;examplebumber++){
for(itau=0;itau<3;itau++){/////////////tau开始
	tau=gsl_vector_get(tauall,itau);	
	tau_expertile=gsl_vector_get(tauEXall,itau);
	printf("tau=%lf\n",tau);
	printf("tau_expertile=%.9lf\n",tau_expertile);
	printf("gaussian_P=%.9lf\n",gsl_cdf_gaussian_P(1.0,1.0));
for(simk=1;simk<=K;simk++){
	//打开x
	  sprintf(FileName,"I:\\Paper\\2016paper1data\\simulation%d\\%s\\%s_%1.1f_%dx.txt",examplebumber,distribution,distribution,tau,simk);
      puts(FileName);
      fp=fopen(FileName,"r"); 
      if(!fp){
            printf("Can not open the file x!\n");
            exit(0);
             } 	
       for(i=0;i<M_train;i++){  
		   for(j=0;j<p1;j++){
               fscanf(fp,"%lf ",&x1_tempnew[i*p1+j]); //读数据，读入读出要一致double必须是%lf
		     //  printf("%lf\t",x_temp[i*p+j]); //显示
		                    }		 
		   // putchar('\n'); 
                              }
	   
	   for(i=0;i<M_train;i++){  
		   for(j=0;j<p1;j++){
               gsl_matrix_set(x1_temp,i,j,x1_tempnew[i*p1+j]); //读数据，读入读出要一致double必须是%lf		      
		                    } 
		                     }	       
	
	   for(i=0;i<p1;i++){gsl_matrix_get_col(xcol_temp,x1_temp,i);
	                    gsl_matrix_set_col(x_temp,i+1,xcol_temp);
	                    }

	   for(i=0;i<2;i++){for (j = 0;j < p;j++){
			   printf("%lf\t",gsl_matrix_get(x_temp, i, j));
	                            }
	                    printf("\n\n");
	                   }
	  
      printf("\n");
	   int lengt;
	   lengt=numrows_mat(x_temp);
	    printf("%d",lengt);
		printf("\n");
		lengt=numcols_mat(x_temp);
	    printf("%d",lengt);
		printf("\n");
	       
	   //printf("\n");
       fclose(fp); 
	 //下面打开y
	    sprintf(FileName,"I:\\Paper\\2016paper1data\\simulation%d\\%s\\%s_%1.1f_%dy.txt",examplebumber,distribution,distribution,tau,simk);
       puts(FileName);
       fp2=fopen(FileName,"r"); 
       if(!fp2){
            printf("Can not open the file y!\n");
            exit(0);
             } 	
       for(i=0;i<M_train;i++){
             fscanf(fp2,"%lf ",&y_tempnew[i]); //读数据，读入读出要一致double必须是%lf	       
                              }
	   
	   for(i=0;i<M_train;i++){ 
              gsl_vector_set(y_temp,i,y_tempnew[i]); //读数据，读入读出要一致double必须是%lf		      
                             }	 
	
	 
		 fclose(fp2); 
	

         lengt=length_vec(y_temp);
	
		 
	
   srand((int)time(0)); 
   gsl_rng_set(ccr1,1235+simk);


 	char lassoexpEx[]={"lassoexpEx"};
	char adlassoexpEx[]={"adlassoexpEx"};
	
 mixed_adlassoEx(examplebumber,adlassoexpEx,ccr1,y_temp,N_train,x_temp,p,q,tabsubject_train,tau,n1_sampler,simk,distribution,tau_expertile);
    
                }
				}//tau结束
				}//examplebumber结束
			
   finish = clock();  
   duration = (double)(finish - start) / CLOCKS_PER_SEC;   
   printf( "\n\nTotal time is %f seconds\n", duration );  
delete []x1_tempnew;x1_tempnew=nullptr;
delete []y_tempnew;y_tempnew=nullptr;
gsl_vector_free(tauall);
gsl_vector_free(tauEXall);
gsl_vector_free(y_temp);
gsl_matrix_free(x1_temp);
gsl_matrix_free(x_temp);
gsl_vector_free(xcol_temp);
gsl_vector_free(suby);

gsl_rng_free(ccr1);
return 0;
}

int save_datam(const gsl_matrix *needsavedm,const int rowm,const int colm,char method[4], char Filename1[4],const double quantile,const int examplebumber, const int iter,char distribution[7]){

	int i, j;
	char cnn='\n';
    char FileName2[100];
	FILE *fp1;	 
	sprintf(FileName2,"I:\\Paper\\2016paper1data\\simulation%d\\%s\\%.1lf\\%s_%s%d.txt",examplebumber,distribution,quantile,method,Filename1,iter);
	//puts(FileName2);
	if ((fp1 = fopen(FileName2,"w"))==NULL){
           printf("the file can not open..");
           exit(0);
                                          }    //出错返回

   for( i = 0;i <rowm;i++){
         for( j= 0;j <colm;j++){
              fprintf(fp1,"%lf\t",gsl_matrix_get (needsavedm, i, j)); //把needsavedm中的元素依次读到fp1中,要有\t
                            }
          fprintf(fp1,"%c",cnn);  //每一行都加回车
                             }
    fclose(fp1); 
	 return 0;

} 


double rinvGauss1(const gsl_rng *r,double mu, double lambda){
 	double x1, y, z;
 	y  = gsl_ran_chisq(r,1);
 	x1 = (mu/(2*lambda))*(2*lambda + mu*y - sqrt(4*lambda*mu*y + mu*mu*y*y));
 	z  = gsl_ran_beta(r,1.0,1.0);
 	// Rprintf("y=%f, x1=%f, z=%f\n", y, x1, z);
 	if(z < mu/(mu + x1)){
 		return(x1);
 	}else{
 	  return(mu*mu/x1);
  }
}


int rmvnorm_eigen(const gsl_rng *r,const gsl_vector *mean, const gsl_matrix *cvar, gsl_vector *result){
double min_v(const gsl_vector *vect);
if(cvar->size1!=cvar->size2){printf("协方差矩阵不是方阵");exit(0);}
if(cvar->size1!=mean->size){printf("均值和协方差维数不相同");exit(0);}
int k,n=mean->size;
gsl_vector *eval=gsl_vector_alloc (n);
gsl_matrix *evec=gsl_matrix_alloc (n, n);
gsl_eigen_symmv_workspace *w=gsl_eigen_symmv_alloc(n);
gsl_vector *stdnorm=gsl_vector_alloc (n);
gsl_matrix *AA=gsl_matrix_alloc (n, n);
gsl_matrix_memcpy(AA,cvar);
gsl_eigen_symmv (AA,eval,evec, w);
double temp=0.0;

if(min_v(eval)<0.0){for(k=0;k<n;k++){if(gsl_vector_get(eval,k)<0.0)gsl_vector_set(eval,k,0.0);}}
for(k=0; k<n; k++){temp=sqrt(gsl_vector_get(eval,k));gsl_vector_set(stdnorm,k,temp*gsl_ran_ugaussian(r));}
gsl_blas_dgemv(CblasNoTrans,1.0,evec,stdnorm,0.0,result);
gsl_vector_add(result,mean);

gsl_eigen_symmv_free (w);
gsl_matrix_free(evec);
gsl_vector_free(eval);
gsl_vector_free(stdnorm);
gsl_matrix_free(AA);
return 0;
}


double sum_v(const gsl_vector *vect){
//caculate the sum of vector
	int n=vect->size;
 	double sum=0.0;
	for(int i=0;i<n;i++){sum=sum+gsl_vector_get(vect,i);}
 	return(sum);
  }
double mean_v(const gsl_vector *vect){
//caculate the mean of vector
	int n=vect->size;
 	double mean=0.0;
	for(int i=0;i<n;i++){mean=mean+gsl_vector_get(vect,i);}
 	return(mean/n);
  }
double min_v(const gsl_vector *vect){
	double min=gsl_vector_get(vect,0);	
	int n=vect->size;
	for(int i=1;i<n;i++){if(min>gsl_vector_get(vect,i)){min=gsl_vector_get(vect,i);}}
	return(min);
}

double var_v(const gsl_vector *vect){
	//caculate the variance of vector
 	double var=0.0,mean=0.0;
	int i,n=vect->size;
	for(i=0;i<n;i++){mean=mean+gsl_vector_get(vect,i);}
	mean=mean/n;
	for(i=0;i<n;i++){var=var+(gsl_vector_get(vect,i)-mean)*(gsl_vector_get(vect,i)-mean);}	
 	return(var/n);
  }
double cor_v(const gsl_vector *vect1,const gsl_vector *vect2,const char method[6]){
	//caculate the \pho of two vectors
 	double var1=0.0,mean1=0.0,var2=0.0,mean2=0.0,covtemp;	
	int i,n=vect1->size;
	char cov[]={"cov"};
	char cor[]={"cor"};
	for(i=0;i<n;i++){mean1=mean1+gsl_vector_get(vect1,i);mean2=mean2+gsl_vector_get(vect2,i);}
	mean1=mean1/n;mean2=mean2/n;
	covtemp=0.0;
	for(i=0;i<n;i++){var1=var1+(gsl_vector_get(vect1,i)-mean1)*(gsl_vector_get(vect1,i)-mean1);
	                 var2=var2+(gsl_vector_get(vect2,i)-mean2)*(gsl_vector_get(vect2,i)-mean2);
					 covtemp=covtemp+(gsl_vector_get(vect1,i)-mean1)*(gsl_vector_get(vect2,i)-mean2);					 
	                }
	var1=var1/n;
	var2=var2/n;
	covtemp=covtemp/n;
	if(strcmp(method,cov)==0)
 	    return(covtemp);
	else if(strcmp(method,cor)==0)
		return(covtemp/sqrt(var1*var2));
	else return(0);
  }

double cor_m(const gsl_matrix *matr,const int m,const int n,const int k,const int h1,const int h2){
	//caculate the \pho of two (col or row)vectors in a matrix
	//matrix_{m*n}--cor(i,j)
	//k=1----mean(col)(/m);k=2---mean(row)(/n)
	double var1=0.0,mean1=0.0,var2=0.0,mean2=0.0,cov=0.0,cor=0.0;
	int i;
	gsl_vector *vect1=gsl_vector_calloc(m);
	gsl_vector *vect2=gsl_vector_calloc(m);
    gsl_vector *vect3=gsl_vector_calloc(n);
	gsl_vector *vect4=gsl_vector_calloc(n);
	if(k==1){
		gsl_matrix_get_col(vect1,matr,h1);
		gsl_matrix_get_col(vect2,matr,h2);
		for(i=0;i<m;i++){mean1=mean1+gsl_vector_get(vect1,i);mean2=mean2+gsl_vector_get(vect2,i);}
	    mean1=mean1/m;mean2=mean2/m;
	    for(i=0;i<m;i++){var1=var1+(gsl_vector_get(vect1,i)-mean1)*(gsl_vector_get(vect1,i)-mean1);
	                     var2=var2+(gsl_vector_get(vect2,i)-mean2)*(gsl_vector_get(vect2,i)-mean2);
					     cov=cov+(gsl_vector_get(vect1,i)-mean1)*(gsl_vector_get(vect2,i)-mean2);
	                    }
	   cor=cov/sqrt(var1*var2);	
	        }
	if(k==2){
		gsl_matrix_get_row(vect3,matr,h1);
		gsl_matrix_get_row(vect4,matr,h2);		
		for(i=0;i<n;i++){mean1=mean1+gsl_vector_get(vect3,i);mean2=mean2+gsl_vector_get(vect4,i);}
	    mean1=mean1/n;mean2=mean2/n;
	    for(i=0;i<n;i++){var1=var1+(gsl_vector_get(vect3,i)-mean1)*(gsl_vector_get(vect3,i)-mean1);
	                     var2=var2+(gsl_vector_get(vect4,i)-mean2)*(gsl_vector_get(vect4,i)-mean2);
					     cov=cov+ (gsl_vector_get(vect3,i)-mean1)*(gsl_vector_get(vect4,i)-mean2);
	                    }
	   cor=cov/sqrt(var1*var2);	
	        }
	gsl_vector_free(vect1);
	gsl_vector_free(vect2);
	gsl_vector_free(vect3);
	gsl_vector_free(vect4);
	return(cor);	
 	
  }
int apply(const gsl_matrix *A,int rowORcol,char method[6],gsl_vector *result){
	double sum_v(const gsl_vector *vect);
	double mean_v(const gsl_vector *vect);
	double var_v(const gsl_vector *vect);
	int dimrow=A->size1,dimcol=A->size2;
	int k;
	char sum[]={"sum"};
	char mean[]={"mean"};
	char var[]={"var"};	
	if(rowORcol==1){
		if(dimrow!=result->size){printf("result维数不匹配");exit(0);}
	    gsl_vector *rowA=gsl_vector_calloc(dimcol);
		if(strcmp(method,sum)==0){
			for(k=0;k<dimrow;k++){
				gsl_matrix_get_row(rowA,A,k);
				gsl_vector_set(result,k,sum_v(rowA));
			                      }
		                         }
		if(strcmp(method,mean)==0){
		    for(k=0;k<dimrow;k++){
				gsl_matrix_get_row(rowA,A,k);
				gsl_vector_set(result,k,mean_v(rowA));
			                     }
	                              }	
		if(strcmp(method,var)==0){
			for(k=0;k<dimrow;k++){
				gsl_matrix_get_row(rowA,A,k);
				gsl_vector_set(result,k,var_v(rowA));
			                      }
		                           }
	
	               }
	if(rowORcol==2){
		if(dimcol!=result->size){printf("result维数不匹配");exit(0);}
	    gsl_vector *colA=gsl_vector_calloc(dimrow);
		if(strcmp(method,sum)==0){
			for(k=0;k<dimcol;k++){
				gsl_matrix_get_col(colA,A,k);
				gsl_vector_set(result,k,sum_v(colA));
			                      }
		                         }
		if(strcmp(method,mean)==0){
		    for(k=0;k<dimcol;k++){
				gsl_matrix_get_col(colA,A,k);
				gsl_vector_set(result,k,mean_v(colA));
			                     }
	                              }	
		if(strcmp(method,var)==0){
			for(k=0;k<dimcol;k++){
				gsl_matrix_get_col(colA,A,k);
				gsl_vector_set(result,k,var_v(colA));
			                      }
		                           }
	               }
	return(0);
}
int vect_maxtrix1r(const gsl_vector *vec, const int length, const int dim, gsl_matrix *lowermx){

  int i,j;
  gsl_permutation *tabp=gsl_permutation_calloc(dim);
  int *addindex = new int[dim]; 
  addindex[0]=0;  
  for(i=1;i<dim;i++){addindex[i]=addindex[i-1]+gsl_permutation_get(tabp,i);}  

  gsl_matrix_set_identity(lowermx);
  for(i=1;i<dim;i++){
	  for(j=0;j<i;j++){
		  gsl_matrix_set(lowermx,i,j,gsl_vector_get(vec,j+addindex[i-1]));
                          }
                      }

gsl_permutation_free(tabp);
delete []addindex;addindex=nullptr;
return 0;
}
int vect_maxtrix1c(const gsl_vector *vec, const int length, const int dim, gsl_matrix *lowermx){
	int dim1=int((1+sqrt(1+length*8.0))/2);
	if(dim!=dim1){printf("Error function(vect_maxtrix1c):the length and dim is conflictive\n");exit(0);}
  int i,j;
  gsl_permutation *tabp=gsl_permutation_calloc(dim-1);
  gsl_permutation_reverse(tabp);
  int *addindex = new int[dim-1]; 
  addindex[0]=0;  
  for(i=1;i<dim-1;i++){addindex[i]=addindex[i-1]+gsl_permutation_get(tabp,i-1);}  
  gsl_matrix_set_identity(lowermx);
  for(j=0;j<dim-1;j++){
	  for(i=j+1;i<dim;i++){
		  gsl_matrix_set(lowermx,i,j,gsl_vector_get(vec,i-1+addindex[j]));
                          }
                      }

gsl_permutation_free(tabp);
delete []addindex;addindex=nullptr;
return 0;
}

int two_vecmul(const gsl_vector *twovec1,const int len1,const gsl_vector *twovec2,const int len2,gsl_matrix *maxtwo){
    int i,j;
    double temptwo=0.0;
    for(i=0;i<len1;i++){
	    for(j=0;j<len2;j++){
		    temptwo=gsl_vector_get(twovec1,i)*gsl_vector_get(twovec2,j);
		    gsl_matrix_set(maxtwo,i,j,temptwo);	
	                     }
                       }
     return 0;
}

double get_det(const gsl_matrix *A){// determinant of A
 if(A->size1!=A->size2){printf("矩阵不是方阵");exit(0);}
  double det=0.0; 
  int n = A->size1;
  gsl_permutation *p = gsl_permutation_calloc(n);
  gsl_matrix *tmpA = gsl_matrix_calloc(n, n);
  int signum;
  gsl_matrix_memcpy(tmpA, A);
  gsl_linalg_LU_decomp(tmpA, p, &signum);
  det = gsl_linalg_LU_det(tmpA, signum);
  gsl_permutation_free(p);
  gsl_matrix_free(tmpA);
  return det;
}

int solve_m(const gsl_matrix *xx,const int row,const int col,gsl_matrix *solvex){
	if(row!=col){printf("xx is not a square matrix");exit(0);}
	int dtem;
	gsl_permutation *b_per=gsl_permutation_alloc (row);
	gsl_matrix *xxtemp=gsl_matrix_calloc(row,col) ;
    gsl_matrix_memcpy(xxtemp,xx);
    gsl_linalg_LU_decomp (xxtemp, b_per, &dtem);//注意,运行该函数后temp_bCV1发生变化
    gsl_linalg_LU_invert(xxtemp, b_per,solvex);//solve1_cb=solve(temp_bCV)
    gsl_matrix_free(xxtemp);
    gsl_permutation_free(b_per);
    return 0;
}

int LinearE_LU(const gsl_matrix *A,const gsl_vector *b,gsl_vector *x0){
	if(A->size1!=A->size2){printf("xx is not a square matrix");exit(0);}//矩阵是方阵，不知道如果不是会怎么样
	int dtem,dim=A->size1;
	gsl_permutation *b_per=gsl_permutation_alloc (dim);
	gsl_matrix *Atemp=gsl_matrix_calloc(dim,dim) ;
    gsl_matrix_memcpy(Atemp,A);
    gsl_linalg_LU_decomp(Atemp, b_per, &dtem);//注意,运行该函数后xxtemp发生变化
    gsl_linalg_LU_solve(Atemp, b_per,b,x0);//solve1_cb=solve(temp_bCV)
    gsl_matrix_free(Atemp);
    gsl_permutation_free(b_per);
    return 0;
}

int Sub_lower(const gsl_matrix *A, gsl_matrix *A_lower){
	if(A->size1!=A->size2){printf("xx is not a square matrix");exit(0);}//矩阵是方阵，不知道如果不是会怎么样
	int i,j,dim1=A->size1;
	gsl_matrix_set_zero(A_lower);
	for(i=0;i<dim1;i++){
		for(j=0;j<=i;j++){gsl_matrix_set(A_lower,i,j,gsl_matrix_get(A,i,j));}	                       }
    return 0;
}

int Sub_upper(const gsl_matrix *A, gsl_matrix *A_upper){
	if(A->size1!=A->size2){printf("xx is not a square matrix");exit(0);}//矩阵是方阵，不知道如果不是会怎么样
	int i,j,dim1=A->size1,dim2=A->size2;
	gsl_matrix_set_zero(A_upper);
	for(i=0;i<dim1;i++){
		for(j=i;j<dim2;j++){gsl_matrix_set(A_upper,i,j,gsl_matrix_get(A,i,j));}	                       }
    return 0;
}




int submatrix(const gsl_matrix *initialmatrix,const int startrow, const int endrow,const int startcol, const int endcol,gsl_matrix *submaxtrix){
/* submatrix of var */
/*	startrow:endrow, startcol:endcol*/
int i,j;
for(i=startrow; i<endrow; i++){for(j=startcol;j<endcol;j++){gsl_matrix_set(submaxtrix,i-startrow,j-startcol,gsl_matrix_get(initialmatrix,i,j));                                                           }
                              }
return 0;
}
int subvec(const gsl_vector *initialvec,const int start,const int end,gsl_vector *subvec){
	int i;
	for(i=start;i<end;i++){gsl_vector_set(subvec,i-start,gsl_vector_get(initialvec,i));}
	return 0;
}
//矩阵以及向量的维数
int numrows_mat(gsl_matrix *A){
    int rows = A->size1;
    return rows;
                            }
int numcols_mat(gsl_matrix *B){
    int cols = B->size2;
    return cols;
 }
int length_vec(gsl_vector *a){
    int len = a->size;
    return len;
 }
int diag_mat(const gsl_vector *v_c,const int lengthv,gsl_matrix *varSigma){
	int i;//varSigma=diag(v_c)
	gsl_matrix_set_zero(varSigma);
	for(i=0;i<lengthv;i++){gsl_matrix_set(varSigma,i,i,gsl_vector_get(v_c,i));}
	return 0;	 
  }

int Expertile_weight(const gsl_vector *y_c,const gsl_vector *x_c,const double tau_expertile,gsl_vector *weight_c){
	if(y_c->size!=x_c->size){printf("the sizes of y and x are not same");exit(0);}
	int i,len = y_c->size;
	gsl_vector_set_all(weight_c,1.0-tau_expertile); 
	for(i=0;i<len;i++){if(gsl_vector_get(y_c,i)>gsl_vector_get(x_c,i))gsl_vector_set(weight_c,i,tau_expertile);}
	return 0;
  }

int mixed_adlassoEx(const int examplebumber,char method[4],const gsl_rng *ccr,const gsl_vector *y, const int N_train, const gsl_matrix *x, const int p, const int q, const int *tabsubject,const double tau, const int n_sampler,const int simk,char distribution[7],double const tau_expertile){
	//int vect_maxtrix1c(const gsl_vector *vec, const int length, const int dim, gsl_matrix *lowermx);//向量变下三角
	int vect_maxtrix1r(const gsl_vector *vec, const int length, const int dim, gsl_matrix *lowermx);//向量变下三角
	int two_vecmul(const gsl_vector *twovec1,const int len1,const gsl_vector *twovec2,const int len2,gsl_matrix *maxtwo);//x_{p*1}%*%t(x_{p*1})
	int save_datam(const gsl_matrix *needsavedm,const int rowm,const int colm,char method[4], char Filename1[4],const float quantile,const int examplebumber, const int iter,char distribution[7]);//save data
	double sum_v(const gsl_vector *vect);//向量求均值
	double rinvGauss1(const gsl_rng *r,double mu, double lambda);//逆高斯分布	
	int rmvnorm_eign(const gsl_rng *r, const gsl_vector *mean, const gsl_matrix *var, gsl_vector *result);//高斯分布
	int solve_m(const gsl_matrix *xx,const int row,const int col,gsl_matrix *solvex);//逆矩阵
	int diag_mat(const gsl_vector *v_c,const int lengthv,gsl_matrix *varSigma);//varSigma=diag(v_c)
	int subvec(const gsl_vector *initialvec,const int start,const int end,gsl_vector *subvec);//subvec=initialvec[start:end]
	//save data
	char paraname1[]={"v_p"};	
	char paraname2[]={"beta_p"};	
	char paraname3[]={"lambda1_p"};	
	char paraname4[]={"d_p"};		
	char paraname5[]={"lambda2_p"};	
	char paraname6[]={"r_p"};		
	char paraname7[]={"lambda3_p"};	
	char paraname8[]={"b_p"};
	char paraname9[]={"MHbeta_c"};
	char paraname10[]={"MHr_c"};	
	char paraname11[]={"MHd_c"};
	
 	char lassoexpEx[]={"lassoexpEx"};
	char adlassoexpEx[]={"adlassoexpEx"};	
	
	const int N_new=N_train+1,Ni=tabsubject[1];//注意这里只能计算平衡的情况
	int iter, i,j,h,k,index1,index2;//n_sampler 迭代次数
	int M_total=0;
    for(i=0;i<N_train;i++){M_total=M_total+tabsubject[i];}//计算总样本个数
//有问题
	int *addsubject=new int[N_new];
    int sub_temp;
    addsubject[0]=0;
	 addsubject[N_train]=M_total;
    for(i=1;i<N_train;i++){
	sub_temp=0;
	for(j=0;j<i;j++){sub_temp=sub_temp+tabsubject[j];}
	                addsubject[i]=sub_temp;
                      }	
	//为了计算r
   gsl_permutation *permu1=gsl_permutation_calloc(q);
   gsl_permutation_init(permu1);
   int *cumsump=new int[q];
	cumsump[0]=0;
// for(i=0;i<q;i++){printf("%d\t",gsl_permutation_get(permu1,i));}
   int temp_cr;
   for(i=1;i<q;i++){
		temp_cr=0;
		for(j=0;j<=i;j++){temp_cr+=gsl_permutation_get(permu1,j);}
		cumsump[i]=temp_cr;
                    }
	//	The parameters with "_c" are the temporary ones that we use for updating.
    //	The parameters with "_p" are the recorded ones.
	//-------Prior
	double c1=0.1,c2=0.1,g1=0.1,g2=0.1;
	    //-分配空间	
	gsl_vector *beta_c=gsl_vector_calloc(p);
	gsl_vector *lambda1_c=gsl_vector_calloc(p);
	gsl_vector *ts_c=gsl_vector_calloc(p);

	gsl_vector *d_c=gsl_vector_calloc(q);
	gsl_vector *lambda2_c=gsl_vector_calloc(q);
//	gsl_vector *etas_c=gsl_vector_calloc(q);

	int const dim_r=int(q*(q-1)/2);
	gsl_vector *r_c=gsl_vector_calloc(dim_r);
	gsl_vector *lambda3_c=gsl_vector_calloc(dim_r);
	gsl_vector *hs_c=gsl_vector_calloc(dim_r);
	
	gsl_matrix *GA_c=gsl_matrix_calloc(q,q);
	gsl_matrix *b_c=gsl_matrix_calloc(N_train,q);
	gsl_vector *v_c=gsl_vector_calloc(M_total);
	

	     //-初始化--注意修改初始值，beta初值不能为0
	double scale_c=1.0;
	gsl_vector_set_all(beta_c,1.0);
	gsl_vector_set_all(lambda1_c,1.0);

	gsl_vector_set_all(ts_c,1.0);
	gsl_vector_set_all(v_c,1.0);

	gsl_vector_set_all(d_c,1.0);
	gsl_vector_set_all(lambda2_c,1.0);
	//gsl_vector_set_all(etas_c,1.0);

	gsl_vector_set_all(r_c,1.0);
	gsl_vector_set_all(lambda3_c,1.0);
	gsl_vector_set_all(hs_c,1.0);		
	
	for(i=0;i<N_train;i++){for(j=0;j<q;j++)gsl_matrix_set(b_c,i,j,gsl_ran_ugaussian(ccr));}
	
	vect_maxtrix1r(r_c,dim_r,q,GA_c);	
	//-------Iteration records
	gsl_matrix *v_p=gsl_matrix_calloc(n_sampler,M_total);
	gsl_vector *scale_p=gsl_vector_calloc(n_sampler);
	gsl_matrix *beta_p=gsl_matrix_calloc(n_sampler,p);
	gsl_matrix *lambda1_p=gsl_matrix_calloc(n_sampler,p);

	gsl_matrix *d_p=gsl_matrix_calloc(n_sampler,q);
	gsl_matrix *lambda2_p=gsl_matrix_calloc(n_sampler,q);

	gsl_matrix *r_p=gsl_matrix_calloc(n_sampler,dim_r);
	gsl_matrix *lambda3_p=gsl_matrix_calloc(n_sampler,dim_r);

	gsl_matrix *b_p=gsl_matrix_calloc(n_sampler,q*N_train);
	//-------Iteration records end
	
	//temp(i)_calcu(j) -----计算参数（j）时的第（i）个暂时值	
	gsl_matrix *z=gsl_matrix_calloc(M_total,q);
	gsl_matrix_memcpy(z,x);
	gsl_vector *row_x=gsl_vector_calloc(p);
	gsl_vector *row_z=gsl_vector_calloc(q);
	gsl_vector *row_b=gsl_vector_calloc(q);

	// Prepare for v
	double ingaus_temp;//逆高斯分布的随机数
	double temp_nu=0.0,temp_lambda=0.0;
	double dtemp1_cv=0.0,dtemp2_cv=0.0;
	gsl_vector *vtemp1_cv=gsl_vector_calloc(q);
	gsl_vector *vtemp2_cv=gsl_vector_calloc(q);	
	gsl_matrix *varV=gsl_matrix_calloc(Ni,Ni);	
	
	// ---Prepare for lambda
	double temp_shape=0.0,temp_scale=0.0;
	// ---Prepare for bi	     
    gsl_matrix *temp_bCV=gsl_matrix_calloc(q,q);
    gsl_vector *temp_bMU=gsl_vector_calloc(q);
	gsl_matrix *zid_cbi=gsl_matrix_calloc(Ni,q);
	gsl_vector *zdtempc_cbi=gsl_vector_calloc(Ni);
	gsl_matrix *zidGA_cbi=gsl_matrix_calloc(Ni,q);

	int index1_cbi;
	double temp1_cbi,temp2_cbi;	
	gsl_vector *bi_c=gsl_vector_calloc(q);
	gsl_matrix *solve1_cbi=gsl_matrix_calloc(q,q);
    gsl_vector *temp_bMU1=gsl_vector_calloc(q);

    gsl_matrix *mtemp1_cb=gsl_matrix_calloc(q,q);
	double dtemp1_cb=0.0;
	
	gsl_permutation *b_per=gsl_permutation_alloc (q);
	gsl_matrix *solve1_cb=gsl_matrix_calloc(q,q);
	gsl_matrix *mtemp2_cb=gsl_matrix_calloc(q,q);

	
	//////计算Pb_cbi和Qnew_cbi
	double ywy_cbi, biIbi_cbi,Pb_cbi,Qnew_cbi,CVbdet_cbi;
	gsl_matrix *WzidGA_cbi=gsl_matrix_calloc(Ni,q);			  
	gsl_vector *zidGAbi_cbi=gsl_vector_calloc(Ni);
	gsl_vector *subyi2_cbi=gsl_vector_calloc(Ni);
	gsl_vector *xbSzdbi_cbi=gsl_vector_calloc(Ni);
	gsl_vector *WEitemp2_cbi=gsl_vector_calloc(Ni);
	gsl_vector *WzidGAr_cbi=gsl_vector_calloc(q);
	gsl_vector *subyi1_cbi=gsl_vector_calloc(Ni);
	gsl_vector *rowbNew_c=gsl_vector_calloc(q);
	gsl_vector *rowbNewQn_c=gsl_vector_calloc(q);
	gsl_vector *rowbNewCV_cbi=gsl_vector_calloc(q);
    ///////////////////计算Pnew_cbi和Qb_cbi
	double Pnew_cbi,Qb_cbi,CVnewdet_cbi,alpha_cbi,unif_cbi;
	gsl_matrix *WnewzidGA_cbi=gsl_matrix_calloc(Ni,q);
	gsl_vector *WEitemp2New_cbi=gsl_vector_calloc(Ni);
	gsl_vector *WnewzidGAr_cbi=gsl_vector_calloc(q);
	gsl_matrix *temp_bCVNew=gsl_matrix_calloc(q,q);
	gsl_matrix *solve1_cbiNew=gsl_matrix_calloc(q,q);
    gsl_vector *temp_bMUNew=gsl_vector_calloc(q);		
	gsl_vector *temp_bMU1New=gsl_vector_calloc(q);				   
	gsl_vector *rowbQn_c=gsl_vector_calloc(q);
	gsl_vector *rowbCV_cbi=gsl_vector_calloc(q);
	// end bi

    //---Prepare for d
	double d_sigma,d_mu,temp1_cd,temp2_cd,temp3_cd,temp4_cd,ck_cd,dk_cd,trancated_cd;
	gsl_vector *rbi_cd=gsl_vector_calloc(q);
	gsl_matrix *Gi_cd=gsl_matrix_calloc(Ni,q);
	gsl_vector *Gic_cd=gsl_vector_calloc(Ni);
	gsl_vector *zidrbi_cd=gsl_vector_calloc(Ni);
	gsl_matrix *zid_cd=gsl_matrix_calloc(Ni,q);
	gsl_vector *zdtempc_cd=gsl_vector_calloc(Ni);
	gsl_vector *Gbitemp_cd=gsl_vector_calloc(q);
	gsl_vector *Gic1_cd=gsl_vector_calloc(Ni);   		 
   ///计算Pb_cd和Qnew_cd
	double ywy_cd,dkNew_cd,Pb_cd,Qnew_cd,CVbdet_cd;
   gsl_vector *zidrbiSxb_cd=gsl_vector_calloc(Ni);
   gsl_vector *subyi2_cd=gsl_vector_calloc(Ni);
   gsl_vector *subyi3_cd=gsl_vector_calloc(Ni);
   gsl_vector *WEitemp2_cd=gsl_vector_calloc(Ni);
   gsl_vector *WGir_cd=gsl_vector_calloc(q);
   gsl_vector *dNew_cd=gsl_vector_calloc(q);
   ///计算Pnew_cd和Qb_cd
   double d_sigmaNew,d_muNew,ywy_cdNew,Pnew_cd,Qb_cd,CVnewdet_cd,alpha_cd,unif_cd;
   gsl_vector *WEitemp2New_cd=gsl_vector_calloc(Ni);

	 //---Prepare for r
   
    gsl_matrix *temp_rCV=gsl_matrix_calloc(dim_r,dim_r);
	gsl_vector *temp_rMU=gsl_vector_calloc(dim_r);
	gsl_matrix *zid=gsl_matrix_calloc(Ni,q);
	gsl_vector *zdtempc_cr=gsl_vector_calloc(Ni);
	gsl_matrix *zdtemp=gsl_matrix_calloc(Ni,q);
    gsl_matrix *Fir=gsl_matrix_calloc(Ni,dim_r);
	gsl_vector *zdbitemp_cr=gsl_vector_calloc(Ni);
	gsl_vector *xibeta=gsl_vector_calloc(Ni);	

    double dtemp1_cr;
	gsl_matrix *solve1_cr=gsl_matrix_calloc(dim_r,dim_r);
	gsl_vector *temp_rMU1=gsl_vector_calloc(dim_r); 
	gsl_matrix *mtemp2_cr=gsl_matrix_calloc(dim_r,dim_r);

			//计算Pb_cr、Qnew_cr	 
	double ywy_cr,temp1_cr,temp2_cr;	
	double beDtbe_cr,Pb_cr,Qnew_cr,CVbdet_cr;	
	gsl_vector *FirMr_cr=gsl_vector_calloc(Ni);
	gsl_vector *subyi2_cr=gsl_vector_calloc(Ni);
	gsl_vector *xbSzdbiSFr_cr=gsl_vector_calloc(Ni);
	gsl_vector *xbSzdbi_cr=gsl_vector_calloc(Ni);
	gsl_vector *WEitemp2_cr=gsl_vector_calloc(Ni);
	gsl_matrix *WiFir_cr=gsl_matrix_calloc(Ni,dim_r);
	gsl_vector *WiFirRow_cr=gsl_vector_calloc(dim_r);
	gsl_matrix *FWFtemp_cr=gsl_matrix_calloc(dim_r,dim_r);
	gsl_vector *subyi1_cr=gsl_vector_calloc(Ni);
	gsl_vector *FWytemp_cr=gsl_vector_calloc(dim_r);
	gsl_vector *rNew_c=gsl_vector_calloc(dim_r);	
	gsl_vector *rNewQn_c=gsl_vector_calloc(dim_r);
	gsl_vector *rNewCV_cbe=gsl_vector_calloc(dim_r);
	//计算Pnew_cr、Qb_cr
	double Pnew_cr,CVnewdet_cr,alpha_cr,unif_cr,Qb_cr;
	gsl_matrix *temp_rCVNew=gsl_matrix_calloc(dim_r,dim_r);
	gsl_vector *temp_rMUNew=gsl_vector_calloc(dim_r);
	gsl_vector *WEitemp2New_cr=gsl_vector_calloc(Ni);



	//--Prepare for scale
	double temp_sshape,temp_srate,temp_sscale;
    gsl_vector *vtemp2_csc=gsl_vector_calloc(q);
    gsl_vector *vtemp1_csc=gsl_vector_calloc(q);
	 //---Prepare for beta
	double Pb_cbe,Qnew_cbe,CVbdet_cbe;
	double temp1_cbe,temp2_cbe,ywy_cbe=0,beDtbe_cbe,Pnew_cbe,Qb_cbe,CVnewdet_cbe;
	gsl_matrix *temp_betaCV=gsl_matrix_calloc(p,p);
    gsl_vector *temp_betaMU=gsl_vector_calloc(p);
	gsl_matrix *temp_betaCVNEW=gsl_matrix_calloc(p,p);
    gsl_vector *temp_betaMUNEW=gsl_vector_calloc(p);
	gsl_matrix *subxi=gsl_matrix_calloc(Ni,p);
	gsl_vector *subyi=gsl_vector_calloc(Ni);
	gsl_matrix *vixi=gsl_matrix_calloc(Ni,q);
	gsl_matrix *subzi=gsl_matrix_calloc(Ni,q);
	
		gsl_matrix *Wixi_cbe=gsl_matrix_calloc(Ni,p);
	gsl_vector *Wixir_cbe=gsl_vector_calloc(p);
	gsl_matrix *xWx_cbe=gsl_matrix_calloc(p,p);
	gsl_vector *subyi1_cbe=gsl_vector_calloc(Ni);	
	gsl_vector *xWytemp_cbe	=gsl_vector_calloc(p);

	gsl_vector *subyi2_cbe=gsl_vector_calloc(Ni);
	gsl_vector *xbSzdbi_cbe=gsl_vector_calloc(Ni);
	gsl_vector *WEitemp2_cbe=gsl_vector_calloc(Ni);
	gsl_vector *WEitemp2New_cbe=gsl_vector_calloc(Ni);
	gsl_vector *betaNew_c=gsl_vector_calloc(p);
	gsl_vector *betaNewQn_c=gsl_vector_calloc(p);
	gsl_vector *betaNewCV_cbe=gsl_vector_calloc(p);
	double alpha_cbe,unif_cbe;
	
	gsl_vector *zdtempc_cbe=gsl_vector_calloc(Ni);
	gsl_vector *Gbitemp_cbe=gsl_vector_calloc(q);
	
	gsl_vector *zdrbitemp_cbe=gsl_vector_calloc(Ni);	
	gsl_matrix *solve1_cbe=gsl_matrix_calloc(p,p);
	gsl_vector *temp_betaMU1=gsl_vector_calloc(p); 	
    double dtemp1_cbe;	

	gsl_matrix *MHbeta_c=gsl_matrix_calloc(n_sampler,1);	
	gsl_matrix *MHr_c=gsl_matrix_calloc(n_sampler,1);	
	gsl_matrix *MHd_c=gsl_matrix_calloc(n_sampler,q);
	gsl_matrix_set_all(MHbeta_c,0.0);
	gsl_matrix_set_all(MHr_c,0.0);
	gsl_matrix_set_all(MHd_c,0.0);

	for(iter=0;iter<n_sampler;iter++){
		if((iter+1)%5000==0){printf("This is step %d\n",iter+1);}
		
//------The full conditional for r 	//计算Pb_cr、Qnew_cr	 
		  gsl_matrix_set_all(temp_rCV,0.0);
	      gsl_vector_set_all(temp_rMU,0.0); 
		  ywy_cr=0.0;
		  for(i=0;i<N_train;i++){
			  index1=addsubject[i],index2=addsubject[i+1];				
			  submatrix(z,index1,index2,0,q,subzi);	
			  gsl_matrix_memcpy(zid,subzi);	  

			  for(j=0;j<q;j++){gsl_matrix_get_col(zdtempc_cr,zid,j);			                    
				               gsl_vector_scale(zdtempc_cr,gsl_vector_get(d_c,j));
							   gsl_matrix_set_col(zid,j,zdtempc_cr);
			                   }
			  gsl_matrix_get_row(row_b,b_c,i);	
			 gsl_matrix_memcpy(zdtemp,zid);
		      for(k=0;k<q-1;k++){
			     for(j=0,h=cumsump[k];j<(k+1),h<cumsump[k+1];j++,h++){
					                                                  gsl_matrix_get_col(zdtempc_cr,zdtemp,k+1);
			                                                          gsl_vector_scale(zdtempc_cr,gsl_vector_get(row_b,j));
																      gsl_matrix_set_col(Fir,h,zdtempc_cr);
		                                                              }
				                                                   
	                            }
	
               
              gsl_blas_dgemv(CblasNoTrans,1.0,Fir,r_c,0.0,FirMr_cr);	
			   
		      submatrix(x,index1,index2,0,p,subxi);	
			  gsl_blas_dgemv(CblasNoTrans,1.0,subxi,beta_c,0.0,xibeta);
	          gsl_blas_dgemv(CblasNoTrans,1.0,zid,row_b,0.0,zdbitemp_cr);		

			  gsl_vector_memcpy(xbSzdbi_cr,xibeta);
			  gsl_vector_add(xbSzdbi_cr,zdbitemp_cr);
			  gsl_vector_memcpy(xbSzdbiSFr_cr,xbSzdbi_cr);// 计算均值xbSzdbi_cr
			  gsl_vector_add(xbSzdbiSFr_cr,FirMr_cr);

			   subvec(y,index1,index2,subyi);
			  Expertile_weight(subyi,xbSzdbiSFr_cr,tau_expertile,WEitemp2_cr);//计算权重
			  subvec(y,index1,index2,subyi2_cr);
			  gsl_vector_sub(subyi2_cr,xbSzdbiSFr_cr);//	计算Pb_cr 
			   
			  gsl_matrix_memcpy(WiFir_cr,Fir);//计算抽取暂时新样本的后验分布
			  for(j=0;j<Ni;j++){gsl_matrix_get_row(WiFirRow_cr,WiFir_cr,j);			                    
				                 gsl_vector_scale(WiFirRow_cr,gsl_vector_get(WEitemp2_cr,j));
							     gsl_matrix_set_row(WiFir_cr,j,WiFirRow_cr);
			                     }

 
			  gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0,Fir,WiFir_cr,0.0,FWFtemp_cr);
			  gsl_matrix_add(temp_rCV,FWFtemp_cr);

			  //下面开始计算temp_rMU	
			  subvec(y,index1,index2,subyi1_cr);
			  gsl_vector_sub(subyi1_cr,xbSzdbi_cr);
			  gsl_blas_dgemv(CblasTrans,1.0,WiFir_cr,subyi1_cr,0.0,FWytemp_cr);			 			 
			  gsl_vector_add(temp_rMU,FWytemp_cr); 
              //////计算后验分布 计算Pb_cbe 
			   temp2_cr=0.0;
			  for(j=0;j<Ni;j++){temp1_cr=gsl_vector_get(subyi2_cr,j);
			        	temp2_cr=temp2_cr+temp1_cr*temp1_cr*gsl_vector_get(WEitemp2_cr,j);	
				                 }
			   ywy_cr=ywy_cr+temp2_cr;


		                       }  // end i
		//修改了
		  gsl_matrix_scale(temp_rCV,2.0*scale_c);
          for(h=0;h<dim_r;h++){dtemp1_cr=gsl_matrix_get(temp_rCV,h,h)+1.0/(gsl_vector_get(hs_c,h));gsl_matrix_set(temp_rCV,h,h,dtemp1_cr);}		  		
		  gsl_vector_scale(temp_rMU,2.0*scale_c);
           
		//
          solve_m(temp_rCV,dim_r,dim_r,solve1_cr);
          gsl_blas_dgemv(CblasNoTrans,1.0,solve1_cr,temp_rMU,0.0,temp_rMU1);//temp_bMU1=solve1_cb%*%temp_bMU       
		 
		  rmvnorm_eigen(ccr,temp_rMU1,solve1_cr,rNew_c);//产生多元正态随机数	
        ////////////////////////////////// Pb_cr
	  
		  beDtbe_cr=0.0;
		 for(h=0;h<dim_r;h++){
			     beDtbe_cr=beDtbe_cr+gsl_vector_get(r_c,h)*gsl_vector_get(r_c,h)/gsl_vector_get(hs_c,h)/2.0;
			                 }

			 Pb_cr=scale_c*ywy_cr+beDtbe_cr;

			 gsl_vector_memcpy(rNewQn_c,rNew_c);
			 gsl_vector_sub(rNewQn_c,temp_rMU1);

			 gsl_blas_dgemv(CblasNoTrans,1.0,temp_rCV,rNewQn_c,0.0,rNewCV_cbe);			
			 gsl_blas_ddot(rNewQn_c,rNewCV_cbe,&Qnew_cr);
			 Qnew_cr=Qnew_cr/2.0;	 

			 CVbdet_cr=sqrt(get_det(temp_rCV));// 对吗

//计算Pnew_cr、Qb_cr
	// 注意主要不同的地方在 r_c---rNew_c 和	WEitemp2_cr-WEitemp2New_cr	

          gsl_matrix_set_all(temp_rCVNew,0.0);
	      gsl_vector_set_all(temp_rMUNew,0.0); 
		  ywy_cr=0.0;
		  for(i=0;i<N_train;i++){
			  index1=addsubject[i],index2=addsubject[i+1];				
			  submatrix(z,index1,index2,0,q,subzi);			 
			  	
			  gsl_matrix_memcpy(zid,subzi);	  

			  for(j=0;j<q;j++){gsl_matrix_get_col(zdtempc_cr,zid,j);			                    
				               gsl_vector_scale(zdtempc_cr,gsl_vector_get(d_c,j));
							   gsl_matrix_set_col(zid,j,zdtempc_cr);
			                   }
              gsl_matrix_get_row(row_b,b_c,i);
			  gsl_matrix_memcpy(zdtemp,zid);
		      for(k=0;k<q-1;k++){
			     for(j=0,h=cumsump[k];j<(k+1),h<cumsump[k+1];j++,h++){
					                                                  gsl_matrix_get_col(zdtempc_cr,zdtemp,k+1);
			                                                          gsl_vector_scale(zdtempc_cr,gsl_vector_get(row_b,j));
																      gsl_matrix_set_col(Fir,h,zdtempc_cr);
		                                                              }
				                                                   
	                            }
	
               
              gsl_blas_dgemv(CblasNoTrans,1.0,Fir,rNew_c,0.0,FirMr_cr);//计算Pnew_cr以及均值， rNew_c与MH分母不同 

		      submatrix(x,index1,index2,0,p,subxi);	
			  gsl_blas_dgemv(CblasNoTrans,1.0,subxi,beta_c,0.0,xibeta);
	          gsl_blas_dgemv(CblasNoTrans,1.0,zid,row_b,0.0,zdbitemp_cr);		

			  gsl_vector_memcpy(xbSzdbi_cr,xibeta);
			  gsl_vector_add(xbSzdbi_cr,zdbitemp_cr);
			  gsl_vector_memcpy(xbSzdbiSFr_cr,xbSzdbi_cr);// 计算均值xbSzdbi_cr
			  gsl_vector_add(xbSzdbiSFr_cr,FirMr_cr);
			   subvec(y,index1,index2,subyi);
			  Expertile_weight(subyi,xbSzdbiSFr_cr,tau_expertile,WEitemp2New_cr);//计算权重WEitemp2New_cr	与MH分母不同
			  subvec(y,index1,index2,subyi2_cr);
			  gsl_vector_sub(subyi2_cr,xbSzdbiSFr_cr);//	计算Pnew_cr 
			   
			  gsl_matrix_memcpy(WiFir_cr,Fir);//计算Qb_cr
			  for(j=0;j<Ni;j++){gsl_matrix_get_row(WiFirRow_cr,WiFir_cr,j);			                    
				                 gsl_vector_scale(WiFirRow_cr,gsl_vector_get(WEitemp2New_cr,j));//计算权重WEitemp2New_cr	与MH分母不同
							     gsl_matrix_set_row(WiFir_cr,j,WiFirRow_cr);
			                     }

 
			  gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0,Fir,WiFir_cr,0.0,FWFtemp_cr);
			  gsl_matrix_add(temp_rCVNew,FWFtemp_cr);//temp_rCVNew

			   //下面开始计算temp_rMUNew	
			  subvec(y,index1,index2,subyi1_cr);
			  gsl_vector_sub(subyi1_cr,xbSzdbi_cr);
			  gsl_blas_dgemv(CblasTrans,1.0,WiFir_cr,subyi1_cr,0.0,FWytemp_cr);			 			 
			  gsl_vector_add(temp_rMUNew,FWytemp_cr); //temp_rMUNew
              //////计算后验分布 计算Pnew_cbe 
			   temp2_cr=0.0;
			  for(j=0;j<Ni;j++){temp1_cr=gsl_vector_get(subyi2_cr,j);
			        	temp2_cr=temp2_cr+temp1_cr*temp1_cr*gsl_vector_get(WEitemp2New_cr,j);	//权重WEitemp2New_c与MH分母不同
				                 }
			   ywy_cr=ywy_cr+temp2_cr;

		                       }  // end i
		//修改了
		  gsl_matrix_scale(temp_rCVNew,2.0*scale_c);
          for(h=0;h<dim_r;h++){dtemp1_cr=gsl_matrix_get(temp_rCVNew,h,h)+1.0/(gsl_vector_get(hs_c,h));gsl_matrix_set(temp_rCVNew,h,h,dtemp1_cr);}//temp_rMUNew
		    gsl_vector_scale(temp_rMUNew,2.0*scale_c);
        //

          solve_m(temp_rCVNew,dim_r,dim_r,solve1_cr);//temp_rMUNew
          gsl_blas_dgemv(CblasNoTrans,1.0,solve1_cr,temp_rMUNew,0.0,temp_rMU1);//temp_bMU1=solve1_cb%*%temp_bMUNew   
	  
		  beDtbe_cr=0.0;
		 for(h=0;h<dim_r;h++){
			     beDtbe_cr=beDtbe_cr+gsl_vector_get(rNew_c,h)*gsl_vector_get(rNew_c,h)/gsl_vector_get(hs_c,h)/2.0;
			                 }

			 Pnew_cr=scale_c*ywy_cr+beDtbe_cr;
			 
			 gsl_vector_memcpy(rNewQn_c,r_c);//注意与分母的rNew_c不同
			 gsl_vector_sub(rNewQn_c,temp_rMU1);

			 gsl_blas_dgemv(CblasNoTrans,1.0,temp_rCVNew,rNewQn_c,0.0,rNewCV_cbe);	//temp_rCVNew		
			 gsl_blas_ddot(rNewQn_c,rNewCV_cbe,&Qb_cr);
			 Qb_cr=Qb_cr/2.0;	 

			 CVnewdet_cr=sqrt(get_det(temp_rCVNew));// 对吗

			 alpha_cr=(CVnewdet_cr/CVbdet_cr)*exp(-Pnew_cr-Qb_cr+Pb_cr+Qnew_cr);			
             unif_cr=gsl_ran_flat(ccr,0.0,1.0);
             if(unif_cr<=alpha_cr){gsl_vector_memcpy(r_c,rNew_c);	
			         gsl_matrix_set(MHr_c,iter,0,1.0);			 
			                      }	
		  gsl_matrix_set_row(r_p,iter,r_c); 
		  vect_maxtrix1r(r_c, dim_r, q, GA_c);

		
	//------The full conditional for d
		 for(k=0;k<q;k++){
		 d_sigma=0.0,d_mu=0.0,ywy_cd=0.0;
		 for(i=0;i<N_train;i++){
			  index1=addsubject[i],index2=addsubject[i+1];
			  submatrix(z,index1,index2,0,q,subzi);
			  gsl_matrix_get_row(row_b,b_c,i);
			  gsl_blas_dgemv(CblasNoTrans,1.0,GA_c,row_b,0.0,rbi_cd);
			  gsl_matrix_memcpy(Gi_cd,subzi);  

			  for(j=0;j<q;j++){gsl_matrix_get_col(Gic_cd,Gi_cd,j);			                    
				               gsl_vector_scale(Gic_cd,gsl_vector_get(rbi_cd,j));
							   gsl_matrix_set_col(Gi_cd,j,Gic_cd);
			                   }
              
			  gsl_matrix_get_col(Gic_cd,Gi_cd,k);			
			  submatrix(x,index1,index2,0,p,subxi);	
			  gsl_blas_dgemv(CblasNoTrans,1.0,subxi,beta_c,0.0,xibeta);
	          gsl_blas_dgemv(CblasNoTrans,1.0,Gi_cd,d_c,0.0,zidrbi_cd);	  
			  gsl_vector_memcpy(zidrbiSxb_cd,xibeta);

			  gsl_vector_add(zidrbiSxb_cd,zidrbi_cd);
			  subvec(y,index1,index2,subyi);
			  gsl_vector_memcpy(subyi2_cd,subyi);
			  Expertile_weight(subyi,zidrbiSxb_cd,tau_expertile,WEitemp2_cd);//计算权重	


			  gsl_vector_sub(subyi2_cd,zidrbiSxb_cd);//	计算Pb_cd 
			  gsl_vector_memcpy(subyi3_cd,subyi2_cd);//	计算均值
			  temp1_cd=0.0;			  
			  for(j=0;j<Ni;j++){ temp2_cd=gsl_vector_get(Gic_cd,j);
			                     temp3_cd=gsl_vector_get(WEitemp2_cd,j);
								 temp1_cd=temp1_cd+temp2_cd*temp2_cd*temp3_cd;
			                    }
			  d_sigma=d_sigma+temp1_cd;
			  //下面是均值	

			  gsl_matrix_get_col(Gic1_cd,Gi_cd,k);
			  gsl_vector_scale(Gic1_cd,gsl_vector_get(d_c,k));
			  gsl_vector_add(subyi3_cd,Gic1_cd);
			  
			  temp1_cd=0.0;			  
			  for(j=0;j<Ni;j++){ temp2_cd=gsl_vector_get(Gic_cd,j);
			                     temp3_cd=gsl_vector_get(WEitemp2_cd,j);
								 temp4_cd=gsl_vector_get(subyi3_cd,j);
								 temp1_cd=temp1_cd+temp2_cd*temp3_cd*temp4_cd;
			                    }
			  d_mu=d_mu+temp1_cd;		 


			  //////计算后验分布 计算Pb_cd 
		  
			  temp2_cd=0.0;
			  for(j=0;j<Ni;j++){temp1_cd=gsl_vector_get(subyi2_cd,j);
			        	temp2_cd=temp2_cd+temp1_cd*temp1_cd*gsl_vector_get(WEitemp2_cd,j);	
				                 }
			  ywy_cd=ywy_cd+temp2_cd;  
	 
		                       } // end i
		 // Pb_cd
		 Pb_cd=	scale_c*ywy_cd+gsl_vector_get(lambda2_c,k)*gsl_vector_get(d_c,k);
	 //修改了
		 d_sigma=d_sigma*(2*scale_c);	
     //
		 d_sigma=1.0/d_sigma;
	 //修改了
		 d_mu=d_mu*(2*scale_c);
		 d_mu=d_sigma*(d_mu-gsl_vector_get(lambda2_c,k));	
		 //
		 ck_cd=-d_mu/sqrt(d_sigma);
		 trancated_cd=gsl_ran_gaussian_tail(ccr,ck_cd,1.0);
		 dkNew_cd=sqrt(d_sigma)*trancated_cd+d_mu;
		 gsl_vector_memcpy(dNew_cd,d_c);
		 gsl_vector_set(dNew_cd,k,dkNew_cd);

		 temp1_cd=gsl_vector_get(dNew_cd,k)-d_mu;
		 Qnew_cd=temp1_cd*temp1_cd/d_sigma/2.0;
		 CVbdet_cd=1.0/sqrt(d_sigma);// 对吗
		 CVbdet_cd=CVbdet_cd/gsl_cdf_gaussian_P(d_mu/sqrt(d_sigma),1.0);//gsl_cdf_gaussian_P是表示P（Z<=z）吗一会验证一下

//////////////////////////////////////// 计算Pnew_cd和Qb_cd
         d_sigmaNew=0.0,d_muNew=0.0,ywy_cdNew=0.0;
		 for(i=0;i<N_train;i++){
			  index1=addsubject[i],index2=addsubject[i+1];
			  submatrix(z,index1,index2,0,q,subzi);
			  gsl_matrix_get_row(row_b,b_c,i);
			  gsl_blas_dgemv(CblasNoTrans,1.0,GA_c,row_b,0.0,rbi_cd);
			  gsl_matrix_memcpy(Gi_cd,subzi);  

			  for(j=0;j<q;j++){gsl_matrix_get_col(Gic_cd,Gi_cd,j);			                    
				               gsl_vector_scale(Gic_cd,gsl_vector_get(rbi_cd,j));
							   gsl_matrix_set_col(Gi_cd,j,Gic_cd);
			                   }
              
			  gsl_matrix_get_col(Gic_cd,Gi_cd,k);
			
			  submatrix(x,index1,index2,0,p,subxi);	
			  gsl_blas_dgemv(CblasNoTrans,1.0,subxi,beta_c,0.0,xibeta);
	          gsl_blas_dgemv(CblasNoTrans,1.0,Gi_cd,dNew_cd,0.0,zidrbi_cd);	  ///不是d_c
			  gsl_vector_memcpy(zidrbiSxb_cd,xibeta);

			  gsl_vector_add(zidrbiSxb_cd,zidrbi_cd);
			  subvec(y,index1,index2,subyi);
			  gsl_vector_memcpy(subyi2_cd,subyi);
			  Expertile_weight(subyi,zidrbiSxb_cd,tau_expertile,WEitemp2New_cd);//计算权重	与WEitemp2New_cd不同


			  gsl_vector_sub(subyi2_cd,zidrbiSxb_cd);//	计算Pnew_cd 
			  gsl_vector_memcpy(subyi3_cd,subyi2_cd);//	计算均值
			  temp1_cd=0.0;			  
			  for(j=0;j<Ni;j++){ temp2_cd=gsl_vector_get(Gic_cd,j);
			                     temp3_cd=gsl_vector_get(WEitemp2New_cd,j);//计算权重	与WEitemp2New_cd不同
								 temp1_cd=temp1_cd+temp2_cd*temp2_cd*temp3_cd;
			                    }
			  d_sigmaNew=d_sigmaNew+temp1_cd;
			  //下面是均值	

			  gsl_matrix_get_col(Gic1_cd,Gi_cd,k);
			  gsl_vector_scale(Gic1_cd,gsl_vector_get(dNew_cd,k));///不是d_c
			  gsl_vector_add(subyi3_cd,Gic1_cd);
			  
			  temp1_cd=0.0;			  
			  for(j=0;j<Ni;j++){ temp2_cd=gsl_vector_get(Gic_cd,j);
			                     temp3_cd=gsl_vector_get(WEitemp2New_cd,j);//计算权重	与WEitemp2New_cd不同
								 temp4_cd=gsl_vector_get(subyi3_cd,j);
								 temp1_cd=temp1_cd+temp2_cd*temp3_cd*temp4_cd;
			                    }
			  d_muNew=d_muNew+temp1_cd;		 


			  //////计算后验分布 计算Pnew_cd 
		  
			  temp2_cd=0.0;
			  for(j=0;j<Ni;j++){temp1_cd=gsl_vector_get(subyi2_cd,j);
			        	temp2_cd=temp2_cd+temp1_cd*temp1_cd*gsl_vector_get(WEitemp2New_cd,j);	//计算权重	与WEitemp2New_cd不同
				                 }
			  ywy_cdNew=ywy_cdNew+temp2_cd;


			               } // end i
		  // Pnew_cd
		 Pnew_cd=scale_c*ywy_cdNew+gsl_vector_get(lambda2_c,k)*gsl_vector_get(dNew_cd,k);
 //修改了
		 d_sigmaNew=d_sigmaNew*(2*scale_c);	
     //
		 d_sigmaNew=1.0/d_sigmaNew;
 //修改了
		 d_muNew=d_muNew*(2*scale_c);
		 d_muNew=d_sigmaNew*(d_muNew-gsl_vector_get(lambda2_c,k));
//
		 temp1_cd=gsl_vector_get(d_c,k)-d_muNew;
		 Qb_cd=temp1_cd*temp1_cd/d_sigmaNew/2.0;
		 CVnewdet_cd=1.0/sqrt(d_sigmaNew);// 对吗
		 CVnewdet_cd=CVnewdet_cd/gsl_cdf_gaussian_P(d_muNew/sqrt(d_sigmaNew),1.0);			 		 
	     alpha_cd=(CVnewdet_cd/CVbdet_cd)*exp(-Pnew_cd-Qb_cd+Pb_cd+Qnew_cd);// 对吗
         unif_cd=gsl_ran_flat(ccr,0.0,1.0);
               if(unif_cd<=alpha_cd){gsl_vector_set(d_c,k,dkNew_cd);
			      gsl_matrix_set(MHd_c,iter,k,1.0);
			                         }	
		 
	                  }  

					  
	    gsl_matrix_set_row(d_p,iter,d_c);  				 
			 	
		
		
		//---------------------------The full conditional for beta
		
		 gsl_matrix_set_all(temp_betaCV,0.0);
         gsl_vector_set_all(temp_betaMU,0.0);

		 ywy_cbe=0;
		 for(i=0;i<N_train;i++){			 
			  index1=addsubject[i],index2=addsubject[i+1];  			  	 
	    	  submatrix(z,index1,index2,0,q,subzi);
		      gsl_matrix_memcpy(zid,subzi);	         
			  for(j=0;j<q;j++){gsl_matrix_get_col(zdtempc_cbe,zid,j);			                    
				               gsl_vector_scale(zdtempc_cbe,gsl_vector_get(d_c,j));
							   gsl_matrix_set_col(zid,j,zdtempc_cbe);
			                   }	

			   gsl_matrix_get_row(row_b,b_c,i); 
		       gsl_blas_dgemv(CblasNoTrans,1.0,GA_c,row_b,0.0,Gbitemp_cbe);
			   gsl_blas_dgemv(CblasNoTrans,1.0,zid,Gbitemp_cbe,0.0,zdrbitemp_cbe);
			   //printf("zdrbitemp_cbe0=%f\t",gsl_vector_get(zdrbitemp_cbe,0));
			   //printf("zdrbitemp_cbe1=%f\t",gsl_vector_get(zdrbitemp_cbe,1));		
			   submatrix(x,index1,index2,0,p,subxi);
			   gsl_blas_dgemv(CblasNoTrans,1.0,subxi,beta_c,0.0,xibeta);	

			   gsl_vector_memcpy(xbSzdbi_cbe,xibeta);
			   gsl_vector_add(xbSzdbi_cbe,zdrbitemp_cbe);	
			   subvec(y,index1,index2,subyi);	
			    Expertile_weight(subyi,xbSzdbi_cbe,tau_expertile,WEitemp2_cbe);//计算权重	
				 subvec(y,index1,index2,subyi2_cbe);
			   gsl_vector_sub(subyi2_cbe,xbSzdbi_cbe);//	计算Pb_cbe  			 	

			   	
			   submatrix(x,index1,index2,0,p,Wixi_cbe);//计算抽取暂时新样本的后验分布			   

			   for(j=0;j<Ni;j++){gsl_matrix_get_row(Wixir_cbe,Wixi_cbe,j);			                    
				                 gsl_vector_scale(Wixir_cbe,gsl_vector_get(WEitemp2_cbe,j));
							     gsl_matrix_set_row(Wixi_cbe,j,Wixir_cbe);
			                     }
              submatrix(x,index1,index2,0,p,subxi);
			  gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0,subxi,Wixi_cbe,0.0,xWx_cbe);
			  gsl_matrix_add(temp_betaCV,xWx_cbe);
			  ////////////// 计算temp_bMU			  
			  subvec(y,index1,index2,subyi1_cbe);
			  gsl_vector_sub(subyi1_cbe,zdrbitemp_cbe);
		      gsl_blas_dgemv(CblasTrans,1.0,Wixi_cbe,subyi1_cbe,0.0,xWytemp_cbe);			 			 
			  gsl_vector_add(temp_betaMU,xWytemp_cbe);  
			  //printf("zdrbitemp_cbe3=%f\t",gsl_vector_get(zdrbitemp_cbe,0));
			 // printf("zdrbitemp_cbe4=%f\t",gsl_vector_get(zdrbitemp_cbe,1));      	  
			  //printf("\n");
//////计算后验分布 计算Pb_cbe 
		  
			  temp2_cbe=0.0;
			  for(j=0;j<Ni;j++){temp1_cbe=gsl_vector_get(subyi2_cbe,j);
			        	temp2_cbe=temp2_cbe+temp1_cbe*temp1_cbe*gsl_vector_get(WEitemp2_cbe,j);	
				                 }
			  ywy_cbe=ywy_cbe+temp2_cbe;			   	  
			                  }// end i

		 //修改了
		    gsl_matrix_scale(temp_betaCV,2.0*scale_c);
             for(h=0;h<p;h++){dtemp1_cbe=gsl_matrix_get(temp_betaCV,h,h)+1.0/(gsl_vector_get(ts_c,h));gsl_matrix_set(temp_betaCV,h,h,dtemp1_cbe);}
            gsl_vector_scale(temp_betaMU,2.0*scale_c);
		 //
             solve_m(temp_betaCV,p,p,solve1_cbe);

             gsl_blas_dgemv(CblasNoTrans,1.0,solve1_cbe,temp_betaMU,0.0,temp_betaMU1);
             rmvnorm_eigen(ccr,temp_betaMU1,solve1_cbe,betaNew_c);//新样本抽样

			 beDtbe_cbe=0.0;
			 for(h=0;h<p;h++){
			     beDtbe_cbe=beDtbe_cbe+gsl_vector_get(beta_c,h)*gsl_vector_get(beta_c,h)/gsl_vector_get(ts_c,h)/2.0;
			                 }

			 Pb_cbe=scale_c*ywy_cbe+beDtbe_cbe;

			 gsl_vector_memcpy(betaNewQn_c,betaNew_c);
			 gsl_vector_sub(betaNewQn_c,temp_betaMU1);

			 gsl_blas_dgemv(CblasNoTrans,1.0,temp_betaCV,betaNewQn_c,0.0,betaNewCV_cbe);			
			 gsl_blas_ddot(betaNewQn_c,betaNewCV_cbe,&Qnew_cbe);
			 Qnew_cbe=Qnew_cbe/2.0;	 

			 CVbdet_cbe=sqrt(get_det(temp_betaCV));// 对吗
		  
//计算Pnew_cbe、Qb_cbe
	// 注意主要不同的地方在 beta_c---betaNew_c 和	WEitemp2_cbe-WEitemp2New_cbe			

		    gsl_matrix_set_all(temp_betaCVNEW,0.0);
            gsl_vector_set_all(temp_betaMUNEW,0.0);
			 ywy_cbe=0;
		 for(i=0;i<N_train;i++){			 
			  index1=addsubject[i],index2=addsubject[i+1];  			  	 
	    	  submatrix(z,index1,index2,0,q,subzi);
		      gsl_matrix_memcpy(zid,subzi);	         
			  for(j=0;j<q;j++){gsl_matrix_get_col(zdtempc_cbe,zid,j);			                    
				               gsl_vector_scale(zdtempc_cbe,gsl_vector_get(d_c,j));
							   gsl_matrix_set_col(zid,j,zdtempc_cbe);
			                   }	

			   gsl_matrix_get_row(row_b,b_c,i); 
		       gsl_blas_dgemv(CblasNoTrans,1.0,GA_c,row_b,0.0,Gbitemp_cbe);
			   gsl_blas_dgemv(CblasNoTrans,1.0,zid,Gbitemp_cbe,0.0,zdrbitemp_cbe);			   			   			   			   
		       
			   submatrix(x,index1,index2,0,p,subxi);
			   gsl_blas_dgemv(CblasNoTrans,1.0,subxi,betaNew_c,0.0,xibeta);	//注意betaNew_c与MH分母不同

			   gsl_vector_memcpy(xbSzdbi_cbe,xibeta);
			   gsl_vector_add(xbSzdbi_cbe,zdrbitemp_cbe);
			   subvec(y,index1,index2,subyi);
			   Expertile_weight(subyi,xbSzdbi_cbe,tau_expertile,WEitemp2New_cbe);//计算权重	注意WEitemp2New_cbe与MH分母不同
			   subvec(y,index1,index2,subyi2_cbe);
			   gsl_vector_sub(subyi2_cbe,xbSzdbi_cbe);		//计算Pnew_cbe 	

			   submatrix(x,index1,index2,0,p,Wixi_cbe);	//计算Qb_cbe里的方差
			   for(j=0;j<Ni;j++){gsl_matrix_get_row(Wixir_cbe,Wixi_cbe,j);			                    
				                 gsl_vector_scale(Wixir_cbe,gsl_vector_get(WEitemp2New_cbe,j));//注意WEitemp2New_cbe与MH分母不同
							     gsl_matrix_set_row(Wixi_cbe,j,Wixir_cbe);
			                     }
			  submatrix(x,index1,index2,0,p,subxi); 
			  gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0,subxi,Wixi_cbe,0.0,xWx_cbe);
			  gsl_matrix_add(temp_betaCVNEW,xWx_cbe);//注意temp_betaCVNEW
			  ////////////// 计算temp_bMUNEW		
			  subvec(y,index1,index2,subyi1_cbe);			
			  gsl_vector_sub(subyi1_cbe,zdrbitemp_cbe);
		      gsl_blas_dgemv(CblasTrans,1.0,Wixi_cbe,subyi1_cbe,0.0,xWytemp_cbe);			 			 
			  gsl_vector_add(temp_betaMUNEW,xWytemp_cbe);  //注意temp_betaMUNEW			         	  

             //////计算后验分布 Pnew_cbe 		  
			  temp2_cbe=0.0;
			  for(j=0;j<Ni;j++){temp1_cbe=gsl_vector_get(subyi2_cbe,j);
			        	temp2_cbe=temp2_cbe+temp1_cbe*temp1_cbe*gsl_vector_get(WEitemp2New_cbe,j);	//注意WEitemp2New_cbe与MH分母不同
				                 }
			   ywy_cbe=ywy_cbe+temp2_cbe;			   	  
			                  }// end i

		 //修改了
		      gsl_matrix_scale(temp_betaCVNEW,2.0*scale_c);
		      for(h=0;h<p;h++){dtemp1_cbe=gsl_matrix_get(temp_betaCVNEW,h,h)+1.0/(gsl_vector_get(ts_c,h));gsl_matrix_set(temp_betaCVNEW,h,h,dtemp1_cbe);}//注意temp_betaCVNEW
			  gsl_vector_scale(temp_betaMUNEW,2.0*scale_c);
         //     
			  solve_m(temp_betaCVNEW,p,p,solve1_cbe);//注意temp_betaCVNEW

              gsl_blas_dgemv(CblasNoTrans,1.0,solve1_cbe,temp_betaMUNEW,0.0,temp_betaMU1); //注意temp_betaMUNEW           

			  beDtbe_cbe=0.0;
			  for(h=0;h<p;h++){
			      beDtbe_cbe=beDtbe_cbe+gsl_vector_get(betaNew_c,h)*gsl_vector_get(betaNew_c,h)/gsl_vector_get(ts_c,h)/2.0;//注意betaNew_c与MH分母不同
			                 }

			  Pnew_cbe=scale_c*ywy_cbe+beDtbe_cbe;

			   gsl_vector_memcpy(betaNewQn_c,beta_c);//注意beta_c与MH分子不同,betaNewQn_c，betaNewCV_cbe符号不再变化了
			   gsl_vector_sub(betaNewQn_c,temp_betaMU1);

			   gsl_blas_dgemv(CblasNoTrans,1.0,temp_betaCVNEW,betaNewQn_c,0.0,betaNewCV_cbe);	//	temp_betaCVNEW	
			   gsl_blas_ddot(betaNewQn_c,betaNewCV_cbe,&Qb_cbe);
			   Qb_cbe=Qb_cbe/2.0;	 

			   CVnewdet_cbe=sqrt(get_det(temp_betaCVNEW));;// 对吗
   ///////////////////////////////////////////////////////////////////////////////////////
			   alpha_cbe=(CVnewdet_cbe/CVbdet_cbe)*exp(-Pnew_cbe-Qb_cbe+Pb_cbe+Qnew_cbe);   
               unif_cbe=gsl_ran_flat(ccr,0.0,1.0);
               if(unif_cbe<=alpha_cbe){gsl_vector_memcpy(beta_c,betaNew_c);
			   gsl_matrix_set(MHbeta_c,iter,0,1.0);
			                           }	
									   
									   
               gsl_matrix_set_row(beta_p,iter,beta_c);  


	//----------------------------The full conditional for bi   
		 for(i=0;i<N_train;i++){
			 gsl_matrix_set_all(temp_bCV,0.0);
			 gsl_vector_set_all(temp_bMU,0.0);	
			 index1=addsubject[i],index2=addsubject[i+1];
			 submatrix(z,index1,index2,0,q,subzi);
			 gsl_matrix_memcpy(zid_cbi,subzi);
			 for(j=0;j<q;j++){gsl_matrix_get_col(zdtempc_cbi,zid_cbi,j);			                    
				               gsl_vector_scale(zdtempc_cbi,gsl_vector_get(d_c,j));
							   gsl_matrix_set_col(zid_cbi,j,zdtempc_cbi);
			                  }
			  gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,zid_cbi,GA_c,0.0,zidGA_cbi);
			  gsl_matrix_memcpy(WzidGA_cbi,zidGA_cbi);//保留zidGA_cbi 计算抽取暂时新样本的后验分布		  
				
			  gsl_matrix_memcpy(WnewzidGA_cbi,zidGA_cbi);//保留zidGA_cbi 计算Pnew_cbi和Qb_cbi

			  gsl_matrix_get_row(row_b,b_c,i); 
			  gsl_blas_dgemv(CblasNoTrans,1.0,zidGA_cbi,row_b,0.0,zidGAbi_cbi);	
		      
			  submatrix(x,index1,index2,0,p,subxi);
			   gsl_blas_dgemv(CblasNoTrans,1.0,subxi,beta_c,0.0,xibeta);

			   gsl_vector_memcpy(xbSzdbi_cbi,xibeta);//保留xibeta
			   gsl_vector_add(xbSzdbi_cbi,zidGAbi_cbi);	
			   subvec(y,index1,index2,subyi);
			   Expertile_weight(subyi,xbSzdbi_cbi,tau_expertile,WEitemp2_cbi);//计算权重			  
			   subvec(y,index1,index2,subyi2_cbi);
			  gsl_vector_sub(subyi2_cbi,xbSzdbi_cbi);//	计算Pb_cbi 

			   for(j=0;j<Ni;j++){gsl_matrix_get_row(WzidGAr_cbi,WzidGA_cbi,j);			                    
				                 gsl_vector_scale(WzidGAr_cbi,gsl_vector_get(WEitemp2_cbi,j));
							     gsl_matrix_set_row(WzidGA_cbi,j,WzidGAr_cbi);
			                     }

			  gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0,zidGA_cbi,WzidGA_cbi,0.0,temp_bCV);
		 //修改了
			  gsl_matrix_scale(temp_bCV,2.0*scale_c);
			  for(j=0;j<q;j++){temp1_cbi=gsl_matrix_get(temp_bCV,j,j)+1.0;gsl_matrix_set(temp_bCV,j,j,temp1_cbi);}
		 //
			  solve_m(temp_bCV,q,q,solve1_cbi);
			  ////////////// 计算temp_bMU		
			   submatrix(x,index1,index2,0,p,subxi);	
			  gsl_blas_dgemv(CblasNoTrans,1.0,subxi,beta_c,0.0,xibeta);
			  subvec(y,index1,index2,subyi1_cbi);
			  gsl_vector_sub(subyi1_cbi,xibeta);
		      gsl_blas_dgemv(CblasTrans,1.0,WzidGA_cbi,subyi1_cbi,0.0,temp_bMU);	
			         	  
			//修改了
			  gsl_vector_scale(temp_bMU,2.0*scale_c);
           //

             gsl_blas_dgemv(CblasNoTrans,1.0,solve1_cbi,temp_bMU,0.0,temp_bMU1);           
			  rmvnorm_eigen(ccr,temp_bMU1,solve1_cbi,rowbNew_c); 			 
//////计算后验分布 计算Pb_cbi 
		  
			  ywy_cbi=0.0;
			  for(j=0;j<Ni;j++){temp1_cbi=gsl_vector_get(subyi2_cbi,j);
			        	ywy_cbi=ywy_cbi+temp1_cbi*temp1_cbi*gsl_vector_get(WEitemp2_cbi,j);	
				                 }


			  biIbi_cbi=0.0;
			 for(h=0;h<q;h++){
			     biIbi_cbi=biIbi_cbi+gsl_vector_get(row_b,h)*gsl_vector_get(row_b,h)/2.0;
			                 }

			 Pb_cbi=scale_c*ywy_cbi+biIbi_cbi;

			 gsl_vector_memcpy(rowbNewQn_c,rowbNew_c);
			 gsl_vector_sub(rowbNewQn_c,temp_bMU1);

			 gsl_blas_dgemv(CblasNoTrans,1.0,temp_bCV,rowbNewQn_c,0.0,rowbNewCV_cbi);			
			 gsl_blas_ddot(rowbNewQn_c,rowbNewCV_cbi,&Qnew_cbi);
			 Qnew_cbi=Qnew_cbi/2.0;	 
			  
			 CVbdet_cbi=sqrt(get_det(temp_bCV));// 对吗

          //////////////////////////////////////////////////  计算Pnew_cbi和Qb_cbi
			 gsl_blas_dgemv(CblasNoTrans,1.0,zidGA_cbi,rowbNew_c,0.0,zidGAbi_cbi);		     
			 submatrix(x,index1,index2,0,p,subxi);
			 gsl_blas_dgemv(CblasNoTrans,1.0,subxi,beta_c,0.0,xibeta);

			 gsl_vector_memcpy(xbSzdbi_cbi,xibeta);//保留xibeta
			 gsl_vector_add(xbSzdbi_cbi,zidGAbi_cbi);	
			 subvec(y,index1,index2,subyi);	
			 Expertile_weight(subyi,xbSzdbi_cbi,tau_expertile,WEitemp2New_cbi);//计算权重WEitemp2New_cbi与分母不同
			 subvec(y,index1,index2,subyi2_cbi);
			 gsl_vector_sub(subyi2_cbi,xbSzdbi_cbi);//	计算Pnew_cbi 
	
			   for(j=0;j<Ni;j++){gsl_matrix_get_row(WnewzidGAr_cbi,WnewzidGA_cbi,j);			                    
				                 gsl_vector_scale(WnewzidGAr_cbi,gsl_vector_get(WEitemp2New_cbi,j));//计算权重WEitemp2New_cbi与分母不同
							     gsl_matrix_set_row(WnewzidGA_cbi,j,WnewzidGAr_cbi);
			                     }

			  gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0,zidGA_cbi,WnewzidGA_cbi,0.0,temp_bCVNew);//temp_bCVNew
			//修改了
			  gsl_matrix_scale(temp_bCVNew,2.0*scale_c);
			  for(j=0;j<q;j++){temp1_cbi=gsl_matrix_get(temp_bCVNew,j,j)+1.0;gsl_matrix_set(temp_bCVNew,j,j,temp1_cbi);}//temp_bCVNew
		 //
			  solve_m(temp_bCVNew,q,q,solve1_cbiNew);
			  ////////////// 计算temp_bMUNew		
			   submatrix(x,index1,index2,0,p,subxi);	
			  gsl_blas_dgemv(CblasNoTrans,1.0,subxi,beta_c,0.0,xibeta);
			  subvec(y,index1,index2,subyi1_cbi);
			  gsl_vector_sub(subyi1_cbi,xibeta);
		      gsl_blas_dgemv(CblasTrans,1.0,WnewzidGA_cbi,subyi1_cbi,0.0,temp_bMUNew);	//temp_bMUNew
		//修改了
			  gsl_vector_scale(temp_bMUNew,2.0*scale_c);
           //
              gsl_blas_dgemv(CblasNoTrans,1.0,solve1_cbiNew,temp_bMUNew,0.0,temp_bMU1New);   //temp_bMUNew,temp_bMU1New
//////计算后验分布 计算Pnew_cbi 
		  
			  ywy_cbi=0.0;
			  for(j=0;j<Ni;j++){temp1_cbi=gsl_vector_get(subyi2_cbi,j);
			        	ywy_cbi=ywy_cbi+temp1_cbi*temp1_cbi*gsl_vector_get(WEitemp2New_cbi,j);	//计算权重WEitemp2New_cbi与分母不同
				                 }


			  biIbi_cbi=0.0;
			 for(h=0;h<q;h++){
			     biIbi_cbi=biIbi_cbi+gsl_vector_get(rowbNew_c,h)*gsl_vector_get(rowbNew_c,h)/2.0;//rowbNew_c
			                 }
			 
			 Pnew_cbi=scale_c*ywy_cbi+biIbi_cbi;

			 gsl_vector_memcpy(rowbQn_c,row_b);
			 gsl_vector_sub(rowbQn_c,temp_bMU1New);//temp_bMU1New

			 gsl_blas_dgemv(CblasNoTrans,1.0,temp_bCVNew,rowbQn_c,0.0,rowbCV_cbi);	//temp_bCVNew		
			 gsl_blas_ddot(rowbQn_c,rowbCV_cbi,&Qb_cbi);
			 Qb_cbi=Qb_cbi/2.0;	 

			 CVnewdet_cbi=sqrt(get_det(temp_bCVNew));// 对吗

                 ///////////////////////////////////////////////////////////////////////////////////////
			   alpha_cbi=(CVnewdet_cbi/CVbdet_cbi)*exp(-Pnew_cbi-Qb_cbi+Pb_cbi+Qnew_cbi);
               unif_cbi=gsl_ran_flat(ccr,0.0,1.0);
               if(unif_cbi<=alpha_cbi){gsl_matrix_set_row(b_c,i,rowbNew_c);}				  
			 
			 for(h=0;h<q;h++){
				   index1_cbi=i*q+h;
				  temp2_cbi=gsl_matrix_get(b_c,i,h);
				   gsl_matrix_set(b_p,iter,index1_cbi,temp2_cbi);
			                   }
                      ////////////////////////////////////////////////////////////
			 
		                   } 
		  		 //-----The full conditional for scale###
            temp_sshape=g1+M_total/2;
            temp_srate=g2;
            for(i=0;i<N_train;i++){			 
			  index1=addsubject[i],index2=addsubject[i+1];  			  	 
	    	  submatrix(z,index1,index2,0,q,subzi);
		      gsl_matrix_memcpy(zid,subzi);	         
			  for(j=0;j<q;j++){gsl_matrix_get_col(zdtempc_cbe,zid,j);			                    
				               gsl_vector_scale(zdtempc_cbe,gsl_vector_get(d_c,j));
							   gsl_matrix_set_col(zid,j,zdtempc_cbe);
			                   }	

			   gsl_matrix_get_row(row_b,b_c,i); 
		       gsl_blas_dgemv(CblasNoTrans,1.0,GA_c,row_b,0.0,Gbitemp_cbe);
			   gsl_blas_dgemv(CblasNoTrans,1.0,zid,Gbitemp_cbe,0.0,zdrbitemp_cbe);	
		       
			   submatrix(x,index1,index2,0,p,subxi);
			   gsl_blas_dgemv(CblasNoTrans,1.0,subxi,beta_c,0.0,xibeta);	

			   gsl_vector_memcpy(xbSzdbi_cbe,xibeta);
			   gsl_vector_add(xbSzdbi_cbe,zdrbitemp_cbe);	
			   subvec(y,index1,index2,subyi);
			    Expertile_weight(subyi,xbSzdbi_cbe,tau_expertile,WEitemp2_cbe);//计算权重	
				subvec(y,index1,index2,subyi2_cbe);
			   gsl_vector_sub(subyi2_cbe,xbSzdbi_cbe);//	
		  
			  temp2_cbe=0.0;
			  for(j=0;j<Ni;j++){temp1_cbe=gsl_vector_get(subyi2_cbe,j);
			        	temp2_cbe=temp2_cbe+temp1_cbe*temp1_cbe*gsl_vector_get(WEitemp2_cbe,j);	
				                 }
			   temp_srate= temp_srate+temp2_cbe;			   	  
			                  }// end i

		   temp_sscale=1.0/(temp_srate);
		   scale_c=gsl_ran_gamma(ccr,temp_sshape,temp_sscale);
		   gsl_vector_set(scale_p,iter,scale_c);
  
		   
		
		 


					 
 



		


	 //------The full conditional for lambda	
			 //dummy=1:adlassoexp;dummy=2:lassoexp

	   if(strcmp(method,adlassoexpEx)==0){ 
			          temp_shape=c1+1;
					  //lambda1
                      for(i=0;i<p;i++){ 
		                    temp_scale=1.0/(c2+gsl_vector_get(ts_c,i)/2.0);				                 
		                    gsl_vector_set(lambda1_c,i,gsl_ran_gamma(ccr,temp_shape,temp_scale));          
                                       }
			          gsl_matrix_set_row(lambda1_p,iter,lambda1_c); 

                      //lambda2
					  for(i=0;i<q;i++){ 
			               
			                 temp_scale=1.0/(c2+gsl_vector_get(d_c,i));			              
		                     gsl_vector_set(lambda2_c,i,gsl_ran_gamma(ccr,temp_shape,temp_scale));                            
                                        }
					   //lambda3
					   temp_shape=c1+dim_r;		 
	                 temp_scale=1.0/(c2+sum_v(hs_c)/2.0);		
	  	             gsl_vector_set_all(lambda3_c,gsl_ran_gamma(ccr,temp_shape,temp_scale));   

					/*  for(i=0;i<dim_r;i++){
			                 temp_scale=1.0/(c2+gsl_vector_get(hs_c,i)/2.0);
		                     gsl_vector_set(lambda3_c,i,gsl_ran_gamma(ccr,temp_shape,temp_scale));                           
                                           }	
		              gsl_matrix_set_row(lambda3_p,iter,lambda3_c);	*/
                         }

		if(strcmp(method,lassoexpEx)==0){//lambda1
			         temp_shape=c1+p;		 
	                 temp_scale=1.0/(c2+sum_v(ts_c)/2.0);		
	  	             gsl_vector_set_all(lambda1_c,gsl_ran_gamma(ccr,temp_shape,temp_scale));  
					 //lambda2
					 temp_shape=c1+q;	
					 //temp_scale=1.0/(c2+sum_v(etas_c)/2.0);	//截断正态分布先验	
	                 temp_scale=1.0/(c2+sum_v(d_c));		
	            	 gsl_vector_set_all(lambda2_c,gsl_ran_gamma(ccr,temp_shape,temp_scale));   
					 //lambda3
					 temp_shape=c1+dim_r;		 
	                 temp_scale=1.0/(c2+sum_v(hs_c)/2.0);		
	  	             gsl_vector_set_all(lambda3_c,gsl_ran_gamma(ccr,temp_shape,temp_scale));   
		             }

	  //	gsl_matrix_set_row(lambda1_p,iter,lambda1_c);   
		//gsl_matrix_set_row(lambda2_p,iter,lambda2_c);	
		// gsl_matrix_set_row(lambda3_p,iter,lambda3_c);

		//------The full conditional for ts,hs
		 for(i=0;i<p;i++){
			 temp_lambda=gsl_vector_get(lambda1_c,i);
             temp_nu=sqrt(gsl_vector_get(lambda1_c,i)/gsl_pow_2(gsl_vector_get(beta_c,i)));			
		     ingaus_temp=rinvGauss1(ccr,temp_nu,temp_lambda);
			 if(ingaus_temp<1e-10){//printf("Warning:random number from inverse Gaussion(ts) < 1e-10, replaced it by 1e-10\n ");
				                   ingaus_temp= 1e-10;}
		     gsl_vector_set(ts_c,i,1.0/ingaus_temp);             
                         }

         for(i=0;i<dim_r;i++){ 
			 temp_lambda=gsl_vector_get(lambda3_c,i);				 
			 temp_nu=sqrt(gsl_vector_get(lambda3_c,i)/gsl_pow_2(gsl_vector_get(r_c,i)));			
             ingaus_temp=rinvGauss1(ccr,temp_nu,temp_lambda);
			 if(ingaus_temp<1e-10){//printf("Warning:random number from inverse Gaussion(hs) < 1e-10, replaced it by 1e-10\n ");
			                       ingaus_temp= 1e-10;}
		   gsl_vector_set(hs_c,i,1.0/ingaus_temp);                                                 
                              } 
		

			
	                                  }//end iteration
    //save data    


//	save_datam(v_p,n_sampler,M_total,method,paraname1,tau,examplebumber,simk,distribution);	
	save_datam(beta_p,n_sampler,p,method,paraname2,tau,examplebumber,simk,distribution);	
//	save_datam(lambda1_p,n_sampler,p,method,paraname3,tau,examplebumber,simk,distribution);	
	save_datam(d_p,n_sampler,q,method,paraname4,tau,examplebumber,simk,distribution);	
//	save_datam(lambda2_p,n_sampler,q,method,paraname5,tau,examplebumber,simk,distribution);	//d_c假设截断正态才有
	save_datam(r_p,n_sampler,dim_r,method,paraname6,tau,examplebumber,simk,distribution);
//	save_datam(lambda3_p,n_sampler,dim_r,method,paraname7,tau,examplebumber,simk,distribution);	
	save_datam(b_p,n_sampler,q*N_train,method,paraname8,tau,examplebumber,simk,distribution);
	save_datam(MHbeta_c,n_sampler,1,method,paraname9,tau,examplebumber,simk,distribution);
	save_datam(MHr_c,n_sampler,1,method,paraname10,tau,examplebumber,simk,distribution);
	save_datam(MHd_c,n_sampler,q,method,paraname11,tau,examplebumber,simk,distribution);	

    //remove variables
	delete []addsubject;addsubject=nullptr;	
	gsl_permutation_free(permu1);
	delete []cumsump;cumsump=nullptr;	
	gsl_vector_free(beta_c);
	gsl_vector_free(lambda1_c);
	gsl_vector_free(ts_c);
	gsl_vector_free(d_c);
	gsl_vector_free(lambda2_c);
//	gsl_vector_free(etas_c);
	gsl_vector_free(r_c);
	gsl_vector_free(lambda3_c);
	gsl_vector_free(hs_c);
	gsl_matrix_free(GA_c);
	gsl_matrix_free(b_c);
	gsl_vector_free(v_c);

	gsl_matrix_free(v_p);
	gsl_vector_free(scale_p);
	gsl_matrix_free(beta_p);
	gsl_matrix_free(lambda1_p);
	gsl_matrix_free(d_p);
	gsl_matrix_free(lambda2_p);
	gsl_matrix_free(r_p);
	gsl_matrix_free(lambda3_p);
	gsl_matrix_free(b_p);

	//start gibbs sampler
	gsl_matrix_free(z);
	gsl_vector_free(row_x);
	gsl_vector_free(row_z);
	gsl_vector_free(row_b);

	//v
	gsl_vector_free(vtemp1_cv);
	gsl_vector_free(vtemp2_cv);

	//bi	 
    gsl_matrix_free(temp_bCV);
	gsl_vector_free(temp_bMU);
	gsl_matrix_free(zid_cbi);
	gsl_vector_free(zdtempc_cbi);
	gsl_matrix_free(zidGA_cbi);
	gsl_vector_free(bi_c);
	gsl_matrix_free(solve1_cbi);
	gsl_vector_free(temp_bMU1);

	gsl_matrix_free(mtemp1_cb);
	gsl_permutation_free(b_per);
	gsl_matrix_free(solve1_cb);
	gsl_matrix_free(mtemp2_cb);
	//////计算Pb_cbi和Qnew_cbi 
	gsl_matrix_free(WzidGA_cbi);			  
	gsl_vector_free(zidGAbi_cbi);
	gsl_vector_free(subyi2_cbi);
	gsl_vector_free(xbSzdbi_cbi);
	gsl_vector_free(WEitemp2_cbi);
	gsl_vector_free(WzidGAr_cbi);
	gsl_vector_free(subyi1_cbi);
	gsl_vector_free(rowbNew_c);
	gsl_vector_free(rowbNewQn_c);
	gsl_vector_free(rowbNewCV_cbi);

	///////////////////计算Pnew_cbi和Qb_cbi
	gsl_matrix_free(WnewzidGA_cbi);
	gsl_vector_free(WEitemp2New_cbi);
	gsl_vector_free(WnewzidGAr_cbi);
	gsl_matrix_free(temp_bCVNew);
	gsl_matrix_free(solve1_cbiNew);
	gsl_vector_free(temp_bMUNew);
	gsl_vector_free(temp_bMU1New);
	gsl_vector_free(rowbQn_c);
	gsl_vector_free(rowbCV_cbi);
  /////	end bi

	//d
	gsl_vector_free(rbi_cd);
	gsl_matrix_free(Gi_cd);
	gsl_vector_free(Gic_cd);
	gsl_vector_free(zidrbi_cd);
	gsl_matrix_free(zid_cd);
	gsl_vector_free(zdtempc_cd);
	gsl_vector_free(Gbitemp_cd);
	gsl_vector_free(Gic1_cd);	
//Pnew_cd,Qb_cd,Qnew_cd,Pb,cd
	gsl_vector_free(zidrbiSxb_cd);
	gsl_vector_free(subyi2_cd);
	gsl_vector_free(WEitemp2_cd);
	gsl_vector_free(WGir_cd);
	gsl_vector_free(subyi3_cd);
	gsl_vector_free(WEitemp2New_cd);
	gsl_vector_free(dNew_cd);
		
	//r
	gsl_matrix_free(temp_rCV);
	gsl_vector_free(temp_rMU);
	gsl_vector_free(zdtempc_cr);
	gsl_matrix_free(zid);
	gsl_matrix_free(zdtemp);
	gsl_matrix_free(Fir);
	gsl_vector_free(zdbitemp_cr);
	gsl_vector_free(xibeta);	
	gsl_matrix_free(solve1_cr);
	gsl_vector_free(temp_rMU1);
	gsl_matrix_free(mtemp2_cr);


		//计算Pb_cr、Qnew_cr
	gsl_vector_free(FirMr_cr);
	gsl_vector_free(subyi2_cr);
	gsl_vector_free(xbSzdbiSFr_cr);
	gsl_vector_free(xbSzdbi_cr);
	gsl_vector_free(WEitemp2_cr);
	gsl_matrix_free(WiFir_cr);
	gsl_vector_free(WiFirRow_cr);
	gsl_matrix_free(FWFtemp_cr);
	gsl_vector_free(subyi2_cr);
	gsl_vector_free(FWytemp_cr);
	gsl_vector_free(rNew_c);
	gsl_vector_free(rNewQn_c);
	gsl_vector_free(rNewCV_cbe);
	//计算Pnew_cr、Qb_cr   gsl_matrix_free(temp_rCVNew);
    gsl_vector_free(temp_rMUNew);	
	gsl_vector_free(WEitemp2New_cr);


	//scale
	 gsl_vector_free(vtemp1_csc);
	 gsl_vector_free(vtemp2_csc);
	//beta
	gsl_matrix_free(temp_betaCV);
    gsl_vector_free(temp_betaMU);
	gsl_matrix_free(temp_betaCVNEW);
    gsl_vector_free(temp_betaMUNEW);
	gsl_matrix_free(subxi);
	gsl_vector_free(subyi);
	gsl_matrix_free(vixi);
	gsl_matrix_free(subzi);

	
	gsl_matrix_free(Wixi_cbe);
	gsl_vector_free(Wixir_cbe);
	gsl_matrix_free(xWx_cbe);
	gsl_vector_free(subyi1_cbe);
	gsl_vector_free(xWytemp_cbe);

	gsl_vector_free(subyi2_cbe);	
	gsl_vector_free(xbSzdbi_cbe);
	gsl_vector_free(WEitemp2_cbe);
	gsl_vector_free(WEitemp2New_cbe);
	gsl_vector_free(betaNew_c);	
	gsl_vector_free(betaNewQn_c);
	gsl_vector_free(betaNewCV_cbe);

	gsl_vector_free(zdtempc_cbe);
	gsl_vector_free(Gbitemp_cbe);
	gsl_vector_free(zdrbitemp_cbe);
	gsl_matrix_free(solve1_cbe);
	gsl_vector_free(temp_betaMU1);

	gsl_matrix_free(MHbeta_c);
	gsl_matrix_free(MHr_c);
	gsl_matrix_free(MHd_c);
	//scale
	
return 0;
}


