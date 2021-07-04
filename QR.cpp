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
int mixed_adlassoexp(const int examplebumber,char method[4],const gsl_rng *ccr,const gsl_vector *y, const int N_train, const gsl_matrix *x, const int p, const int q, const int *tabsubject,const double tau, const int n_sampler,const int simk,char distribution[7]);
int simk, i, j;//simk--第simk个数据examplebumber 注意修改
const int N_train=50,N_new=51,Ni=5,p1=8,q1=8,K=100,sd=1,n1_sampler=20000,p=9,q=9;
double tau=0.0;//the tau quantile
	///////////////////误差分布
//	char distribution[]={"norm"};
	char distribution[]={"t3"};
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
for(examplebumber=1;examplebumber<=1;examplebumber++){
for(itau=0;itau<1;itau++){/////////////tau开始
	tau=gsl_vector_get(tauall,itau);	
	printf("tau=%lf\n",tau);
for(simk=1;simk<=5;simk++){
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
             fscanf(fp2,"%lf",&y_tempnew[i]); //读数据，读入读出要一致double必须是%lf	       
                              }
	   
	   for(i=0;i<M_train;i++){ 
              gsl_vector_set(y_temp,i,y_tempnew[i]); //读数据，读入读出要一致double必须是%lf		      
                             }	 

	  printf("\n");

		printf("\n");
		 fclose(fp2); 
	

         lengt=length_vec(y_temp);
	
		 
	
   srand((int)time(0));
  //  utim=random(50000);
   gsl_rng_set(ccr1,1235+simk);	//1236+simk

/////////////////////////方法	
 	char lassoexp[]={"lassoexp"};
	char adlassoexp[]={"adlassoexp"};
	//////////////////////
 // mixed_adlassoexp(examplebumber,lassoexp,ccr1,y_temp,N_train,x_temp,p,q,tabsubject_train,tau,n1_sampler,simk,distribution);
mixed_adlassoexp(examplebumber,adlassoexp,ccr1,y_temp,N_train,x_temp,p,q,tabsubject_train,tau,n1_sampler,simk,distribution);
	


                }
				}//tau结束
				}//examplebumber结束
			
   finish = clock();  
   duration = (double)(finish - start) / CLOCKS_PER_SEC;   
   printf( "\n\nTotal time is %f seconds\n", duration );  
delete []x1_tempnew;x1_tempnew=nullptr;
delete []y_tempnew;y_tempnew=nullptr;
gsl_vector_free(tauall);
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

////////////////////////////////////////
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
/* multivariate normal distribution random number generator using cholesky*/
/*
*	n	dimension of the random vector
*	mean	vector of means of size n
*	var	variance matrix of dimension n x n
*	result	output variable with a single random vector normal distribution generation
*/
double min_v(const gsl_vector *vect);
if(cvar->size1!=cvar->size2){printf("协方差矩阵不是方阵");exit(0);}
if(cvar->size1!=mean->size){printf("均值和协方差维数不相同");exit(0);}
int k,n=mean->size,kk;
gsl_vector *eval=gsl_vector_alloc (n);
gsl_matrix *evec=gsl_matrix_alloc (n, n);
gsl_eigen_symmv_workspace *w=gsl_eigen_symmv_alloc(n);
gsl_vector *stdnorm=gsl_vector_alloc (n);
gsl_matrix *AA=gsl_matrix_alloc (n, n);
gsl_matrix_memcpy(AA,cvar);
gsl_eigen_symmv (AA,eval,evec, w);
double temp=0.0;
//if(min_v(eval)<0.0){rmvnorm_svd(r,mean,cvar,result);printf("the svd function run");return 0;}
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
	  for(j=0;j<i;j++){//printf("%d\t",j+addindex[i-1]);
		  gsl_matrix_set(lowermx,i,j,gsl_vector_get(vec,j+addindex[i-1]));
                          }
                      }

gsl_permutation_free(tabp);
delete []addindex;addindex=nullptr;
return 0;
}
int vect_maxtrix1c(const gsl_vector *vec, const int length, const int dim, gsl_matrix *lowermx){
	int dim1=int((1+sqrt(1+length*8.0))/2);//将向量vec变为\vargamma(下三角矩阵)按照列(c)排,对角线为1
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

int mixed_adlassoexp(const int examplebumber,char method[4],const gsl_rng *ccr,const gsl_vector *y, const int N_train, const gsl_matrix *x, const int p, const int q, const int *tabsubject,const double tau, const int n_sampler,const int simk,char distribution[7]){
	//int vect_maxtrix1c(const gsl_vector *vec, const int length, const int dim, gsl_matrix *lowermx);//向量变下三角
	int vect_maxtrix1r(const gsl_vector *vec, const int length, const int dim, gsl_matrix *lowermx);//向量变下三角
	int two_vecmul(const gsl_vector *twovec1,const int len1,const gsl_vector *twovec2,const int len2,gsl_matrix *maxtwo);//x_{p*1}%*%t(x_{p*1})
	int save_datam(const gsl_matrix *needsavedm,const int rowm,const int colm,char method[4], char Filename1[4],const float quantile,const int examplebumber, const int iter,char distribution[7]);//save data
	double sum_v(const gsl_vector *vect);//向量求均值
	double rinvGauss1(const gsl_rng *r,double mu, double lambda);//逆高斯分布
	int rmvnorm(const gsl_rng *r, const gsl_vector *mean, const gsl_matrix *var, gsl_vector *result);//高斯分布
	int rmvnorm_svd(const gsl_rng *r, const gsl_vector *mean, const gsl_matrix *var, gsl_vector *result);//高斯分布
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
	 char lassoexp[]={"lassoexp"};
	char adlassoexp[]={"adlassoexp"};
	const double xi1=(1.0-2.0*tau)/(tau*(1.0-tau)),xi2=sqrt(2.0/(tau*(1.0-tau)));
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
	int index_temp;
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
	gsl_vector *subvi=gsl_vector_calloc(Ni);	
	
	// ---Prepare for lambda
	double temp_shape=0.0,temp_scale=0.0;
	// ---Prepare for bi	     
    gsl_matrix *temp_bCV=gsl_matrix_calloc(q,q);
    gsl_vector *temp_bMU=gsl_vector_calloc(q);
	gsl_matrix *zid_cbi=gsl_matrix_calloc(Ni,q);
	gsl_vector *zdtempc_cbi=gsl_vector_calloc(Ni);
	gsl_matrix *zidGA_cbi=gsl_matrix_calloc(Ni,q);
	gsl_matrix *vizidGA_cbi=gsl_matrix_calloc(Ni,q);
	gsl_vector *vizidGAr_cbi=gsl_vector_calloc(q);	
	int index1_cbi;
	double temp1_cbi,temp2_cbi;	
	gsl_vector *bi_c=gsl_vector_calloc(q);
	gsl_matrix *solve1_cbi=gsl_matrix_calloc(q,q);
    gsl_vector *temp_bMU1=gsl_vector_calloc(q);

    gsl_matrix *mtemp1_cb=gsl_matrix_calloc(q,q);
	double dtemp1_cb=0.0;
	int index_p;
	double tempbi_c;
	gsl_permutation *b_per=gsl_permutation_alloc (q);
	gsl_matrix *solve1_cb=gsl_matrix_calloc(q,q);
	gsl_matrix *mtemp2_cb=gsl_matrix_calloc(q,q);
//	gsl_matrix *temp_bCV1=gsl_matrix_calloc(q,q);
//	gsl_vector *temp_bMU1=gsl_vector_calloc(q);
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
   
	 //---Prepare for r
    gsl_vector *temp_u=gsl_vector_calloc(dim_r);
    gsl_matrix *temp_rCV=gsl_matrix_calloc(dim_r,dim_r);
	gsl_vector *temp_rMU=gsl_vector_calloc(dim_r);
	gsl_matrix *zid=gsl_matrix_calloc(Ni,q);
	gsl_vector *zdtempc_cr=gsl_vector_calloc(Ni);
	gsl_matrix *zdtemp=gsl_matrix_calloc(Ni,q);
    gsl_matrix *Fir=gsl_matrix_calloc(Ni,dim_r);
	gsl_matrix *viFi=gsl_matrix_calloc(Ni,dim_r);
	gsl_vector *viFitempr_cr=gsl_vector_calloc(dim_r);
	gsl_matrix *FvFtemp_cr=gsl_matrix_calloc(dim_r,dim_r);
	gsl_vector *zdbitemp_cr=gsl_vector_calloc(Ni);
	gsl_vector *xibeta=gsl_vector_calloc(Ni);
	gsl_vector *Fvytemp_cr=gsl_vector_calloc(dim_r);
	

	gsl_matrix *mtemp1_crt=gsl_matrix_calloc(dim_r,dim_r);
	gsl_vector *vtemp2_cr=gsl_vector_calloc(q);	
	gsl_vector *vtemp1_cr=gsl_vector_calloc(dim_r);	
    double dtemp1_cr,dtemp2_cr;
	gsl_matrix *solve1_cr=gsl_matrix_calloc(dim_r,dim_r);
	gsl_vector *temp_rMU1=gsl_vector_calloc(dim_r); 
	gsl_matrix *mtemp2_cr=gsl_matrix_calloc(dim_r,dim_r);
	//--Prepare for scale
	double temp_sshape,temp_srate,temp_sscale,dtemp1_csc,dtemp2_csc;
    gsl_vector *vtemp2_csc=gsl_vector_calloc(q);
    gsl_vector *vtemp1_csc=gsl_vector_calloc(q);
	 //---Prepare for beta
	gsl_matrix *temp_betaCV=gsl_matrix_calloc(p,p);
    gsl_vector *temp_betaMU=gsl_vector_calloc(p);
	gsl_matrix *subxi=gsl_matrix_calloc(Ni,p);
	gsl_vector *subyi=gsl_vector_calloc(Ni);
	gsl_matrix *vixi=gsl_matrix_calloc(Ni,q);
	gsl_matrix *subzi=gsl_matrix_calloc(Ni,q);
	
	gsl_vector *vxtempr_cbe=gsl_vector_calloc(p);
	gsl_vector *zdtempc_cbe=gsl_vector_calloc(Ni);
	gsl_vector *Gbitemp_cbe=gsl_vector_calloc(q);
	gsl_matrix *xvxtemp_cbe=gsl_matrix_calloc(p,p);
	gsl_vector *zdrbitemp_cbe=gsl_vector_calloc(Ni);
	gsl_vector *xvytemp_cbe=gsl_vector_calloc(p);
	gsl_matrix *solve1_cbe=gsl_matrix_calloc(p,p);
	gsl_vector *temp_betaMU1=gsl_vector_calloc(p); 

  /*  gsl_matrix *mtemp1_cbe=gsl_matrix_calloc(p,p);
    gsl_vector *vtemp2_cbe=gsl_vector_calloc(q);
    gsl_vector *vtemp1_cbe=gsl_vector_calloc(q);
	gsl_vector *vtemp1x_cbe=gsl_vector_calloc(p);*/
    double dtemp1_cbe;
	gsl_matrix *wi_cbe=gsl_matrix_calloc(Ni,q);
	gsl_matrix *solveui_cbe=gsl_matrix_calloc(Ni,Ni);
	gsl_matrix *ui_cbe=gsl_matrix_calloc(Ni,Ni);
	gsl_matrix *xiui_cbe=gsl_matrix_calloc(p,Ni);
	gsl_matrix *xux_cbe=gsl_matrix_calloc(p,p);
	gsl_vector *xuytemp_cbe=gsl_vector_calloc(p);
	

	for(iter=0;iter<n_sampler;iter++){
		if((iter+1)%5000==0){printf("This is step %d\n",iter+1);}
		
		  //---------------------------The full conditional for beta积掉bi PCG
		 gsl_matrix_set_all(temp_betaCV,0.0);
         gsl_vector_set_all(temp_betaMU,0.0);
		 for(i=0;i<N_train;i++){			 
			  index1=addsubject[i],index2=addsubject[i+1];
			  submatrix(x,index1,index2,0,p,subxi);	 
	    	  submatrix(z,index1,index2,0,q,subzi);
		      gsl_matrix_memcpy(zid,subzi);	         
			  for(j=0;j<q;j++){gsl_matrix_get_col(zdtempc_cbe,zid,j);			                    
				               gsl_vector_scale(zdtempc_cbe,gsl_vector_get(d_c,j));
							   gsl_matrix_set_col(zid,j,zdtempc_cbe);
			                   }	

			  gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,zid,GA_c,0.0,wi_cbe);
			  gsl_blas_dgemm(CblasNoTrans,CblasTrans,1.0,wi_cbe,wi_cbe,0.0,solveui_cbe);

			  subvec(v_c,index1,index2,subvi);
			  gsl_vector_scale(subvi,xi2*xi2/scale_c);
			  for(h=0;h<Ni;h++){dtemp1_cbe=gsl_matrix_get(solveui_cbe,h,h)+gsl_vector_get(subvi,h);gsl_matrix_set(solveui_cbe,h,h,dtemp1_cbe);}
			  solve_m(solveui_cbe,Ni,Ni,ui_cbe);

			  submatrix(x,index1,index2,0,p,subxi);
			  gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0,subxi,ui_cbe,0.0,xiui_cbe);
			  gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,xiui_cbe,subxi,0.0,xux_cbe);
			  gsl_matrix_add(temp_betaCV,xux_cbe);
			 //下面是temp_betaMU	
			  subvec(y,index1,index2,subyi);
			  subvec(v_c,index1,index2,subvi);
			  gsl_vector_scale(subvi,xi1);
			  gsl_vector_sub(subyi,subvi);
			  gsl_blas_dgemv(CblasNoTrans,1.0,xiui_cbe,subyi,0.0,xuytemp_cbe);
			  gsl_vector_add(temp_betaMU,xuytemp_cbe);  
		                        }

            for(h=0;h<p;h++){dtemp1_cbe=gsl_matrix_get(temp_betaCV,h,h)+1.0/gsl_vector_get(ts_c,h);gsl_matrix_set(temp_betaCV,h,h,dtemp1_cbe);}
            solve_m(temp_betaCV,p,p,solve1_cbe);
            gsl_blas_dgemv(CblasNoTrans,1.0,solve1_cbe,temp_betaMU,0.0,temp_betaMU1);
            rmvnorm_eigen(ccr,temp_betaMU1,solve1_cbe,beta_c);
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
			  gsl_matrix_memcpy(vizidGA_cbi,zidGA_cbi);

			  subvec(v_c,index1,index2,subvi); 
			  for(j=0;j<Ni;j++){gsl_matrix_get_row(vizidGAr_cbi,vizidGA_cbi,j);			                    
				               gsl_vector_scale(vizidGAr_cbi,1.0/gsl_vector_get(subvi,j));
							   gsl_matrix_set_row(vizidGA_cbi,j,vizidGAr_cbi);
			                   }

			  gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0,zidGA_cbi,vizidGA_cbi,0.0,temp_bCV);
			  gsl_matrix_scale(temp_bCV,scale_c/(xi2*xi2));

			  for(j=0;j<q;j++){temp1_cbi=gsl_matrix_get(temp_bCV,j,j)+1.0;gsl_matrix_set(temp_bCV,j,j,temp1_cbi);}
			  solve_m(temp_bCV,q,q,solve1_cbi);

			  //计算temp_bMU
			 submatrix(x,index1,index2,0,p,subxi);	
			 gsl_blas_dgemv(CblasNoTrans,1.0,subxi,beta_c,0.0,xibeta);
		     subvec(y,index1,index2,subyi);

			 gsl_vector_sub(subyi,xibeta);
			 subvec(v_c,index1,index2,subvi); 
			 gsl_vector_scale(subvi,xi1);
			 gsl_vector_sub(subyi,subvi);

			 gsl_blas_dgemv(CblasTrans,1.0,vizidGA_cbi,subyi,0.0,temp_bMU);
			 gsl_vector_scale(temp_bMU,scale_c/(xi2*xi2));

             gsl_blas_dgemv(CblasNoTrans,1.0,solve1_cbi,temp_bMU,0.0,temp_bMU1);
            //rmvnorm(ccr,temp_bMU1,solve1_cbi,bi_c); 
			  rmvnorm_eigen(ccr,temp_bMU1,solve1_cbi,bi_c); 
			 gsl_matrix_set_row(b_c,i,bi_c);
			 for(h=0;h<q;h++){
				   index1_cbi=i*q+h;
				   temp2_cbi=gsl_vector_get(bi_c,h);
				   gsl_matrix_set(b_p,iter,index1_cbi,temp2_cbi);
			                   }
		                   } 	
		  		  //-----The full conditional for scale###
	//	  printf("%d",2);printf("\n\n");
        temp_sshape=g1+3*M_total/2;
           temp_srate=0.0;
           for(i=0;i<N_train;i++){
			   gsl_matrix_get_row(row_b,b_c,i);
			   for(j=0;j<tabsubject[i];j++){
				   index_temp=addsubject[i]+j;	
                   gsl_matrix_get_row(row_x,x,index_temp);  
                   gsl_blas_ddot(row_x,beta_c,&dtemp2_csc);
            
                   gsl_matrix_get_row(row_z,z,index_temp);
                   gsl_vector_memcpy(vtemp2_csc,row_z);	            
                   gsl_vector_mul(vtemp2_csc,d_c);

		          gsl_blas_dgemv(CblasNoTrans,1.0,GA_c,row_b,0.0,vtemp1_csc);							
	              gsl_blas_ddot(vtemp2_csc,vtemp1_csc,&dtemp1_csc);
                  dtemp1_csc=gsl_vector_get(y,index_temp)-dtemp2_csc-dtemp1_csc-xi1*gsl_vector_get(v_c,index_temp);
                  dtemp1_csc=gsl_pow_2(dtemp1_csc);
                  dtemp1_csc=dtemp1_csc/(2*xi2*xi2*gsl_vector_get(v_c,index_temp))+gsl_vector_get(v_c,index_temp);            
                  temp_srate+=dtemp1_csc;
                                             }
                                  }
		   temp_srate+=g2;
		 //  printf("\n%lf,%lf\n",temp_sshape,temp_srate);
		   temp_sscale=1.0/(temp_srate);
		   scale_c=gsl_ran_gamma(ccr,temp_sshape,temp_sscale);
		   gsl_vector_set(scale_p,iter,scale_c);
		   
		//-----The full conditional for v
	//	printf("%d",1);printf("\n\n");
		temp_lambda=xi1*xi1*scale_c/(xi2*xi2)+2*scale_c;		
		for(i=0;i<N_train;i++){	
			gsl_matrix_get_row (row_b,b_c,i);
	        for(j=0;j<tabsubject[i];j++){
				index_temp=addsubject[i]+j;			        
			    gsl_matrix_get_row (row_x,x,index_temp);
		        gsl_matrix_get_row (row_z,z,index_temp);
				gsl_vector_memcpy(vtemp2_cv,row_z);	
		        gsl_blas_ddot(row_x,beta_c,&dtemp1_cv);
				
				gsl_blas_dgemv(CblasNoTrans,1.0,GA_c,row_b,0.0,vtemp1_cv);
				gsl_vector_mul(vtemp2_cv,d_c);				
				gsl_blas_ddot(vtemp2_cv,vtemp1_cv,&dtemp2_cv);
				temp_nu=gsl_vector_get(y,index_temp)-dtemp1_cv-dtemp2_cv;				
				temp_nu=sqrt(xi1*xi1+2.0*xi2*xi2)/abs(temp_nu);	
				//printf("%11.8lf\t",temp_nu);
				ingaus_temp=rinvGauss1(ccr,temp_nu,temp_lambda);
				if(ingaus_temp < 1e-10){//printf("Warning:random number from inverse Gaussion(v) < 1e-10, replaced it by 1e-10\n ");
				                        ingaus_temp= 1e-10;}
    			 gsl_vector_set(v_c,index_temp,1.0/ingaus_temp);		
	                                   }      
			
                                }
			
			 gsl_matrix_set_row(v_p,iter,v_c);		
								
		 

		   //------The full conditional for d
	 for(k=0;k<q;k++) {
		 d_sigma=0.0,d_mu=0.0;
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
			  temp1_cd=0.0;
			  subvec(v_c,index1,index2,subvi); 
			  for(j=0;j<Ni;j++){ temp2_cd=gsl_vector_get(Gic_cd,j);
			                     temp3_cd=1.0/gsl_vector_get(subvi,j);
								 temp1_cd=temp1_cd+temp2_cd*temp2_cd*temp3_cd;
			                    }
			  d_sigma=d_sigma+temp1_cd;
			  //下面是均值
			  submatrix(x,index1,index2,0,p,subxi);	
			  gsl_blas_dgemv(CblasNoTrans,1.0,subxi,beta_c,0.0,xibeta);
	          gsl_blas_dgemv(CblasNoTrans,1.0,Gi_cd,d_c,0.0,zidrbi_cd);	        
			 // for(j=0;j<Ni;j++){printf("%lf\t",gsl_vector_get(zidrbi_cd,j));}
			//  printf("\n\n");		
			
			  subvec(y,index1,index2,subyi);
			  gsl_vector_sub(subyi,xibeta);
	          gsl_vector_sub(subyi,zidrbi_cd);
			  subvec(v_c,index1,index2,subvi); 
			  gsl_vector_scale(subvi,xi1);
			  gsl_vector_sub(subyi,subvi);
			  gsl_matrix_get_col(Gic1_cd,Gi_cd,k);
			  gsl_vector_scale(Gic1_cd,gsl_vector_get(d_c,k));
			  gsl_vector_add(subyi,Gic1_cd);

			  gsl_matrix_get_col(Gic_cd,Gi_cd,k);
			  temp1_cd=0.0;
			  subvec(v_c,index1,index2,subvi); 
			  for(j=0;j<Ni;j++){ temp2_cd=gsl_vector_get(Gic_cd,j);
			                     temp3_cd=1.0/gsl_vector_get(subvi,j);
								 temp4_cd=gsl_vector_get(subyi,j);
								 temp1_cd=temp1_cd+temp2_cd*temp4_cd*temp3_cd;
			                    }
			  d_mu=d_mu+temp1_cd;		 
		                       }
		 d_sigma=d_sigma*scale_c/(xi2*xi2);
		 d_sigma=1.0/d_sigma;
		 d_mu=d_sigma*(d_mu*scale_c/(xi2*xi2)-gsl_vector_get(lambda2_c,k));
		// printf("%12.9lf,%12.9lf\t",d_mu,d_sigma);
		 ck_cd=-d_mu/sqrt(d_sigma);
		 trancated_cd=gsl_ran_gaussian_tail(ccr,ck_cd,1.0);
		 dk_cd=sqrt(d_sigma)*trancated_cd+d_mu;
		 gsl_vector_set(d_c,k,dk_cd);
	                  }
	    gsl_matrix_set_row(d_p,iter,d_c);  			 
					 
 //------The full conditional for r 		 
		  gsl_matrix_set_all(temp_rCV,0.0);
	      gsl_vector_set_all(temp_rMU,0.0); 
		  for(i=0;i<N_train;i++){// printf("%d",i);printf("\n\n");
			  index1=addsubject[i],index2=addsubject[i+1];				
			  submatrix(z,index1,index2,0,q,subzi);			 
			  subvec(v_c,index1,index2,subvi); 
			  gsl_matrix_get_row(row_b,b_c,i);	
			  gsl_matrix_memcpy(zid,subzi);	  

			  for(j=0;j<q;j++){gsl_matrix_get_col(zdtempc_cr,zid,j);			                    
				               gsl_vector_scale(zdtempc_cr,gsl_vector_get(d_c,j));
							   gsl_matrix_set_col(zid,j,zdtempc_cr);
			                   }

			 gsl_matrix_memcpy(zdtemp,zid);
		      for(k=0;k<q-1;k++){
			     for(j=0,h=cumsump[k];j<(k+1),h<cumsump[k+1];j++,h++){//printf("zi=%d\t,bi=%d\t,F=%d\t",k+1,j,h);
					                                                   //  printf("%d\t",h);
																		//  printf("%d\t",k+1);
																		  // printf("%d\t",j);
					                                                  gsl_matrix_get_col(zdtempc_cr,zdtemp,k+1);
			                                                          gsl_vector_scale(zdtempc_cr,gsl_vector_get(row_b,j));
																      gsl_matrix_set_col(Fir,h,zdtempc_cr);
		                                                              }
				                                                   //printf("\n");
	                            }
	//printf("\n\n");
			  gsl_matrix_memcpy(viFi,Fir);			  
			  for(j=0;j<Ni;j++){gsl_matrix_get_row(viFitempr_cr,viFi,j);			                    
				               gsl_vector_scale(viFitempr_cr,1.0/gsl_vector_get(subvi,j));
							   gsl_matrix_set_row(viFi,j,viFitempr_cr);
			                   }

			  gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0,Fir,viFi,0.0,FvFtemp_cr);
			  gsl_matrix_add(temp_rCV,FvFtemp_cr);
			 //下面开始计算temp_rMU		
			  subvec(y,index1,index2,subyi);			
		      submatrix(x,index1,index2,0,p,subxi);	
			  gsl_blas_dgemv(CblasNoTrans,1.0,subxi,beta_c,0.0,xibeta);
	          gsl_blas_dgemv(CblasNoTrans,1.0,zid,row_b,0.0,zdbitemp_cr);		  

			 gsl_vector_sub(subyi,xibeta);
	          gsl_vector_sub(subyi,zdbitemp_cr);
			   subvec(v_c,index1,index2,subvi); 
			  gsl_vector_scale(subvi,xi1);
			  gsl_vector_sub(subyi,subvi);

			   gsl_blas_dgemv(CblasTrans,1.0,viFi,subyi,0.0,Fvytemp_cr);
			  gsl_vector_add(temp_rMU,Fvytemp_cr);  

		                       }
		
		  gsl_matrix_scale(temp_rCV,scale_c/(xi2*xi2));
          gsl_vector_scale(temp_rMU,scale_c/(xi2*xi2));
          for(h=0;h<dim_r;h++){dtemp1_cr=gsl_matrix_get(temp_rCV,h,h)+1.0/gsl_vector_get(hs_c,h);gsl_matrix_set(temp_rCV,h,h,dtemp1_cr);}
          solve_m(temp_rCV,dim_r,dim_r,solve1_cr);
          gsl_blas_dgemv(CblasNoTrans,1.0,solve1_cr,temp_rMU,0.0,temp_rMU1);//temp_bMU1=solve1_cb%*%temp_bMU
        // rmvnorm(ccr,temp_rMU1,solve1_cr,r_c);//产生多元正态随机数
		    rmvnorm_eigen(ccr,temp_rMU1,solve1_cr,r_c);//产生多元正态随机数
		  gsl_matrix_set_row(r_p,iter,r_c); 
		  vect_maxtrix1r(r_c, dim_r, q, GA_c);




		

	   if(strcmp(method,adlassoexp)==0){ 
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

					
                         }

		if(strcmp(method,lassoexp)==0){//lambda1
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
	gsl_vector_free(subvi);
	//bi	 
    gsl_matrix_free(temp_bCV);
	gsl_vector_free(temp_bMU);
	gsl_matrix_free(zid_cbi);
	gsl_vector_free(zdtempc_cbi);
	gsl_matrix_free(zidGA_cbi);
	gsl_matrix_free(vizidGA_cbi);
	gsl_vector_free(vizidGAr_cbi);
	gsl_vector_free(bi_c);
	gsl_matrix_free(solve1_cbi);
	gsl_vector_free(temp_bMU1);

	gsl_matrix_free(mtemp1_cb);
	gsl_permutation_free(b_per);
	gsl_matrix_free(solve1_cb);
	gsl_matrix_free(mtemp2_cb);
		

	//d
	gsl_vector_free(rbi_cd);
	gsl_matrix_free(Gi_cd);
	gsl_vector_free(Gic_cd);
	gsl_vector_free(zidrbi_cd);
	gsl_matrix_free(zid_cd);
	gsl_vector_free(zdtempc_cd);
	gsl_vector_free(Gbitemp_cd);
	gsl_vector_free(Gic1_cd);	

	//r
	gsl_vector_free(temp_u);
	gsl_matrix_free(temp_rCV);
	gsl_vector_free(temp_rMU);
	gsl_vector_free(zdtempc_cr);
	gsl_matrix_free(zid);
	gsl_matrix_free(zdtemp);
	gsl_matrix_free(Fir);
	gsl_matrix_free(viFi);	
	gsl_vector_free(viFitempr_cr);
	gsl_matrix_free(FvFtemp_cr);	
	gsl_vector_free(zdbitemp_cr);
	gsl_vector_free(xibeta);	
	gsl_vector_free(Fvytemp_cr);	
	

	gsl_matrix_free(mtemp1_crt);
	gsl_vector_free(vtemp2_cr);
	gsl_vector_free(vtemp1_cr);	
	gsl_matrix_free(solve1_cr);
	gsl_vector_free(temp_rMU1);
	gsl_matrix_free(mtemp2_cr);
	//scale
	 gsl_vector_free(vtemp1_csc);
	 gsl_vector_free(vtemp2_csc);
	//beta
	gsl_matrix_free(temp_betaCV);
    gsl_vector_free(temp_betaMU);
	gsl_matrix_free(subxi);
	gsl_vector_free(subyi);
	gsl_matrix_free(vixi);
	gsl_matrix_free(subzi);
	
	gsl_vector_free(vxtempr_cbe);
	gsl_vector_free(zdtempc_cbe);
	gsl_vector_free(Gbitemp_cbe);
	gsl_matrix_free(xvxtemp_cbe);
	gsl_vector_free(zdrbitemp_cbe);
	gsl_vector_free(xvytemp_cbe);

	gsl_matrix_free(solve1_cbe);
	gsl_vector_free(temp_betaMU1);
	gsl_matrix_free(wi_cbe);
	gsl_matrix_free(solveui_cbe);
	gsl_matrix_free(ui_cbe);
	gsl_matrix_free(xiui_cbe);
	gsl_matrix_free(xux_cbe);
	gsl_vector_free(xuytemp_cbe);
	//scale
	
return 0;
}


