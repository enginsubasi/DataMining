/**
 * @author: Engin Subasi
 * @e-mail: enginsubasi@gmail.com
 * @uni   : gtu.edu.tr
 * @prof. : Koksal Hocaoglu
 * @lesson: ELM669
 */

 /**
  * Library includes
  */
#include <time.h>   // srand rand
#include <math.h>   // to exp
#include <stdio.h>  // to printf
#include <stdlib.h> // to atof
#include <string.h> // to strlen strstr
#include <stdint.h> // to std int types

// Neuron couns definitions
#define KCNT    3 // OUTPUT NUM
#define JCNT    10 // +1 FOR BIAS
#define ICNT    5 // +1 FOR BIAS

// Iris dataset structures
typedef struct Dataset{
    double  features[ICNT];
    double  weightKJ[KCNT][JCNT];
    double  weightJI[JCNT][ICNT];
    double  diracok[KCNT];
    double  diracyj[JCNT];
    double  hlVals[JCNT];
    double  ouVals[KCNT];
    int     classType;
}Dataset;

// Desired outputs.
double d0[3] = { 1.0,0.0,0.0};
double d1[3] = { 0.0,1.0,0.0};
double d2[3] = { 0.0,0.0,1.0};

// This values gets from input file
double      EMAX;
double      LEARNRATE;
double      LRMIN;
uint32_t    LOOPMAX;
double      ALFA;
uint8_t     SHUFFLE;

// This functions initialize and fill dataset struct array
void datasetInit(Dataset *inp, uint32_t len);
void fileToStruct(Dataset *inp,uint32_t *datasetLen,char *fileName);
void shuffleDataset(Dataset *inp, uint32_t len);
void swapDataset(Dataset *inp, uint32_t i1, uint32_t i2);

// This function print all dataset on terminal
void printAllDataset(Dataset *inp,uint32_t len);

// This functions is calculations routines
double hardLimiter(double inp,double alfa);
double yjCalc(double *y ,double *w,double alfa);
double calcOk(double *y ,double *w,double alfa);
double errorCalc(double *desired, double *output);
void diracokCalc(Dataset *inp,uint32_t index, double *desired, double *output);
void diracyjCalc(Dataset *inp,uint32_t index, double *hlVals);
void kjUpdate(Dataset *inp,uint32_t index);
void jiUpdate(Dataset *inp,uint32_t index);

// This function runnig MLP-EBP algorithm
void runMLPEBP(Dataset *inp,uint32_t len);

////////////////////////////////////////////////////////////////////////////////
// Main function ///////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
int main(void){

    char        dataFileName[16]= "iris.data"; // Data file name
    Dataset     iris[256]; // iris data structure array
    uint32_t    irisLen; // iris data structure array lenght

    srand(0);

    fileToStruct(iris,&irisLen,dataFileName); // load iris data array from file
    datasetInit(iris,irisLen); // initialize weights

//    printAllDataset(iris,irisLen); // to print screen all data


    // Test function
    runMLPEBP(iris,irisLen);

    return(0);
}

/**
 * @brief: Initialize dataset weights
 * @param: Iris data structure array
 * @param: Iris data structure array lenght
 */
void datasetInit(Dataset *inp, uint32_t len){
    unsigned int p = 0, i = 0, j = 0;

    for( ; p < len ; ++p ){

            for( j = 0 ; j < JCNT ; ++j ){ // Features count
                for( i = 0 ; i < ICNT ; ++i ){ // Features count
                    inp[p].weightJI[j][i] = 1;
                }
            }

            for( i = 0 ; i < KCNT ; ++i ){ // Features count
                for( j = 0 ; j < JCNT ; ++j ){ // Features count
                    inp[p].weightKJ[i][j] = 1;
                }
            }

            if( p < 50 ){
                inp[p].classType = 0;
            }else if( p < 100 ){
                inp[p].classType = 1;
            }else if( p < 150 ){
                inp[p].classType = 2;
            }
    }

    // Shuffle dataset or no
    if( SHUFFLE ){
        shuffleDataset(inp,len);
    }

}

/**
 * @brief: This func. reads and parse data from file
 * @param: Iris data structure array
 * @param: Iris data structure array lenght
 * @param: Dataset file name string
 */
void fileToStruct(Dataset *inp,uint32_t *datasetLen,char *fileName){
    char fileBuffer[128];
    int cntr = 0;
    FILE *fp;
    fp = fopen(fileName,"r");

    while( fgets(fileBuffer, 255, (FILE*)fp) != NULL ){
        if( fileBuffer[0] != ':' ){
            inp[cntr].features[0] = atof(fileBuffer);
            inp[cntr].features[1] = atof(fileBuffer+4);
            inp[cntr].features[2] = atof(fileBuffer+8);
            inp[cntr].features[3] = atof(fileBuffer+12);
            inp[cntr].features[4] = 1; // to bias

            ++cntr;
        }else if(fileBuffer[1]==':'){ // PARAMETER INPUTS FROM FILE
            if(strstr(fileBuffer,"EMAX")){
                char* tempCharPtr = strstr(fileBuffer,"EMAX");
                tempCharPtr += 5;
                EMAX = atof(tempCharPtr);
            }else if(strstr(fileBuffer,"LEARNRATE")){
                char* tempCharPtr = strstr(fileBuffer,"LEARNRATE");
                tempCharPtr += 10;
                LEARNRATE = atof(tempCharPtr);
            }else if(strstr(fileBuffer,"LOOPMAX")){
                char* tempCharPtr = strstr(fileBuffer,"LOOPMAX");
                tempCharPtr += 8;
                LOOPMAX = atoi(tempCharPtr);
            }else if(strstr(fileBuffer,"ALFA")){
                char* tempCharPtr = strstr(fileBuffer,"ALFA");
                tempCharPtr += 5;
                ALFA = atof(tempCharPtr);
            }else if(strstr(fileBuffer,"SHUFFLE")){
                char* tempCharPtr = strstr(fileBuffer,"SHUFFLE");
                tempCharPtr += 8;
                SHUFFLE = atoi(tempCharPtr);
            }else if(strstr(fileBuffer,"LRMIN")){
                char* tempCharPtr = strstr(fileBuffer,"LRMIN");
                tempCharPtr += 6;
                LRMIN = atof(tempCharPtr);
            }else{

            }
        }
    }

    *datasetLen = cntr;
}

/**
 * @brief: Prints all datas to console
 * @param: *inp: Dataset array pointer
 * @param: len: Dataset lenght
 */
void printAllDataset(Dataset *inp,uint32_t len){
    int i = 0;

    for( ; i < len ; ++i){
        printf("%1.1f %1.1f %1.1f %1.1f %1.1f %d\r\n",  inp[i].features[0],
                                                        inp[i].features[1],
                                                        inp[i].features[2],
                                                        inp[i].features[3],
                                                        inp[i].features[4],
                                                        inp[i].classType);
    }
}

/**
 * hardLimiter= (    2/      )-1
 *              ( 1+exp(-av) )
 */
double hardLimiter(double inp,double alfa){
    return( ( ( 1 / ( 1 + exp( -1.0 * alfa * inp )) ) * 2 ) - 1 );
}

/**
 * @about: This function calculates Yj
 */
double yjCalc(double *y ,double *w,double alfa){
    double retval = 0;
    uint32_t i = 0;

    for( ; i < ICNT ; ++i ){
        retval += y[i] * w[i];
    }
    return(hardLimiter(retval,alfa));
}

/**
 * @about: This function calculates Ok
 */
double calcOk(double *y ,double *w,double alfa){
    double retval = 0;
    uint32_t i = 0;

    for( ; i < ICNT ; ++i ){
        retval += y[i] * w[i];
    }
    return(hardLimiter(retval,alfa));
}

/**
 * @about: This function calculates error per one input
 */
double errorCalc(double *desired, double *output){
    return( 0.5 * ( pow( ( desired[0] - output[0] ) , 2 )+
                                 pow( ( desired[1] - output[1] ) , 2 )+
                                 pow( ( desired[2] - output[2] ) , 2 ) ) );
}

/**
 * @about: This function calculates diracOK
 */
void diracokCalc(Dataset *inp,uint32_t index, double *desired, double *output){
    uint32_t k;

    for( k = 0 ; k < KCNT ; ++k ){
        inp[index].diracok[k] = ( 0.5 * ( desired[k] - output[k] ) *
        ( 1 - ( output[k] * output[k] ) ) );
    }
}

/**
 * @about: This function calculates diracYJ
 */
void diracyjCalc(Dataset *inp,uint32_t index, double *hlVals){
    uint32_t j, k;
    for( j = 0 ; j < JCNT ; ++j ){
        double tmp = 0;
        for( k = 0 ; k < KCNT ; ++k ){
            tmp += inp[index].diracok[k] * inp[index].weightKJ[k][j];
        }

        inp[index].diracyj[j] = 0.5 * ( 1 - hlVals[j] * hlVals[j] ) *
                                                               tmp ;
    }
}

/**
 * @about: This function updates Wkj (weight)
 */
void kjUpdate(Dataset *inp,uint32_t index){
    uint32_t j, k;

    for( j = 0 ; j < JCNT ; ++j ){
        for( k = 0 ; k < KCNT ; ++k ){
            inp[index].weightKJ[k][j] += LEARNRATE *
                                            inp[index].diracok[k] *
                                            inp[index].hlVals[j];

        }
    }
}

/**
 * @about: This function updates Tji (weight)
 */
void jiUpdate(Dataset *inp,uint32_t index){
    uint32_t j, k;

    for( j = 0 ; j < JCNT ; ++j ){
        for( k = 0 ; k < ICNT ; ++k ){
            inp[index].weightJI[j][k] += LEARNRATE *
                                            inp[index].diracyj[k] *
                                            inp[index].features[k];

        }
    }
}

/**
 * @about: Shuffle all dataset
 */
void shuffleDataset(Dataset *inp, uint32_t len){

    uint32_t i = 0;
    uint32_t index1,index2;

    for( ; i < 10000 ; ++i ){
        index1 = rand() % len;
        do{
            index2 = rand() % len;
        }while(index1==index2);

        swapDataset(inp,index1,index2);
    }
}

/**
 * @about: Swap dataset array elements
 */
void swapDataset(Dataset *inp, uint32_t i1, uint32_t i2){
    Dataset temp;
    memcpy(&temp, &inp[i1], sizeof(Dataset));
    memcpy(&inp[i1], &inp[i2], sizeof(Dataset));
    memcpy(&inp[i2], &temp, sizeof(Dataset));
}

/**
 * @about:
 */
void runMLPEBP(Dataset *inp,uint32_t len){
    // STEP 1 //////////////////////////////////////////////////////////
    double error = 0; // TOTAL ERROR PER LOOP
    uint32_t q = 0, p = 0; // COUNTERS
    uint32_t j = 0, k = 0;

    do{
        error = 0;
        for( p = 0 ; p < len ; ++p ){
            // STEP 2 //////////////////////////////////////////////////////////
            // Calculate hidden layer values and store at hlVals array
            for( j = 0 ; j < JCNT ; ++j ){
                inp[p].hlVals[j] = yjCalc(inp[p].features,inp[p].weightJI[j],ALFA);
            }

            inp[p].hlVals[j] = 1;

            // Calculate output layer values and store at ouVals array
            for( k = 0 ; k < KCNT ; ++k ){
                double tmp = 0;
                for( j = 0 ; j < JCNT ; ++j ){
                    tmp += inp[p].weightKJ[k][j]*inp[p].hlVals[j];
                }
                inp[p].ouVals[k] = hardLimiter(tmp,ALFA);
            }

            // STEP 3 & 4 //////////////////////////////////////////////////////
            if( inp[p].classType == 0 ){ // FIRST CLASS
                error += errorCalc(d0,inp[p].ouVals);

                diracokCalc(inp,p,d0,inp[p].ouVals);
                diracyjCalc(inp,p,inp[p].hlVals);

            }else if( inp[p].classType == 1 ){ // SECOND CLASS
                error += errorCalc(d1,inp[p].ouVals);

                diracokCalc(inp,p,d1,inp[p].ouVals);
                diracyjCalc(inp,p,inp[p].hlVals);

            }else if( inp[p].classType == 2 ){ // THIRD CLASS
                error += errorCalc(d2,inp[p].ouVals);

                diracokCalc(inp,p,d2,inp[p].ouVals);
                diracyjCalc(inp,p,inp[p].hlVals);
            }

            // STEP 5 //////////////////////////////////////////////////////////
            kjUpdate(inp,p);
            jiUpdate(inp,p);
        }

        printf("%2.4f\r\n",error);
        //printf("LRate:%f\r\n",LEARNRATE);

        if(LRMIN<LEARNRATE){
            LEARNRATE /= 2.0;
        }

        ++q;
    }while( ( error > EMAX ) && ( q < LOOPMAX ) );

    printf("\r\nTotal epoch count: %d\r\n",q);
    printf("Last error: %2.4f\r\n",error);
    printf("Last lrate: %2.4f\r\n",LEARNRATE);
}


