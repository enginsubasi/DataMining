/**
 * @author: Engin Subasi
 * @e-mail: enginsubasi@gmail.com
 * @uni   : gtu.edu.tr
 * @prof. : Asst. Prof. Burcu Yilmaz
 */

/**
 * Within the data mining lesson(BIT561), the naive bayes algorithm has been
 * implemented in this program
 */

 /**
  * Library includes
  */
#include <stdio.h>      // printf
#include <stdlib.h>     // atio
#include <stdint.h>     // std int types
#include <string.h>     // strlen, strcmp, strcpy

#define TRAINDATANUM    14
#define TESTNUM         13-1

typedef struct Dataset{
    uint8_t age;
    char    income[8];
    char    student[8];
    char    creditRating[16];
    char    buysComputer[4];
}Dataset;

/**
 * Function Prototypes
 */
void fileToStruct(Dataset *inp,uint32_t *datasetLen,char *fileName);
void printAllDataset(Dataset *inp,uint32_t len);
int calcPoss(Dataset *inp,uint32_t len,int fnum,int fNumeric,char *inpF);
int calcPosCond(Dataset *inp,uint32_t len,int fnum,int fNumeric,
                                                        char *inpF ,char *cond);
void calcNaiveBayesVal(Dataset *inp, uint32_t len,Dataset test);


/**
 * Main Function
 */

 int main(void){

    char        dataFileName[16]= "dataset.txt"; // Data file name
    Dataset     myDataset[16];
    uint32_t    datasetLen = 0;

    fileToStruct(myDataset,&datasetLen,dataFileName);

    //printAllDataset(myDataset,datasetLen);

    printf("Naive Bayes test for %d. data at dataset.txt\r\n",TESTNUM+1);

    calcNaiveBayesVal(myDataset,TRAINDATANUM,myDataset[TESTNUM]);

    return(1);
 }

/**
 * @brief: This func. reads and parse data from file
 */
void fileToStruct(Dataset *inp,uint32_t *datasetLen,char *fileName){
    char fileBuf[128];
    char *fileBuffer;
    uint32_t cntr = 0;
    uint32_t i;
    FILE *fp;
    fp = fopen(fileName,"r");

    while( fgets(fileBuf, 255, (FILE*)fp) != NULL ){
        if(strlen(fileBuf)>2){
            fileBuffer = fileBuf;
            inp[cntr].age = atoi(fileBuffer);

            if( inp[cntr].age < 30 ){
                inp[cntr].age = 1;
            }else if( inp[cntr].age < 40 ){
                inp[cntr].age = 2;
            }else if( inp[cntr].age >= 40){
                inp[cntr].age = 3;
            }

            fileBuffer = strchr(fileBuffer,',') + 1;

            i = 0;
            do{
                inp[cntr].income[i] = fileBuffer[0];
                ++fileBuffer;
                ++i;
            }while( fileBuffer[0] != ',' );
            inp[cntr].income[i] = '\0';

            ++fileBuffer;

            i = 0;
            do{
                inp[cntr].student[i] = fileBuffer[0];
                ++fileBuffer;
                ++i;
            }while( fileBuffer[0] != ',' );
            inp[cntr].student[i] = '\0';

            ++fileBuffer;

            i = 0;
            do{
                inp[cntr].creditRating[i] = fileBuffer[0];
                ++fileBuffer;
                ++i;
            }while( fileBuffer[0] != ',' );
            inp[cntr].creditRating[i] = '\0';

            ++fileBuffer;

            i = 0;
            do{
                inp[cntr].buysComputer[i] = fileBuffer[0];
                ++fileBuffer;
                ++i;
            }while( fileBuffer[0] != '\n' );
            inp[cntr].buysComputer[i] = '\0';

            ++cntr;
        }
    }

    *datasetLen = cntr;

    fclose(fp);
}

/**
 * @brief: Prints all datas to console
 * @param: *inp: Dataset array pointer
 * @param: len: Dataset lenght
 */
void printAllDataset(Dataset *inp,uint32_t len){
    int i = 0;

    for( ; i < len ; ++i){
        printf("AGE: %d %s\t%s\t%s\t%s\r\n",inp[i].age,
                                    inp[i].income,
                                    inp[i].student,
                                    inp[i].creditRating,
                                    inp[i].buysComputer );

    }
}

/**
 * @brief: Calculate possibilities in all dataset
 */
int calcPoss(Dataset *inp,uint32_t len,int fnum,int fNumeric,char *inpF ){

    if( fnum == 0 ){
        int cntr = 0;
        int i = 0;
        for( ; i < len ; ++i ){
            if( inp[i].age == fNumeric ){
                ++cntr;
            }
        }
        return(cntr);
    }else if( fnum == 1 ){
        int cntr = 0;
        int i = 0;
        for( ; i < len ; ++i ){
            if( !strcmp( inp[i].income, inpF ) ){
                ++cntr;
            }
        }
        return(cntr);
    }else if( fnum == 2 ){
        int cntr = 0;
        int i = 0;
        for( ; i < len ; ++i ){
            if( !strcmp( inp[i].student, inpF ) ){
                ++cntr;
            }
        }
        return(cntr);
    }else if( fnum == 3 ){
        int cntr = 0;
        int i = 0;
        for( ; i < len ; ++i ){
            if( !strcmp( inp[i].creditRating, inpF ) ){
                ++cntr;
            }
        }
        return(cntr);
    }else if( fnum == 4 ){
        int cntr = 0;
        int i = 0;
        for( ; i < len ; ++i ){
            if( !strcmp( inp[i].buysComputer, inpF ) ){
                ++cntr;
            }
        }
        return(cntr);
    }else{
        return(-1);
    }
}

/**
 * @brief: Calculate possibilities in dataset pre. conditional
 */
int calcPosCond(Dataset *inp,uint32_t len,int fnum,int fNumeric,
                                                        char *inpF ,char *cond){

    if( fnum == 0 ){
        int cntr = 0;
        int i = 0;
        for( ; i < len ; ++i ){
            if( inp[i].age == fNumeric &&
                                        !strcmp( inp[i].buysComputer, cond ) ){
                ++cntr;
            }
        }
        return(cntr);
    }else if( fnum == 1 ){
        int cntr = 0;
        int i = 0;
        for( ; i < len ; ++i ){
            if( !strcmp( inp[i].income, inpF ) &&
                                        !strcmp( inp[i].buysComputer, cond ) ){
                ++cntr;
            }
        }
        return(cntr);
    }else if( fnum == 2 ){
        int cntr = 0;
        int i = 0;
        for( ; i < len ; ++i ){
            if( ( !strcmp( inp[i].student, inpF ) )  &&
                                        !strcmp( inp[i].buysComputer, cond ) ){
                ++cntr;
            }
        }
        return(cntr);
    }else if( fnum == 3 ){
        int cntr = 0;
        int i = 0;
        for( ; i < len ; ++i ){
            if( ( !strcmp( inp[i].creditRating, inpF ) ) &&
                                        !strcmp( inp[i].buysComputer, cond ) ){
                ++cntr;
            }
        }
        return(cntr);
    }else if( fnum == 4 ){
        int cntr = 0;
        int i = 0;
        for( ; i < len ; ++i ){
            if( ( !strcmp( inp[i].buysComputer, inpF ) ) &&
                                        !strcmp( inp[i].buysComputer, cond ) ){
                ++cntr;
            }
        }
        return(cntr);
    }else{
        return(-1);
    }
}

/**
 * @brief: Calculate and print Naive Bayes result
 */
void calcNaiveBayesVal(Dataset *inp, uint32_t len,Dataset test){

    int yesCnt  = calcPoss(inp,TRAINDATANUM,4,0,"Y");
    int noCnt   = calcPoss(inp,TRAINDATANUM,4,0,"N");

    int age = calcPosCond(inp,len,0,test.age,"","Y");
    int inc = calcPosCond(inp,len,1,0,test.income,"Y");
    int stu = calcPosCond(inp,len,2,0,test.student,"Y");
    int crr = calcPosCond(inp,len,3,0,test.creditRating,"Y");

    double yesPos = ( age * inc * stu * crr ) / ( ( yesCnt *
                   yesCnt * yesCnt * yesCnt ) * 1.0 );

    age = calcPosCond(inp,len,0,test.age,"","N");
    inc = calcPosCond(inp,len,1,0,test.income,"N");
    stu = calcPosCond(inp,len,2,0,test.student,"N");
    crr = calcPosCond(inp,len,3,0,test.creditRating,"N");

    double noPos = ( age * inc * stu * crr ) / ( ( noCnt *
                   noCnt * noCnt * noCnt ) * 1.0 );

    if( yesPos > noPos ){
        printf("Probably %f will buy computer\r\n",yesPos);
    }else{
        printf("Probably %f will not-buy computer\r\n",noPos);
    }

}
