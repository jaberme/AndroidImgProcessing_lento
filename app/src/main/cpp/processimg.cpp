#include <string.h>
#include <jni.h>
#include <pthread.h>
#include <omp.h>
#include <android/log.h>
#include <stdbool.h>
#include <vector>

using namespace std;

#define MAX_NUM_THREADS 16

typedef struct paramST {  // data structure holding all operands needed by a worker thread
    const unsigned char * data;
    int * pixels;
    int width;
    int height;
    int my_id;
    int nthr;
    signed char * matrix;
    int divisor;
} paramST;

// return RGB value from YUV
static int convertYUVtoRGB(int y, int u, int v)
{
    int r,g,b;

    r = y + (int)1.14f*v;
    g = y - (int)(0.395f*u +0.581f*v);
    b = y + (int)2.033f*u;
    r = r>255? 255 : r<0 ? 0 : r;
    g = g>255? 255 : g<0 ? 0 : g;
    b = b>255? 255 : b<0 ? 0 : b;
    return 0xff000000 | (r<<16) | (g<<8) | b; // one byte for alpha, one for red, one for green, one for blue
}

// process the whole image
static void convertYUV420_NV21toRGB8888old(const unsigned char * data, int * pixels, int width, int height)
// pixels must have room for width*height ints, one per pixel
{
    int size = width*height;
    int u, v, y1, y2, y3, y4;
    int i, k;

    // i traverses Y
    // k traverses U and V
    for(i=0, k=0; i < size; i+=2, k+=2) { // each 4x4 region of the bitmap has the same (u,v) but different y1,y2,y3,y4
        y1 = data[i  ]&0xff;
        y2 = data[i+1]&0xff;
        y3 = data[width+i  ]&0xff;
        y4 = data[width+i+1]&0xff;

        v = data[size+k  ]&0xff;
        u = data[size+k+1]&0xff;
        u = u-128;
        v = v-128;

        pixels[i  ] = convertYUVtoRGB(y1, u, v);
        pixels[i+1] = convertYUVtoRGB(y2, u, v);
        pixels[width+i  ] = convertYUVtoRGB(y3, u, v);
        pixels[width+i+1] = convertYUVtoRGB(y4, u, v);

        if (i!=0 && (i+2)%width==0)
            i+=width;
    }
}

// process the whole image
static void convertYUV420_NV21toRGB8888(const unsigned char * data, int * pixels, int width, int height)
// pixels must have room for width*height ints, one per pixel
{
    int size = width*height;
    int u, v, y1, y2, y3, y4;
    int i;

    // row1 and row2 traverses Y
    // uvp traverses U and V
    int queda=width;
//         for(i=0, k=0; i < size; i+=2, k+=2) { // each 4x4 region of the bitmap has the same (u,v) but different y1,y2,y3,y4
    char * row1=(char*)data;
    char * row2=row1+width;
    char * uvp= row1+size;
    int * pixrow1 = pixels;
    int * pixrow2 = pixels + width;
    for(i=0; i < (size>>2); i++) { // one iteration for each 2x2 region of the bitmap has the same (u,v) but different y1,y2,y3,y4

        y1 = (*row1++)&0xff;
        y2 = (*row1++)&0xff;
        y3 = (*row2++)&0xff;
        y4 = (*row2++)&0xff;

        v = (*uvp++)&0xff;
        u = (*uvp++)&0xff;
        u = u-128;
        v = v-128;

        *pixrow1++ = convertYUVtoRGB(y1, u, v);
        *pixrow1++ = convertYUVtoRGB(y2, u, v);
        *pixrow2++ = convertYUVtoRGB(y3, u, v);
        *pixrow2++ = convertYUVtoRGB(y4, u, v);

        queda-=2;
        if (!queda){queda=width;row1+=width;row2+=width;pixrow1+=width;pixrow2+=width;}
    }
}


// Process a chunk of the image
void *convertYUV420_NV21toRGB8888Chunk(void *args)
// pixels must have room for width*height ints, one per pixel. See https://en.wikipedia.org/wiki/YUV
{
    paramST *param = (paramST *)args;

    const unsigned char * data = param->data;
    int * pixels = param->pixels;
    int width = param->width;
    int height = param->height;
    int my_id = param->my_id;
    int nThreads = param->nthr;

    int size = width*height;
    int u=0, v=0, y1, y2, y3, y4;
    int myEnd, myOff;
    int i, k;

    int myInitRow = (int)(((float)my_id/(float)nThreads)*((float)height/2.0f));
    int nextThreadRow = (int)(((float)(my_id+1)/(float)nThreads)*((float)height/2.0f));
    myOff = 2*width*myInitRow;
    myEnd = 2*width*nextThreadRow;
    int myUVOff = size+myInitRow*width;
    for(i=myOff, k=myUVOff; i < myEnd; i+=2, k+=2) { // each 4x4 region of the bitmap has the same (u,v) but different y1,y2,y3,y4
        y1 = data[i  ]&0xff;
        y2 = data[i+1]&0xff;
        y3 = data[width+i  ]&0xff;
        y4 = data[width+i+1]&0xff;

        v = data[k  ]&0xff;
        u = data[k+1]&0xff;
        u = u-128;
        v = v-128;

        pixels[i  ] = convertYUVtoRGB(y1, u, v);
        pixels[i+1] = convertYUVtoRGB(y2, u, v);
        pixels[width+i  ] = convertYUVtoRGB(y3, u, v);
        pixels[width+i+1] = convertYUVtoRGB(y4, u, v);

        if (i!=0 && (i+2)%width==0)
            i+=width;
    }
    pthread_exit(NULL);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%% GREY SCALE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/


static void convertYUV420_NV21toGrey(const unsigned char * data, int * pixels, int width, int height)
{
    int size = width*height;
    int y1;
    int i;
    for(i=0; i < size; i++) {
        y1 = data[i  ]&0xff;
        pixels[i] = 0xff000000 | (y1<<16) | (y1<<8) | y1;
    }
}

// Process a chunk of the image
void *convertYUV420_NV21toGreyScaleChunk(void *args)
// pixels must have room for width*height ints, one per pixel. See https://en.wikipedia.org/wiki/YUV
{
    paramST *param = (paramST *)args;

    const unsigned char * data = param->data;
    int * pixels = param->pixels;
    int width = param->width;
    int height = param->height;
    int my_id = param->my_id;
    int nThreads = param->nthr;

    int y1;
    int myInitRow = (int)(((float)my_id/(float)nThreads)*((float)height/2.0f));
    int nextThreadRow = (int)(((float)(my_id+1)/(float)nThreads)*((float)height/2.0f));
    int myEnd, myOff;
    myOff = 2*width*myInitRow;
    myEnd = 2*width*nextThreadRow;
    int i;
    for(i=myOff; i < myEnd;i++) {
        y1 = data[i  ]&0xff;
        pixels[i] = 0xff000000 | (y1<<16) | (y1<<8) | y1;
    }
    pthread_exit(NULL);
}

// Process the whole image in parallel using nthr pthreads
static void convertYUV420_NV21toGreyScaleParallel(const unsigned char * data, int * pixels, int width, int height, int nthr)
{
    pthread_t th[MAX_NUM_THREADS];
    paramST params[MAX_NUM_THREADS];
    int status;
    int my_nthr = nthr;
    int i;

    if (my_nthr > MAX_NUM_THREADS)
        my_nthr = MAX_NUM_THREADS;
    for (i=0; i < my_nthr; i++) {
        params[i].data = data;
        params[i].pixels = pixels;
        params[i].width = width;
        params[i].height = height;
        params[i].my_id = i;
        params[i].nthr = my_nthr;
        pthread_create(&(th[i]), NULL, convertYUV420_NV21toGreyScaleChunk, (void *)&(params[i]));
    }
    for (i=0; i < nthr; i++) {
        pthread_join(th[i], NULL);
    }
}

// process the whole image
static void convertYUV420_NV21toGreyScale_OMP(const unsigned char * data, int * pixels, int width, int height)
// pixels must have room for width*height ints, one per pixel
{
    int size = width*height;
    int y1;
    int i, y, yrow, ycol;

    // "y" traverses the set of four "Y"-pixels that share components U and V.
#pragma omp parallel for schedule(guided) private(y1)
    for (i=0; i<size; i++) {
        y1 = data[i]&0xff;
        pixels[i] = 0xff000000 | (y1<<16) | (y1<<8) | y1;
    }
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%% CONVOLUTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/

// comprueba si el índice i de un array es el borde de una matriz con dimensiones width, height
static bool esBorde(int i,int width, int height){
    return //i < width || i >= (height-1)*(width) ||  // es primera fila o ultima fila
            (i%(width)) % (width -1) == 0 || (i%(width)) % (width) == 0; // es borde izquierdo o derecho
}

static void convertYUV420_NV21toConvolution(const unsigned char * data, int * pixels, signed char * matrix, int divisor, int width, int height)
{
    int size = width*height;
    int color;
    int i,j,x, nextRow, previousRow;
    // recorre la imagen excepto los bordes

    __android_log_print(ANDROID_LOG_INFO, "APDM_", "k0 %d k4 %d k7 %d", matrix[0], matrix[4], matrix[7]);

    for(i=1;i < height - 1 ;i++){
        for( j=1; j < width -1; j++){
            x = i * width + j;
            nextRow = x + width;
            previousRow = x - width;

            color = matrix[0] * ((int) data[previousRow-1]&0xff) +
                    matrix[1] * ((int)data[previousRow]&0xff) +
                    matrix[2] * ((int)data[previousRow+1]&0xff) +
                    matrix[3] * ((int)data[x-1]&0xff) +
                    matrix[4] * ((int)data[x]&0xff) +
                    matrix[5] * ((int)data[x+1]&0xff) +
                    matrix[6] * ((int)data[nextRow-1]&0xff) +
                    matrix[7] * ((int)data[nextRow]&0xff) +
                    matrix[8] * ((int)data[nextRow+1]&0xff)/divisor;

            color = color>255? 255 : color<0 ? 0 : color;
            // asigna el mismo color en RGB y valor alfa a 1
            pixels[x] = 0xff000000 | (color<<16) | (color<<8) | color;
        }
    }
}


// Process a chunk of the image
void *convertYUV420_NV21toConvolutionChunk(void *args)
// pixels must have room for width*height ints, one per pixel. See https://en.wikipedia.org/wiki/YUV
{
    paramST *param = (paramST *)args;

    const unsigned char * data = param->data;
    int * pixels = param->pixels;
    int width = param->width;
    int height = param->height;
    int my_id = param->my_id;
    int nThreads = param->nthr;
    signed char * matrix = param->matrix;
    int divisor = param->divisor;

    int size = width*height;
    int color;
    int offset = size;

    int myInitRow = (int)(((float)my_id/(float)nThreads)*((float)height/2.0f));
    int nextThreadRow = (int)(((float)(my_id+1)/(float)nThreads)*((float)height/2.0f));
    int myEnd, myOff;
    int i,j,x,nextRow,previousRow;

    myOff = 2*width*myInitRow;
    myEnd = 2*width*nextThreadRow;
    myInitRow = 2*myInitRow;
    int myheight = myInitRow + (myEnd - myOff)/width;


    if (my_id == 0) myInitRow += 1;             // Si es el primer Thread salta la primera fila
    if( my_id == nThreads-1) myheight -= 1;      // Si es el ultimo Thread salta la ultima fila


    for(i=myInitRow;i < myheight;i++){
        for( j=1; j < width -1; j++){
            x = i * width + j;
            nextRow = x + width;
            previousRow = x - width;

            color = matrix[0] * ((int) data[previousRow-1]&0xff) +
                    matrix[1] * ((int)data[previousRow]&0xff) +
                    matrix[2] * ((int)data[previousRow+1]&0xff) +
                    matrix[3] * ((int)data[x-1]&0xff) +
                    matrix[4] * ((int)data[x]&0xff) +
                    matrix[5] * ((int)data[x+1]&0xff) +
                    matrix[6] * ((int)data[nextRow-1]&0xff) +
                    matrix[7] * ((int)data[nextRow]&0xff) +
                    matrix[8] * ((int)data[nextRow+1]&0xff)/divisor;

            color = color>255? 255 : color<0 ? 0 : color;
            // asigna el mismo color en RGB y valor alfa a 1
            pixels[x] = 0xff000000 | (color<<16) | (color<<8) | color;
        }
    }

    pthread_exit(NULL);
}

// Process the whole image in parallel using nthr pthreads
static void convertYUV420_NV21toConvolutionParallel(const unsigned char * data, int * pixels,int divisor, signed char* matrix, int width, int height, int nthr)
{
    pthread_t th[MAX_NUM_THREADS];
    paramST params[MAX_NUM_THREADS];
    int status;
    int my_nthr = nthr;
    int i;

    if (my_nthr > MAX_NUM_THREADS)
        my_nthr = MAX_NUM_THREADS;
    for (i=0; i < my_nthr; i++) {
        params[i].data = data;
        params[i].pixels = pixels;
        params[i].width = width;
        params[i].height = height;
        params[i].my_id = i;
        params[i].nthr = my_nthr;
        params[i].divisor = divisor;
        params[i].matrix = matrix;
        pthread_create(&(th[i]), NULL, convertYUV420_NV21toConvolutionChunk, (void *)&(params[i]));
    }
    for (i=0; i < nthr; i++) {
        pthread_join(th[i], NULL);
    }
}

// process the whole image
static void convertYUV420_NV21toConvolution_OMP(const unsigned char * data, int * pixels,int divisor, signed char* matrix, int width, int height)
// pixels must have room for width*height ints, one per pixel
{
    int size = width*height;
    int color;
    int i,j;
    int nextRow, previousRow, x;

    // "y" traverses the set of four "Y"-pixels that share components U and V.
#pragma omp parallel for schedule(guided) private(color,nextRow,previousRow,x)
    // recorre la imagen excepto la primera fila y ultima fila (que no se calculan)
    for( i=1;i < height - 1 ;i++){
        for( j=1; j < width -1; j++){
            x = i * width + j;
            nextRow = x + width;
            previousRow = x - width;
            // calcula el valor de cada posicion del array operando con la matriz dada
            color = (matrix[0] * ((int) data[previousRow-1]&0xff) +
                    matrix[1] * ((int)data[previousRow]&0xff) +
                    matrix[2] * ((int)data[previousRow+1]&0xff) +
                    matrix[3] * ((int)data[x-1]&0xff) +
                    matrix[4] * ((int)data[x]&0xff) +
                    matrix[5] * ((int)data[x+1]&0xff) +
                    matrix[6] * ((int)data[nextRow-1]&0xff) +
                    matrix[7] * ((int)data[nextRow]&0xff) +
                    matrix[8] * ((int)data[nextRow+1]&0xff))/divisor;

            color = color>255? 255 : color<0 ? 0 : color;
            // asigna el mismo color en RGB y valor alfa a 1
            pixels[x] = 0xff000000 | (color<<16) | (color<<8) | color;
        }
    }
}




/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/



// process the whole image
static void convertYUV420_NV21toRGB8888_OMP(const unsigned char * data, int * pixels, int width, int height)
// pixels must have room for width*height ints, one per pixel
{
    int size = width*height;
    int u, v, y1, y2, y3, y4;
    int i, k, y, yrow, ycol;

    // "y" traverses the set of four "Y"-pixels that share components U and V.
#pragma omp parallel for schedule(guided) private(yrow, ycol, i, k, u, v, y1, y2, y3, y4)
    for (y=0; y<size/4; y++) {
        //__android_log_print(ANDROID_LOG_INFO, "HOOKnative", "num threads %d", omp_get_num_threads());
        yrow = (int)(y/(width/2));              // row of the first "y" set
        ycol = y % (width/2);                   // col of the first "y" set
        i = 2*(yrow*width + ycol);              // offset in array "data" of first "Y" of "y" set
        k = size + yrow*width + 2*ycol;         // offset in array "data" of the "U" component of "y" set

        y1 = data[i  ]&0xff;
        y2 = data[i+1]&0xff;
        y3 = data[width+i  ]&0xff;
        y4 = data[width+i+1]&0xff;

        v = data[k  ]&0xff;
        u = data[k+1]&0xff;
        u = u-128;
        v = v-128;

        pixels[i  ] = convertYUVtoRGB(y1, u, v);
        pixels[i+1] = convertYUVtoRGB(y2, u, v);
        pixels[width+i  ] = convertYUVtoRGB(y3, u, v);
        pixels[width+i+1] = convertYUVtoRGB(y4, u, v);
    }
}

// Process the whole image in parallel using nthr pthreads
static void convertYUV420_NV21toRGB8888Parallel(const unsigned char * data, int * pixels, int width, int height, int nthr)
{
    pthread_t th[MAX_NUM_THREADS];
    paramST params[MAX_NUM_THREADS];
    int status;
    int my_nthr = nthr;
    int i;

    if (my_nthr > MAX_NUM_THREADS)
        my_nthr = MAX_NUM_THREADS;
    for (i=0; i < my_nthr; i++) {
        params[i].data = data;
        params[i].pixels = pixels;
        params[i].width = width;
        params[i].height = height;
        params[i].my_id = i;
        params[i].nthr = my_nthr;
        pthread_create(&(th[i]), NULL, convertYUV420_NV21toRGB8888Chunk, (void *)&(params[i]));
    }
    for (i=0; i < nthr; i++) {
        pthread_join(th[i], NULL);
    }
}

// Native function called from Java to process a image
extern "C"
void Java_es_ual_bermejo_DemoGoingFaster_MainActivity_YUVtoNative( JNIEnv* env, jobject thiz,
                                                                      jint tipo,
                                                                      jbyteArray data,
                                                                      jintArray result,
                                                                      jint divisor,
                                                                      jbyteArray matrix,
                                                                      jint width, jint height)
{
    unsigned char *cData;
    int *cResult;

    cData = (unsigned char *) env->GetByteArrayElements(data,NULL); // While arrays of objects must be accessed one entry at a time, arrays of primitives can be read and written directly as if they were declared in C.   http://developer.android.com/training/articles/perf-jni.html
    if (cData==NULL) __android_log_print(ANDROID_LOG_INFO, "HOOKnative", "Can't not get data array reference");
    else
    {
        cResult= env->GetIntArrayElements(result,NULL);
        if (cResult==NULL) __android_log_print(ANDROID_LOG_INFO, "HOOKnative", "Can't not get result array reference");
        else
        {
            // operates on data
            if(tipo==0){
                convertYUV420_NV21toRGB8888(cData,cResult,width,height);
            }else if(tipo==1){
                convertYUV420_NV21toGrey(cData,cResult,width,height);
            }else if(tipo==2){
                signed char* cMatrix = (signed char *)env->GetByteArrayElements(matrix,NULL);
                convertYUV420_NV21toConvolution(cData,cResult,cMatrix,divisor,width,height);
                if (cMatrix!=NULL) env->ReleaseByteArrayElements(matrix,(jbyte *)cMatrix,0);
            }

        }
    }
    if (cResult!=NULL) env->ReleaseIntArrayElements(result,cResult,0); // care: errors in the type of the pointers are detected in runtime
    if (cData!=NULL) env->ReleaseByteArrayElements(data, (jbyte *) cData, 0); // release must be done even when the original array is not copied into cDATA

    // Log from native (need 'ndk {ldLibs "log"}' in app/build.gradle
    //__android_log_print(ANDROID_LOG_INFO, "HOOKnative", "Log from native function: %s", buf);

}

// Native function called from Java to process a image in parallel using nthreads threads
extern "C"
void Java_es_ual_bermejo_DemoGoingFaster_MainActivity_YUVtoNativeParallel( JNIEnv* env, jobject thiz,
                                                                              jint tipo,
                                                                              jbyteArray data,
                                                                              jintArray result,jint divisor, jbyteArray matrix,
                                                                              jint width, jint height, jint nthreads)
{
    unsigned char *cData;
    int *cResult;

    cData = (unsigned char *) env->GetByteArrayElements(data,NULL); // While arrays of objects must be accessed one entry at a time, arrays of primitives can be read and written directly as if they were declared in C.   http://developer.android.com/training/articles/perf-jni.html
    if (cData==NULL) __android_log_print(ANDROID_LOG_INFO, "HOOKnative", "Can't get data array reference");
    else
    {
        cResult= env->GetIntArrayElements(result,NULL);
        if (cResult==NULL) __android_log_print(ANDROID_LOG_INFO, "HOOKnative", "Can't get result array reference");
        else
        {
            // operates on data
            if(tipo==0){
                convertYUV420_NV21toRGB8888Parallel(cData,cResult,width,height,nthreads);
            }else if(tipo==1){
                convertYUV420_NV21toGreyScaleParallel(cData,cResult,width,height,nthreads);
            }else if(tipo==2){
                signed char* cMatrix = (signed char *)env->GetByteArrayElements(matrix,NULL);
                convertYUV420_NV21toConvolutionParallel(cData,cResult,divisor, cMatrix,width,height,nthreads);
                if (cMatrix!=NULL) env->ReleaseByteArrayElements(matrix,(jbyte *)cMatrix,0);
            }
        }
    }
    if (cResult!=NULL) env->ReleaseIntArrayElements(result,cResult,0); // care: errors in the type of the pointers are detected at runtime
    if (cData!=NULL) env->ReleaseByteArrayElements(data, (jbyte *) cData, 0); // release must be done even when the original array is not copied into cDATA
}

// Native function called from Java to process an image in parallel using nthreads threads
extern "C"
void Java_es_ual_bermejo_DemoGoingFaster_MainActivity_YUVtoNativeParallelOMP( JNIEnv* env, jobject thiz,
                                                                                 jint tipo,
                                                                                 jbyteArray data,
                                                                                 jintArray result, jint divisor, jbyteArray matrix,
                                                                                 jint width, jint height, jint nthreads)
{
    unsigned char *cData;
    int *cResult;

    cData = (unsigned char *) env->GetByteArrayElements(data,NULL); // While arrays of objects must be accessed one entry at a time, arrays of primitives can be read and written directly as if they were declared in C.   http://developer.android.com/training/articles/perf-jni.html
    if (cData==NULL) __android_log_print(ANDROID_LOG_INFO, "HOOKnative", "Can't get data array reference");
    else
    {
        cResult= env->GetIntArrayElements(result,NULL);
        if (cResult==NULL) __android_log_print(ANDROID_LOG_INFO, "HOOKnative", "Can't get result array reference");
        else
        {
            omp_set_num_threads(nthreads);
            // operates on data
            if(tipo==0){
                convertYUV420_NV21toRGB8888_OMP(cData,cResult,width,height);
            }else if(tipo==1){
                convertYUV420_NV21toGreyScale_OMP(cData,cResult,width,height);
            }else if(tipo==2){
                signed char* cMatrix = (signed char *)env->GetByteArrayElements(matrix,NULL);
                convertYUV420_NV21toConvolution_OMP(cData,cResult,divisor, cMatrix,width,height);
                if (cMatrix!=NULL) env->ReleaseByteArrayElements(matrix,(jbyte *)cMatrix,0);
            }
        }
    }
    if (cResult!=NULL) env->ReleaseIntArrayElements(result,cResult,0); // care: errors in the type of the pointers are detected at runtime
    if (cData!=NULL) env->ReleaseByteArrayElements(data, (jbyte *) cData, 0); // release must be done even when the original array is not copied into cDATA
}



