package es.ual.bermejo.DemoGoingFaster;

import android.util.Log;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Created by corbera on 23/10/15.
 */
public class YUVtoParallel {

    private ExecutorService exSrv;
    private int nThreads;

    public static final int RGB = 0;
    public static final int GREYSCALE = 1;
    public static final int CONVOLUTION = 2;

    YUVtoParallel(int _nthreads) {
        nThreads = _nthreads;
        exSrv = Executors.newFixedThreadPool(nThreads);

    }

    public void shutdown() {
        exSrv.shutdown();
    }

    private static int convertYUVtoRGB(int y, int u, int v)
    {
        int r,g,b;

        r = y + (int)1.14f*v;
        g = y - (int)(0.395f*u +0.581f*v);
        b = y + (int)2.033f*u;
        r = r>255? 255 : r<0 ? 0 : r;
        g = g>255? 255 : g<0 ? 0 : g;
        b = b>255? 255 : b<0 ? 0 : b;
        return 0xff000000 | (r<<16) | (g<<8) | b; // one byte for alpha, one for blue, one for green, one for red
    }

    private static void convertYUV420_NV21toRGB8888(byte [] data, int [] pixels, int width, int height, int my_id, int nThreads)
    // pixels must have room for width*height ints, one per pixel. See https://en.wikipedia.org/wiki/YUV
    {
        int size = width*height;
        int offset = size;
        int u=0, v=0, y1, y2, y3, y4;
        int myEnd, myOff;

        int myInitRow = (int)(((float)my_id/(float)nThreads)*((float)height/2.0f));
        int nextThreadRow = (int)(((float)(my_id+1)/(float)nThreads)*((float)height/2.0f));
        myOff = 2*width*myInitRow;
        myEnd = 2*width*nextThreadRow;
        int myUVOff = offset+myInitRow*width;
        //Log.d("HOOK", "convertYUV420_NV21toRGB8888: id "+my_id+" nth "+nThreads+" " + width + "x" + height + " initrow "+myInitRow+" nextrow "+nextThreadRow+" my_off " + myOff + " my_end " + myEnd);
        for(int i=myOff, k=myUVOff; i < myEnd; i+=2, k+=2) { // each 4x4 region of the bitmap has the same (u,v) but different y1,y2,y3,y4
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
    }

    public static void convertYUV420_NV21toGrey(byte [] data, int [] pixels, int width, int height, int my_id, int nThreads)
    {
        int size = width*height;
        int y1;
        int offset = size;
        int myInitRow = (int)(((float)my_id/(float)nThreads)*((float)height/2.0f));
        int nextThreadRow = (int)(((float)(my_id+1)/(float)nThreads)*((float)height/2.0f));
        int myEnd, myOff;
        myOff = 2*width*myInitRow;
        myEnd = 2*width*nextThreadRow;
        for(int i=myOff; i < myEnd;i++) {
            y1 = data[i  ]&0xff;
            pixels[i] = 0xff000000 | (y1<<16) | (y1<<8) | y1;
        }
    }

    public static void convertYUV420_NV21toConvolution(byte [] data, byte[] matrix, int divisor,
                                                       int [] procImage, int width, int height, int my_id, int nThreads){

        int size = width*height;
        int color;
        int offset = size;
        int myInitRow = (int)(((float)my_id/(float)nThreads)*((float)height/2.0f));
        int nextThreadRow = (int)(((float)(my_id+1)/(float)nThreads)*((float)height/2.0f));
        int myEnd, myOff;
        myOff = 2*width*myInitRow;
        myEnd = 2*width*nextThreadRow;

        myInitRow = 2*myInitRow;
        int myheight = myInitRow + (myEnd - myOff)/width;

        int i,j,x,nextRow,previousRow;
        if (my_id == 0){ myInitRow += 1;}             // Si es el primer Thread salta la primera fila
        if( my_id == nThreads-1) {myheight -= 1;}      // Si es el ultimo Thread salta la ultima fila

        // recorre la imagen excepto los bordes
        for(i=myInitRow;i < myheight;i++){
            for(j=1; j < width -1; j++){
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
                procImage[x] = 0xff000000 | (color<<16) | (color<<8) | color;
            }
        }
    }

    private final class myJavaWorker implements Callable<Void> {
        private byte [] data;
        private int [] pixels;
        private int width;
        private int height;
        private int id;
        private int nth;
        private int tipo;
        private byte[] matrix;
        private int divisor;

        public myJavaWorker(int tipo,byte [] _data, byte[] matrix, int divisor, int [] _pixels, int _width, int _height, int _th_id, int _nth) {
            data = _data;
            pixels = _pixels;
            width = _width;
            height = _height;
            id = _th_id;
            nth = _nth;
            this.tipo = tipo;
            this.matrix = matrix;
            this.divisor = divisor;
        }

        public Void call(){
            switch (this.tipo) {
                case RGB:
                    convertYUV420_NV21toRGB8888(data, pixels, width, height, id, nth);
                    break;
                case GREYSCALE:
                    convertYUV420_NV21toGrey(data, pixels, width, height, id, nth);
                    break;
                case CONVOLUTION:
                    convertYUV420_NV21toConvolution(data, matrix, divisor, pixels, width, height, id, nth);
                    break;
            }

            return null;
        }

    }

    public void convertYUV420_NV21to_parallel(int tipo, byte [] data, int [] pixels, int width, int height){
        List<Callable<Void>> todoTasks;
        todoTasks = new ArrayList<Callable<Void>>(nThreads);

        for (int i=0; i < nThreads; i++) {
            todoTasks.add(i, new myJavaWorker(tipo, data, null, 1, pixels, width, height, i, nThreads));
        }
        try {
            exSrv.invokeAll(todoTasks);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    // método para realizar el computo en paralelo de Convolution con la matrix y divisor en parametros
    public void convertYUV420_NV21to_parallel(int tipo, byte [] data, byte[]matrix, int divisor, int [] pixels, int width, int height){
        List<Callable<Void>> todoTasks;
        todoTasks = new ArrayList<Callable<Void>>(nThreads);

        for (int i=0; i < nThreads; i++) {
            todoTasks.add(i, new myJavaWorker(tipo, data, matrix, divisor, pixels, width, height, i, nThreads));
        }
        try {
            exSrv.invokeAll(todoTasks);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    /* Method that rotates and scales a RGB image (in[hIn x wIn] --> out[hOut x wOut] rotated "angle" degrees) */
    public static void downscaleAndRotateImage(int[] in, int[] out, int wIn, int hIn, int wOut, int hOut, int angle) {
        if (((angle == 0 || angle == 180) && (wIn < wOut || hIn < hOut)) ||
                (wIn < hOut || hIn < wOut)){
            Log.i("HOOK",""+angle + ":"+wIn+":"+wOut+ ":"+hIn+":"+hOut);
            throw new RuntimeException();

        }

        float strideW = (angle == 0 || angle == 180) ? (float)wIn / (float)wOut : (float)wIn / (float)hOut;
        float strideH = (angle == 0 || angle == 180) ? (float)hIn / (float)hOut : (float)hIn / (float)wOut;
        int maxsize = wOut*hOut;
        int cont = 0;
        switch (angle) {
            case 0:
                for (float j=0; j<hIn; j+=strideH) {
                    for (float i=0; i<wIn; i+=strideW) {
                        out[cont++] = in[(int)j*wIn + (int)i];
                        if (cont!= 0 && cont%wOut == 0)
                            break;
                    }
                    if (cont == maxsize)
                        return;
                }
                break;
            case 90:
                for (float i=0; i<wIn; i+=strideW) {
                    for (float j=hIn-1; j>=0; j-=strideH) {
                        out[cont++] = in[(int)j*wIn + (int)i];
                        if (cont!= 0 && cont%wOut == 0)
                            break;
                    }
                    if (cont == maxsize)
                        return;
                }
                break;
            case 180:
                for (float j=hIn-1; j>=0; j-=strideH) {
                    for (float i=wIn-1; i>=0; i-=strideW) {
                        out[cont++] = in[(int)j*wIn + (int)i];
                        if (cont!= 0 && cont%wOut == 0)
                            break;
                    }
                    if (cont == maxsize)
                        return;
                }
                break;
            case 270:
                for (float i=wIn-1; i>=0; i-=strideW) {
                    for (float j=0; j<=hIn; j+=strideH) {
                        out[cont++] = in[(int)j*wIn + (int)i];
                        if (cont!= 0 && cont%wOut == 0)
                            break;
                    }
                    if (cont == maxsize)
                        return;
                }
                break;
        }
    }



}
