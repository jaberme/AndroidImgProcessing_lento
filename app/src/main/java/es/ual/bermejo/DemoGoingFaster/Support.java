package es.ual.bermejo.DemoGoingFaster;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.hardware.Camera;
import android.view.Surface;

/**
 * Created by corbera on 6/11/15.
 */
@SuppressWarnings("deprecation")
public class Support {

    /* Class to know if a SurfaceView (its Holder) is ready */
    public static class MySurfaceHolderCallback implements android.view.SurfaceHolder.Callback
    {
        public boolean surfaceready=false;

        public void surfaceCreated(android.view.SurfaceHolder holder)
        {
            this.surfaceready=true;
        }

        public void surfaceDestroyed(android.view.SurfaceHolder holder)
        {
            surfaceready=false;
        }

        public void surfaceChanged(android.view.SurfaceHolder holder, int format, int width, int height)
        {
            this.surfaceready=true;
        }
    }

    /* Method to calculate the rotation angle for the camera images */
    /* This angle depends on the display orientation and the camera info */
    public static int setCameraDisplayOrientation(Activity activity, int cameraId) {
        android.hardware.Camera.CameraInfo info = new android.hardware.Camera.CameraInfo();
        android.hardware.Camera.getCameraInfo(cameraId, info);
        int rotation = activity.getWindowManager().getDefaultDisplay().getRotation();
        int degrees = 0;
        switch (rotation) {
            case Surface.ROTATION_0: degrees = 0; break;
            case Surface.ROTATION_90: degrees = 90; break;
            case Surface.ROTATION_180: degrees = 180; break;
            case Surface.ROTATION_270: degrees = 270; break;
        }
        int result;
        if (info.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
            result = (info.orientation + degrees) % 360;
            result = (360 - result) % 360;  // compensate the mirror
        } else {  // back-facing
            result = (info.orientation - degrees + 360) % 360;
        }
        return result;
    }

    /* Method that rotates and scales a RGB image (in[hIn x wIn] --> out[hOut x wOut] rotated "angle" degrees) */
    public static void downscaleAndRotateImage(int[] in, int[] out, int wIn, int hIn, int wOut, int hOut, int angle) {
        if (((angle == 0 || angle == 180) && (wIn < wOut || hIn < hOut)) ||
                (wIn < hOut || hIn < wOut))
            throw new RuntimeException();
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

    /* Method that rotates a Bitmap */
    public static Bitmap RotateBitmap(Bitmap source, float angle)
    {
        Matrix matrix = new Matrix();
        matrix.postRotate(angle);
        return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(), matrix, true);
    }

    public static int[] histogram(byte[] data, int width, int height){
        int[] histogram = new int[256];
        int size = width*height;
        int y;
        int max = 0;
        for(int i=0; i < size; i++) {
            y = data[i  ]&0xff;
            histogram[y] += 1;
            if (histogram[y]>max) max = histogram[y];

        }
        // normaliza los datos
        for(int j=0;j<256;j++){
            histogram[j] = (int)((double)((histogram[j]*100/(double)max)));
        }
        return histogram;
    }

}
