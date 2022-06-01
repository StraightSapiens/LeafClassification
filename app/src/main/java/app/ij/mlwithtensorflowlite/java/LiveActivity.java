package app.ij.mlwithtensorflowlite.java;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import com.google.common.util.concurrent.ListenableFuture;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.ExecutionException;

import app.ij.mlwithtensorflowlite.R;
import app.ij.mlwithtensorflowlite.ml.Model;

public class LiveActivity extends AppCompatActivity implements ImageAnalysis.Analyzer {

    Button back;
    TextView result;
    Preview preview;
    PreviewView previewView;
    ImageAnalysis imageAnalyzer;
    Model model;
    //ExecutorService cameraExecutor = Executors.newSingleThreadExecutor();
    ListenableFuture<ProcessCameraProvider> cameraProviderF;
    int imageSize = 224;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_live);

        back = findViewById(R.id.backbutton);
        previewView = findViewById(R.id.previewView);
        result = findViewById(R.id.result);
        back.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                finish();
            }
        });

        try {
            model = Model.newInstance(getApplicationContext());
        } catch (IOException e) {
            e.printStackTrace();
        }
        startCamera();

    }
    public void startCamera() {
        cameraProviderF = ProcessCameraProvider.getInstance(this);
        cameraProviderF.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderF.get();
                bindPreview(cameraProvider);
            } catch (ExecutionException | InterruptedException e) {

            }
        }, ContextCompat.getMainExecutor(this));
    }
    public void bindPreview(@NonNull ProcessCameraProvider cameraProvider) {
        cameraProvider.unbindAll();
        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();
        imageAnalyzer = new ImageAnalysis.Builder()
                .setTargetResolution(new Size(imageSize, imageSize))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();
        preview = new Preview.Builder()
                .build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        Camera camera = cameraProvider.bindToLifecycle((LifecycleOwner)this, cameraSelector, preview, imageAnalyzer);
    }
    @Override
    public void analyze(@NonNull ImageProxy imagep) {
            TensorImage tfimage = TensorImage.fromBitmap(toBitmap(imagep));
            result.setText(tfimage.getHeight());
            TensorBuffer tbuffer = tfimage.getTensorBuffer();
            Model.Outputs outputs = model.process(tbuffer);
            TensorBuffer outputbuffer = outputs.getOutputFeature0AsTensorBuffer();
            float[] confidences = outputbuffer.getFloatArray();
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            String[] classes = {"Apple (Scab)", "Apple (Black rot)", "Apple (Cedar apple rust)", "Apple (Healthy)", "Blueberry (Healthy)", "Cherry (Healthy)", "Cherry (Powdery mildew)", "Corn (Cercospora leaf spot: Gray leaf spot)", "Corn (Common rust)", "Corn (Healthy)", "Corn (Northern Leaf blight)", "Grape (Black rot)", "Grape (Esca: BlackMeasles)", "Grape (Healthy)", "Grape (Leaf blight: Isariopsis Leaf spot)", "Orange (Haunglongbing: Citrus greening)", "Peach (Bacterial spot)", "Peach (Healthy)", "Pepper (Bacterial spot)", "Pepper (Healthy)", "Potato (Early blight)", "Potato (Healthy)", "Potato (Late blight)", "Raspberry (Healthy)", "Soy bean (Healthy)", "Squash (Powdery mildew)", "Strawberry (Healthy)", "Strawberry (Leaf scorch)", "Tomato (Bacterial spot)", "Tomato (Early blight)", "Tomato (Healthy)", "Tomato (Late blight)", "Tomato (Leaf Mold)", "Tomato (Septoria leaf spot)", "Tomato (Spider mites: Two-spotted spider mite)", "Tomato (Target Spot)", "Tomato (Mosaic virus)", "Tomato (Yellow Leaf Curl Virus)"};
            if (maxConfidence > 0.92) {
                result.setText(classes[maxPos]);
            }
            else{
                result.setText("Not classifiable");
            }
    }
    public Bitmap toBitmap(ImageProxy image){
        ImageProxy.PlaneProxy[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];
        //U and V are swapped
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 75, out);

        byte[] imageBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }
}