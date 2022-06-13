/*
 * Created by ishaanjav
 * github.com/ishaanjav
 */

package app.ij.mlwithtensorflowlite.java;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;


import com.google.android.material.snackbar.Snackbar;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import app.ij.mlwithtensorflowlite.R;
import app.ij.mlwithtensorflowlite.ml.Model;


public class MainActivity extends AppCompatActivity {

    Button camera, gallery, live;
    ImageView imageView;
    TextView result, confidency;
    int imageSize = 224;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        camera = findViewById(R.id.button);
        gallery = findViewById(R.id.button2);
        live = findViewById(R.id.livebutton);

        result = findViewById(R.id.result);
        confidency = findViewById(R.id.confidency);
        imageView = findViewById(R.id.imageView);

        camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 3);
                } else {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });
        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent cameraIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(cameraIntent, 1);
            }
        });
        live.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    openLive();
                } else {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });
    }
    public void openLive(){
        Intent intent = new Intent(this, LiveActivity.class);
        startActivity(intent);
    }

    public void classifyImage(Bitmap image){
        try {
            Model model = Model.newInstance(getApplicationContext());
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;
            for(int i = 0; i < imageSize; i ++){
                for(int j = 0; j < imageSize; j++){
                    int val = intValues[pixel++];
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            String percentage = String.format("%."+2+"f",maxConfidence*100)+"%";
            String[] classes = {"Apple (Scab)", "Apple (Black rot)", "Apple (Cedar apple rust)", "Apple (Healthy)", "Blueberry (Healthy)", "Cherry (Healthy)", "Cherry (Powdery mildew)", "Corn (Cercospora leaf spot: Gray leaf spot)", "Corn (Common rust)", "Corn (Healthy)", "Corn (Northern Leaf blight)", "Grape (Black rot)", "Grape (Esca: BlackMeasles)", "Grape (Healthy)", "Grape (Leaf blight: Isariopsis Leaf spot)", "Orange (Haunglongbing: Citrus greening)", "Peach (Bacterial spot)", "Peach (Healthy)", "Pepper (Bacterial spot)", "Pepper (Healthy)", "Potato (Early blight)", "Potato (Healthy)", "Potato (Late blight)", "Raspberry (Healthy)", "Soy bean (Healthy)", "Squash (Powdery mildew)", "Strawberry (Healthy)", "Strawberry (Leaf scorch)", "Tomato (Bacterial spot)", "Tomato (Early blight)", "Tomato (Healthy)", "Tomato (Late blight)", "Tomato (Leaf Mold)", "Tomato (Septoria leaf spot)", "Tomato (Spider mites: Two-spotted spider mite)", "Tomato (Target Spot)", "Tomato (Mosaic virus)", "Tomato (Yellow Leaf Curl Virus)"};
            if (maxConfidence > 0.92) {
                result.setText(classes[maxPos]);
                confidency.setText(percentage);
            }
            else{
                result.setText("Not classifiable");
            }
            model.close();
        } catch (IOException e) {
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if(resultCode == RESULT_OK){
            if(requestCode == 3){
                Bitmap image = (Bitmap) data.getExtras().get("data");
                int dimension = Math.min(image.getWidth(), image.getHeight());
                image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                classifyImage(image);
            }else{
                Uri dat = data.getData();
                String extension = getContentResolver().getType(dat);
                if (extension.contains("image")){
                    Bitmap image = null;
                    try {
                        image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
                    } catch (IOException e) {

                    }
                    imageView.setImageBitmap(image);

                    image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                    classifyImage(image);
                }
                else{
                    Snackbar.make(findViewById(R.id.mainlayout), "Not usable file format", Snackbar.LENGTH_SHORT)
                            .show();
                }
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}