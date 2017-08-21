package com.example.eoghk.mlexample;

import android.content.res.AssetManager;
import android.os.Bundle;
import android.support.wearable.activity.WearableActivity;
import android.view.View;
import android.widget.TextView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

public class MainActivity extends WearableActivity {


    private static final String MODEL_FILE = "file:///android_asset/graph.pb";
    private static final String TEST_FILE = "file:///android_asset/shear_detection_daehwakim_test.csv";
    private static final String INPUT_NODE = "input_node";
    private static final String OUTPUT_NODE = "hypothesis";

    private static final int[] INPUT_SIZE = {1, 49};
    private TensorFlowInferenceInterface inferenceInterface;

    AssetManager assetMgr ;
    InputStream inputData ;
    BufferedReader bufferedReader;
    String string=null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        setAmbientEnabled();
        initMyModel();

        View button = findViewById(R.id.button);
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                runMyModel();
            }
        });
        try {
            assetMgr = getAssets();
            inputData = assetMgr.open("input_file");
            bufferedReader= new BufferedReader(new InputStreamReader(inputData,"EUC_KR"));
            while(true){
                string= bufferedReader.readLine();
                if(string != null){
                    runMyModel();
                }else{
                    break;
                }
            }

        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

    }

    private void initMyModel() {
        inferenceInterface = new TensorFlowInferenceInterface();
        inferenceInterface.initializeTensorFlow(getAssets(), MODEL_FILE);
    }

    private void runMyModel() {
        float[] inputs = new float[49];
        String []data = string.split(",");
        for (int i=0;i<49;i++)
            inputs[i]=Float.parseFloat(data[i]);

        MinMaxScaler(inputs);

        inferenceInterface.fillNodeFloat(INPUT_NODE, INPUT_SIZE, inputs);
        inferenceInterface.runInference(new String[] {OUTPUT_NODE});

        float[] res = {0};
        inferenceInterface.readNodeFloat(OUTPUT_NODE, res);

        String msg = "input: ";
        for (int i=0;i<49;i++)
            msg+=inputs[i]+", ";
        msg += "\nResult: " + (res[0] > 0.5) + ", " + res[0];
        System.out.println( res[0]+", "+(res[0] > 0.5)+", "+data[49]);
        TextView tv = (TextView) findViewById(R.id.text_view);
        tv.setText(msg);
    }
    float[] min = {-3,   -3,   -2,   -2,    0,    2,    3,   -3,   -2,   -2,   -1,   16,
                    105,  112,   -2,   -1,   -1,    0,    5,  29,   29,   -2,   -2,   -2,
                    -1,    0,    0,    0,   -3,   -3,   -4,   -2,   -2,   -2,   -2,   -2,
                    -2,   -3,   -4,   -2,   -3,   -3,   -5,   -5,   -5,   -5,   -5,   -6,
                    -5},
            max={1,    0,    1,    8,   79,  185,  160,    1,    0,    1,   16,  167,
                    251,  254,   1,    0,    2,   12,  132,  237,  236,    0,    1,    1,
                    4,   24,  57,   56,    1,    0,    0,    0,    2,    4,    4,    1,
                    1,    1,    2,    2,   2,    3,    2,    1,    2,    2,    2,    2,
                    3};
    private void MinMaxScaler(float[] inputs){
        for(int i=0;i<inputs.length;i++){
            if (min[i]>inputs[i])
                min[i]=inputs[i];
            if (max[i]<inputs[i])
                max[i]=inputs[i];
            float numerator = inputs[i] - min[i];
            float denominator = max[i] - min[i];
            double value=numerator / (denominator + 0.00000007);
            inputs[i]= ((float) value);
        }
    }
}
