package lt.vu.mobapps;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Trace;
import android.util.Log;

import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

/**
 * JNI wrapper class for the Tensorflow native code.
 */

class TensorFlowClassifier implements Classifier {
  private static final String TAG = "TensorflowClassifier";

  // Class labels for PASCAL VOC, used because the current model (YOLO v1)
  // has been trained on this detection task.
  private String[] class_labels =  {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
          "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
          "pottedplant", "sheep", "sofa", "train","tvmonitor"};

  private int target_class_id = 14; // person

  // jni native methods.
  public native int initializeTensorFlow(
      AssetManager assetManager,
      String model,
      String labels,
      int numClasses,
      int inputSize,
      int imageMean,
      float imageStd,
      String inputName,
      String outputName);

  private native String classifyImageBmp(Bitmap bitmap);

  private native String classifyImageRgb(int[] output, int width, int height);

  static {
    System.loadLibrary("tensorflow_demo");
  }

  @Override
  public List<Recognition> recognizeImage(final Bitmap bitmap) {
    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("Recognize");
    final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
    ArrayList<Float> temp_array = new ArrayList<>();
    float[][] target_class_probs = new float[7][7];
    float[][][] scales = new float[7][7][2];
    float[][][][] boxes = new float[7][7][2][4];

    final int w_bitmap = bitmap.getWidth();
    final int h_bitmap = bitmap.getHeight();

    final float threshold = 0;

    Log.i(TAG, "Parsing");
    final String result = classifyImageBmp(bitmap);
    final StringTokenizer st = new StringTokenizer(result);
    assert (!st.hasMoreTokens());

    // Get all of the output and put it into an linear array. (1470)
    for (int i = 0; i < 1470; i++) {
      final float token = Float.parseFloat(st.nextToken());
      temp_array.add(token);

    }

    // Convert the linear array to the output tensor (7x7x30)
    int counter = 0;
    // Corresponds to class probs (7x7x20)
    for (int i = 0; i < 7; i++) {
      for (int j = 0; j < 7; j++) {
        target_class_probs[i][j] = temp_array.get(counter + target_class_id);
        counter += 20;
//        for (int k = 0; k < 20; k++) {
//          class_probs[i][j][k] = temp_array.get(counter);
//          counter++;
//        }
      }
    }
    // Corresponds to scales (7x7x2)
    for (int i = 0; i < 7; i++) {
      for (int j = 0; j < 7; j++) {
        for (int k = 0; k < 2; k++) {
          scales[i][j][k] = temp_array.get(counter + target_class_id);
          counter++;
        }
      }
    }
    // Corresponds to boxes (7x7x2x4)
    for (int i = 0; i < 7; i++) {
      for (int j = 0; j < 7; j++) {
        for (int k = 0; k < 2; k++) {
          for (int l = 0; l < 4; l++) {
            boxes[i][j][k][l] = temp_array.get(counter);
            counter++;
          }
        }
      }
    }

    // Add offset.
    for (int i = 0; i < 7; i++) {
      for (int j = 0; j < 7; j++) {
        for (int k = 0; k < 2; k++) {
          boxes[i][j][k][0] += j;
          boxes[i][j][k][0] = boxes[i][j][k][0] * w_bitmap / 7;
        }
      }
    }
    for (int i = 0; i < 7; i++) {
      for (int j = 0; j < 7; j++) {
        for (int k = 0; k < 2; k++) {
          boxes[i][j][k][1] += i;
          boxes[i][j][k][1] = boxes[i][j][k][1] * h_bitmap / 7;
        }
      }
    }
    for (int i = 0; i < 7; i++) {
      for (int j = 0; j < 7; j++) {
        for (int k = 0; k < 2; k++) {
          boxes[i][j][k][2] = boxes[i][j][k][2] * boxes[i][j][k][2] * w_bitmap;
          boxes[i][j][k][3] = boxes[i][j][k][3] * boxes[i][j][k][3] * h_bitmap;
        }
      }
    }

    float[][][] probs = new float[7][7][2];
    // Combine conditional class probabilities and objectness probability.
    for (int i = 0; i < 2; i++) {
      for (int l = 0; l < 7; l++) {
        for (int m = 0; m < 7; m++) {
          probs[l][m][i] = target_class_probs[l][m] * scales[l][m][i];
        }
      }
    }

    // I will try to output the best bounding box and class.
    // First get best probability
    /*
    float highest_prob = 0;
    int hp_i = 0;
    int hp_j = 0;
    int hp_l = 0;
    int hp_m = 0;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 20; j++) {
        for (int l = 0; l < 7; l++) {
          for (int m = 0; m < 7; m++) {
            if (probs[l][m][i][j] >= highest_prob) {
              highest_prob = probs[l][m][i][j];
              hp_i = i;
              hp_j = j;
              hp_l = l;
              hp_m = m;
            }
          }
        }
      }
    }
    */
    int hp_i = 0;
    int hp_j = 0;
    int hp_l = 0;
    int hp_m = 0;

    float bounding_x = 0;
    float bounding_y = 0;
    float box_width = 0;
    float box_height = 0;

    int predicted_class = 0;
    String prediction_string = "";
    RectF boundingBox = null;

    for (int i = 0; i < 2; i++) {
      for (int l = 0; l < 7; l++) {
        for (int m = 0; m < 7; m++) {
          if (probs[l][m][i] >= 0.2) {
            hp_i = i;
            hp_j = target_class_id;
            hp_l = l;
            hp_m = m;

            // Get x, y, width, height. These will be processed and drawn in BoundingBoxView.
            bounding_x = boxes[hp_l][hp_m][hp_i][0];
            bounding_y = boxes[hp_l][hp_m][hp_i][1];
            box_width = boxes[hp_l][hp_m][hp_i][2] / 2;
            box_height = boxes[hp_l][hp_m][hp_i][3] / 2;

            // Now get the class number.
            predicted_class = hp_j;

            // Now log this prediction.
            prediction_string = Integer.toString(predicted_class) + " | x1: " + Float.toString(bounding_x) +
                    " y1: " + Float.toString(bounding_y) + " width: " + Float.toString(box_width) +
                    " height: " + Float.toString(box_height);
            Log.i("Java prediction --- ", prediction_string);

            // Add recognition to recognition list.
            boundingBox = new RectF(bounding_x, bounding_y, box_width, box_height);
//            recognitions.add(new Recognition(class_labels[predicted_class], probs[l][m][i], boundingBox));
            recognitions.add(new Recognition(class_labels[predicted_class], probs[l][m][i], boundingBox));
          }
        }
      }
    }

    Trace.endSection();
    return recognitions;
  }

  @Override
  public void close() {}

}
