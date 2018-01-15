package org.tensorflow.demo;

import android.graphics.Bitmap;
import android.graphics.RectF;

import java.util.List;

/**
 * Generic interface for interacting with different recognition engines.
 */
public interface Classifier {
  /**
   * An immutable result returned by a Classifier describing what was recognized.
   */
  public class Recognition {
    /**
     * Display name for the recognition.
     */
    private final String title;

    /**
     * A sortable score for how good the recognition is relative to others. Higher should be better.
     */
    private final Float confidence;

    /**
     * Optional location within the source image for the location of the recognized object.
     */
    private final RectF location;

    public Recognition(final String title, final Float confidence, final RectF location) {
      this.title = title;
      this.confidence = confidence;
      this.location = location;
    }

    public String getTitle() {
      return title;
    }

    public Float getConfidence() {
      return confidence;
    }

    public RectF getLocation() {
      return new RectF(location);
    }

    @Override
    public String toString() {
      String resultString = "";
      if (title != null) {
        resultString += title + " ";
      }

      if (confidence != null) {
        resultString += Float.toString(confidence);
      }

      if (location != null) {
        resultString += location + " ";
      }

      return resultString.trim();
    }
  }

  List<Recognition> recognizeImage(Bitmap bitmap);

  void close();
}
