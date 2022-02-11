import math
from typing import Union, Tuple, List
import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# For static images:
# IMAGE_FILES = []
# with mp_face_detection.FaceDetection(
#     model_selection=1, min_detection_confidence=0.5) as face_detection:
#   for idx, file in enumerate(IMAGE_FILES):
#     image = cv2.imread(file)
#     # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
#     results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#     # Draw face detections of each face.
#     if not results.detections:
#       continue
#     annotated_image = image.copy()
#     for detection in results.detections:
#       print('Nose tip:')
#       print(mp_face_detection.get_key_point(
#           detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
#       mp_drawing.draw_detection(annotated_image, detection)
#     cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

def distance_2_point(point_1, point_2):
  x1, y1 = point_1
  x2, y2 = point_2
  distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
  return distance

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

# For webcam input:
cap = cv2.VideoCapture(1)
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)
    '''
    [295.13, 177.64], // right eye
    [382.32, 175.56], // left eye
    [341.18, 205.03], // nose
    [345.12, 250.61], // mouth
    [252.76, 211.37], // right ear
    [431.20, 204.93] // left ear
    '''

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]
    if results.detections:
      for detection in results.detections:

        mp_drawing.draw_detection(image, detection)
        right_eye_norm = [detection.location_data.relative_keypoints[0].x, detection.location_data.relative_keypoints[0].y]
        left_eye_norm = [detection.location_data.relative_keypoints[1].x, detection.location_data.relative_keypoints[1].y]
        nose_norm = [detection.location_data.relative_keypoints[2].x, detection.location_data.relative_keypoints[2].y]
        mouth_norm = [detection.location_data.relative_keypoints[3].x, detection.location_data.relative_keypoints[3].y]
        right_ear_norm = [detection.location_data.relative_keypoints[4].x, detection.location_data.relative_keypoints[4].y]
        left_ear_norm = [detection.location_data.relative_keypoints[5].x, detection.location_data.relative_keypoints[5].y]

        right_eye = _normalized_to_pixel_coordinates(right_eye_norm[0], right_eye_norm[1], width, height)
        left_eye = _normalized_to_pixel_coordinates(left_eye_norm[0], left_eye_norm[1], width, height)
        nose =_normalized_to_pixel_coordinates(nose_norm[0], nose_norm[1], width, height)
        mouth = _normalized_to_pixel_coordinates(mouth_norm[0], mouth_norm[1], width, height)
        right_ear = _normalized_to_pixel_coordinates(right_ear_norm[0], right_ear_norm[1], width, height)
        left_ear = _normalized_to_pixel_coordinates(left_ear_norm[0], left_ear_norm[1], width, height)

        # cv2.circle(image, right_eye, 2, (255,0,0), 2)
        # cv2.circle(image, left_eye, 2, (255,0,0), 2)
        # cv2.circle(image, nose, 2, (255,0,0), 2)
        # cv2.circle(image, mouth, 2, (255,0,0), 2)
        # cv2.circle(image, right_ear, 2, (255,0,0), 2)
        # cv2.circle(image, left_ear, 2, (255,0,0), 2)

        left_distance = distance_2_point(left_ear, nose)
        right_distance = distance_2_point(right_ear, nose)
        ratio = left_distance/right_distance
        print("left_distance", left_distance)
        print("right_distance", right_distance)
        print("ratio", left_distance/right_distance)
        print("=============")

        if ratio < 0.45:
          print("Left face")
        elif ratio > 2.2:
          print("right face")
        else:
          print('frontal ')

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()