Bundled dlib-related runtime assets live here.

Current contents:

- `shape_predictor_5_face_landmarks.dat`: the 5-point landmark predictor used
  by the active EG3D RLHF loss path when facial landmark extraction is enabled.

The active runtime resolves this file repo-relatively, and callers can still
override it with `DLIB_LANDMARK_MODEL` if needed.
