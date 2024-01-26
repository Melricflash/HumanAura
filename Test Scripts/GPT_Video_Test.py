import cv2
import numpy as np
import onnxruntime as ort

def load_model(model_path):
    return ort.InferenceSession(model_path)

def preprocess_image(image, target_size):
    # Resize and normalize the input image
    resized_image = cv2.resize(image, (target_size[1], target_size[2]))
    normalized_image = (resized_image.astype(np.float32) / 255.0).astype(np.uint8)
    # Add a batch dimension to the input data
    input_data = np.expand_dims(normalized_image, axis=0)
    return input_data

def postprocess_output(output, width, height, threshold=0.5):
    # Apply threshold to the output to get predicted bounding boxes
    boxes = []
    for detection in output:
        if detection[4] >= threshold:
            box = detection[0:4] * np.array([width, height, width, height])
            boxes.append(box.astype(int))
    return boxes

def draw_boxes(image, boxes):
    for box in boxes:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    return image

def main(video_path, model_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file.")
        return

    model = load_model(model_path)
    target_size = (1, 288, 384, 3)  # Adjust this size based on your model's input size

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        height, width, _ = frame.shape
        input_data = preprocess_image(frame, target_size)

        # Run inference using the ONNX model
        input_name = model.get_inputs()[0].name
        output_name = model.get_outputs()[0].name
        output = model.run([output_name], {input_name: input_data})

        # Post-process the output and draw bounding boxes
        boxes = postprocess_output(output[0], width, height)
        frame_with_boxes = draw_boxes(frame.copy(), boxes)

        cv2.imshow('Video with Bounding Boxes', frame_with_boxes)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_file = "../Videos/Walk1.mpg"
    onnx_model = "../ONNX Models/mobilenet2.onnx"

    main(video_file, onnx_model)

# import onnx
# onnx_model = onnx.load('testmodel.onnx')

# try:
#     onnx.checker.check_model(onnx_model)
# except onnx.checker.ValidationError as e:
#     print("The model is invalid: %s" % e)
# else:
#     print("The model is valid!")
