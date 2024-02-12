import tensorflow as tf
from PIL import Image
import io
import object_detection.utils.label_map_util as label_map_util
import numpy as np
import os

# def clip(value, min_value, max_value):
#     """Clip the value to be within the specified range."""
#     return max(min(value, max_value), min_value)

def create_tf_example(image_path, annotations, label_map_path):
    # Read the label map
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Note this is a hacky method but this I append the current working directory to the image path name to avoid name collisions as these are separate folder
    # Should hopefully allow evaluation with multiple files...
    currentDir = os.getcwd()
    image_path = os.path.join(currentDir, image_path)

    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_image = fid.read()

    encoded_image_io = io.BytesIO(encoded_image)
    image = Image.open(encoded_image_io)
    width, height = image.size

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes = []
    class_texts = []

    for annotation in annotations:

        # xmin = clip(annotation[0] / width, 0, 1)
        # xmax = clip(annotation[1] / width, 0, 1)
        # ymin = clip(annotation[2] / height, 0, 1)
        # ymax = clip(annotation[3] / height, 0, 1)

        xmin = (annotation[0]/width)
        if xmin < 0.0:
            xmin = 0.0
        elif xmin > 1.0:
            xmin = 1.0
        
        xmax = (annotation[1]/width)
        if xmax < 0.0:
            xmax = 0.0
        elif xmax > 1.0:
            xmax = 1.0

        ymin = (annotation[2]/height)
        if ymin < 0.0:
            ymin = 0.0
        elif ymin > 1.0:
            ymin = 1.0

        ymax = (annotation[3]/height)
        if ymax < 0.0:
            ymax = 0.0
        elif ymax > 1.0:
            ymax = 1.0


        xmin_new = min(xmin, xmax)
        xmax_new = max(xmin, xmax)
        ymin_new = min(ymin, ymax)
        ymax_new = max(ymin, ymax)


        # Check if variables are outside the range [0.0, 1.0]
        if xmin_new < 0.0 or xmin_new > 1.0:
            print("Warning: xmin is outside the range [0.0, 1.0]")

        if xmax_new < 0.0 or xmax_new > 1.0:
            print("Warning: xmax is outside the range [0.0, 1.0]")

        if ymin_new < 0.0 or ymin_new > 1.0:
            print("Warning: ymin is outside the range [0.0, 1.0]")

        if ymax_new < 0.0 or ymax_new > 1.0:
            print("Warning: ymax is outside the range [0.0, 1.0]")

        # xmins.append(xmin)
        # xmaxs.append(xmax)
        # ymins.append(ymin)
        # ymaxs.append(ymax)

        xmins.append(xmin_new)
        xmaxs.append(xmax_new)
        ymins.append(ymin_new)
        ymaxs.append(ymax_new)

        # Use class ID from the label map file
        class_id = 1  # Default to class ID 1 if not specified
        if len(annotation) > 4:
            class_id = int(annotation[4])

        classes.append(class_id)

        # Map class ID to class name using the label map
        class_name = category_index[class_id]['name']
        class_texts.append(class_name.encode('utf-8'))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_path.encode('utf-8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_path.encode('utf-8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'jpeg'])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=class_texts)),
    }))

    return tf_example

def parse_annotation_line(line):
    parts = line.strip().split(' ')
    if parts[1] == 'nothing':
        return parts[0], []
    else:
        return parts[0], [float(parts[i]) for i in range(1, 5)] + ([] if len(parts) == 5 else [int(parts[5])])
    
def main():
    input_file = 'br2gtAmended.txt'
    output_path = 'browse2New.tfrecord'
    label_map_path = 'label_map.pbtxt'

    with open(input_file, 'r') as file:
        lines = file.readlines()

    tf_record_writer = tf.io.TFRecordWriter(output_path)

    image_annotations = {}
    for line in lines:
        image_path, annotations = parse_annotation_line(line)
        if image_path not in image_annotations:
            image_annotations[image_path] = []
        if annotations:
            image_annotations[image_path].append(annotations)

    for image_path, annotations_list in image_annotations.items():
        tf_example = create_tf_example(image_path, annotations_list, label_map_path)
        tf_record_writer.write(tf_example.SerializeToString())

    tf_record_writer.close()

if __name__ == '__main__':
    main()