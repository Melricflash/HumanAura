import tensorflow as tf 

# raw_dataset = tf.data.TFRecordDataset("br1gt.tfrecord")

# for raw_record in raw_dataset.take(1):
#     example = tf.train.Example()
#     example.ParseFromString(raw_record.numpy())
#     print(example)

#print(sum(1 for _ in tf.data.TFRecordDataset("br1gt.tfrecord"))) # prints 1043 images which is correct

import matplotlib.pyplot as plt

# Read the and parse tfrecord
def parse_tfrecord_fn(example):
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)

    # Decodes the image from the tfrecord
    image = tf.image.decode_jpeg(example['image/encoded'], channels=3)

    # Convert sparse tensors to dense tensors to retrieve our data inside tfrecord.
    xmin = tf.sparse.to_dense(example['image/object/bbox/xmin'])
    ymin = tf.sparse.to_dense(example['image/object/bbox/ymin'])
    xmax = tf.sparse.to_dense(example['image/object/bbox/xmax'])
    ymax = tf.sparse.to_dense(example['image/object/bbox/ymax'])
    label = tf.sparse.to_dense(example['image/object/class/label'])

    return image, xmin, ymin, xmax, ymax, label

# use parsed information to decode the bounding boxes back onto an image
def display_image_from_tfrecord(tfrecord_path):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = dataset.map(parse_tfrecord_fn)

    for image, xmin, ymin, xmax, ymax in parsed_dataset:
        plt.imshow(image.numpy())
        plt.axis('off')

        # Draw bounding boxes if available, we get get our coordinates back and convert them to numpy values
        for i in range(tf.size(xmin)):
            xmin_i, ymin_i, xmax_i, ymax_i = xmin[i].numpy(), ymin[i].numpy(), xmax[i].numpy(), ymax[i].numpy()

            # Calculate box coordinates in pixel values by taking the normalised value and multiplying by the height or width.
            xmin_i = int(xmin_i * image.shape[1])
            xmax_i = int(xmax_i * image.shape[1])
            ymin_i = int(ymin_i * image.shape[0])
            ymax_i = int(ymax_i * image.shape[0])

            # Plot using matplotlib
            plt.plot([xmin_i, xmax_i, xmax_i, xmin_i, xmin_i],
                     [ymin_i, ymin_i, ymax_i, ymax_i, ymin_i], color='red')

        plt.show()


# Path to TFRecord file
tfrecord_path = 'Fixed Shops Final/browse_whilewaiting1Fix.tfrecord'
#tfrecord_path = 'Final Shop Dataset/browse1New.tfrecord'
display_image_from_tfrecord(tfrecord_path)



