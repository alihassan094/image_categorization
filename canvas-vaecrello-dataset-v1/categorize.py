import tensorflow as tf
from typing import Any, Dict

def extract_feature(feature: tf.train.Feature) -> Any:
    """Convert tf.train.Feature"""
    if feature.HasField("int64_list"):
        value = [x for x in feature.int64_list.value]
    elif feature.HasField("bytes_list"):
        value = [x for x in feature.bytes_list.value]
    elif feature.HasField("float_list"):
        value = [x for x in feature.float_list.value]
    else:
        raise ValueError("feature is empty")
    return value[0] if len(value) == 1 else value


def extract_sequence_example(serialized: bytes) -> Dict[str, Any]:
    """Parse and convert tf.train.SequenceExample"""
    example = tf.train.SequenceExample()
    example.ParseFromString(serialized)
    output = {}
    for key, feature in example.context.feature.items():
        output[key] = extract_feature(feature)
    for key, feature_list in example.feature_lists.feature_list.items():
        output[key] = [
            extract_feature(feature) for feature in feature_list.feature]
    return output


dataset = tf.data.Dataset.list_files("canvas-vaecrello-dataset-v1/train-*.tfrecord")
dataset = tf.data.TFRecordDataset(dataset)
for example in dataset.take(1).as_numpy_iterator():
    example = extract_sequence_example(example)
    break
print(example)
import io
import numpy as np
import skia

def render(example: Dict[str, tf.Tensor], max_size: float=512.) -> bytes:
    """Render parsed sequence example onto an image and return as PNG bytes."""
    canvas_width = example["canvas_width"]
    canvas_height = example["canvas_height"]

    scale = min(1.0, max_size / canvas_width, max_size / canvas_height)

    surface = skia.Surface(int(scale * canvas_width), int(scale * canvas_height))
    with surface as canvas:
        canvas.scale(scale, scale)
        for index in range(example["length"]):
            with io.BytesIO(example["image_bytes"][index]) as f:
                image = skia.Image.open(f)
            left = example["left"][index] * canvas_width
            top = np.array(example["top"][index]) * canvas_height
            width = example["width"][index] * canvas_width
            height = example["height"][index] * canvas_height
            rect = skia.Rect.MakeXYWH(left, top, width, height)

    image = surface.makeImageSnapshot()
    with io.BytesIO() as f:
        print("Hello inside ",image)
        image.save("./abc.png", f)
        return f.getvalue()
    
render(example)