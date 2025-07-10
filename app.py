from flask import Flask, request, render_template, jsonify
import numpy as np
from PIL import Image
import io
import os
import base64
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib.cm as cm

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = None
class_names = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'mel', 'vasc']
class_names_map = {
    'akiec': 'Actinic Keratoses',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'nv': 'Melanocytic Nevi',
    'mel': 'Melanoma',
    'vasc': 'Vascular Lesions'
}
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# CBAM layer
from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Add, Multiply, Concatenate, Activation, Input

class CBAM(tf.keras.layers.Layer):
    def __init__(self, reduction_ratio=8, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channel = input_shape[-1]

        self.shared_dense_one = Dense(channel // self.reduction_ratio, activation='relu')
        self.shared_dense_two = Dense(channel)

        self.conv_spatial = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2])
        max_pool = tf.reduce_max(inputs, axis=[1, 2])

        avg_dense = self.shared_dense_two(self.shared_dense_one(avg_pool))
        max_dense = self.shared_dense_two(self.shared_dense_one(max_pool))

        channel_attention = tf.nn.sigmoid(avg_dense + max_dense)
        channel_attention = tf.reshape(channel_attention, [-1, 1, 1, inputs.shape[-1]])

        x = inputs * channel_attention

        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)

        spatial_attention = self.conv_spatial(concat)
        refined = x * spatial_attention

        return refined

def load_model_safely():
    global model
    try:
        model = load_model("CBAMEFFICIENTNET3_FINAL(7).keras", compile=False, custom_objects={'CBAM': CBAM})
        print("✅ Model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

# ✅ Ensure model loads at startup (works for Gunicorn too)
if not load_model_safely():
    raise RuntimeError("❌ Could not load model at startup")

def preprocess_image(image, target_size=(224, 224)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    img_array = np.array(image).astype(np.float32)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def get_gradcam_overlay(model, img_array, layer_name='top_conv'):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = np.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    heatmap = heatmap.numpy()

    img = img_array[0]
    img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255).astype(np.uint8)

    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = Image.fromarray(np.uint8(jet_heatmap * 255))
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    img = Image.fromarray(img)

    overlay = Image.blend(img.convert("RGBA"), jet_heatmap.convert("RGBA"), alpha=0.5)

    buffer = io.BytesIO()
    overlay.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid or no file'}), 400

    try:
        image = Image.open(file.stream)
        processed_image = preprocess_image(image)
        preds = model.predict(processed_image)

        if preds.shape[-1] > 1:
            preds = tf.nn.softmax(preds).numpy()

        top_idx = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))
        top_preds = np.argsort(preds[0])[-5:][::-1]
        top_predictions = [
            {
                'class_name': class_names_map[class_names[i]],
                'confidence': float(preds[0][i])
            }
            for i in top_preds
        ]

        gradcam_img = get_gradcam_overlay(model, processed_image)

        return jsonify({
            'success': True,
            'predicted_class': class_names_map[class_names[top_idx]],
            'confidence': confidence,
            'top_predictions': top_predictions,
            'gradcam_image': gradcam_img
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
