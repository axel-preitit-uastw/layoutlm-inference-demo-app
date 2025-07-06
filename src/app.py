import sys
from typing import List

import streamlit as st
import torch
import pandas as pd
import pytesseract

from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor, BatchEncoding
from PIL import Image, ImageDraw, ImageFont

from src.utils.file_ext import get_file_ext
from src.utils.get_config import get_json_config

LAYOUT_BBOX_SCALE = 1000
INPUT_MAX_LENGTH = 512
INPUT_STRIDE = 128


# Load the model and processor
@st.cache_resource
def load_resources(model_path: str, processor_path: str) -> tuple[
    LayoutLMv3ForTokenClassification, LayoutLMv3Processor]:
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
    processor = LayoutLMv3Processor.from_pretrained(processor_path)

    model.eval()
    return model, processor


# Function to perform OCR and extract bounding boxes
def extract_bounding_boxes(image, lang) -> pd.DataFrame:
    result = pytesseract.image_to_data(image, lang=lang, nice=0, output_type=pytesseract.Output.DATAFRAME)

    # Filter out the results where text is NaN or empty
    result = result[result['text'].notna()]
    result = result[result['text'].str.strip() != '']

    # Rename for clarity
    result.rename(columns={
        'left': 'x',
        'top': 'y',
        'width': 'w',
        'height': 'h'
    }, inplace=True)

    return result


def normalize_bbox(bbox: pd.Series, img_width: int, img_height: int) -> List[int]:
    x = bbox['x']
    y = bbox['y']
    w = bbox['w']
    h = bbox['h']

    x_min = x
    y_min = y
    x_max = x + w
    y_max = y + h

    img_width_scale = LAYOUT_BBOX_SCALE / img_width
    img_height_scale = LAYOUT_BBOX_SCALE / img_height
    box = [
        int(x_min * img_width_scale),
        int(y_min * img_height_scale),
        int(x_max * img_width_scale),
        int(y_max * img_height_scale)
    ]
    return box


def unnormalize_bbox(bbox: List[int], img_width: int, img_height: int) -> dict:
    img_width_scale = img_width / LAYOUT_BBOX_SCALE
    img_height_scale = img_height / LAYOUT_BBOX_SCALE
    x_min = bbox[0] * img_width_scale
    y_min = bbox[1] * img_height_scale
    x_max = bbox[2] * img_width_scale
    y_max = bbox[3] * img_height_scale

    return {
        'x': x_min,
        'y': y_min,
        'w': x_max - x_min,
        'h': y_max - y_min
    }


def infer(model: LayoutLMv3ForTokenClassification, encoding: BatchEncoding, words: List[str], img_width: int,
          img_height: int) -> pd.DataFrame:
    all_label_predictions = []
    all_confidences = []
    all_word_ids = []
    all_words = []
    all_bboxes = []

    # Used to track which word indices have been predicted (to avoid duplicates from overlapping windows)
    used_word_ids = set()

    with torch.no_grad():
        # Loop over each window
        num_windows = len(encoding['input_ids'])
        for i in range(num_windows):
            input_ids = torch.tensor([encoding['input_ids'][i]])
            attention_mask = torch.tensor([encoding['attention_mask'][i]])
            bbox = torch.tensor([encoding['bbox'][i]])

            outputs = model(input_ids, attention_mask=attention_mask, bbox=bbox)

            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predictions = logits.argmax(-1).squeeze().tolist()
            confidences = probs.max(-1).values.squeeze().tolist()

            # Ensure correct types (lists)
            if isinstance(predictions, int):
                predictions = [predictions]

            if isinstance(confidences, float):
                confidences = [confidences]

            # Get token-to-word mapping for this window
            word_ids = encoding.word_ids(batch_index=i)
            bboxes = encoding['bbox'][i]

            for (pred, conf, word_id, bbox) in zip(predictions, confidences, word_ids, bboxes):
                # Skip special tokens and padding
                if word_id is None or word_id in used_word_ids:
                    continue

                used_word_ids.add(word_id)

                all_label_predictions.append(model.config.id2label[pred])
                all_confidences.append(conf)
                all_word_ids.append(word_id)
                all_words.append(words[word_id])
                all_bboxes.append(bbox)

    # Convert bounding boxes to original image scale
    all_bboxes = [unnormalize_bbox(bbox, img_width, img_height) for bbox in all_bboxes]

    # Create a DataFrame to hold the results
    results = pd.DataFrame({
        'word': all_words,
        'label': all_label_predictions,
        'confidence': all_confidences,
        'bbox': all_bboxes
    })

    # Order by bbox coordinates for better visualization
    results['bbox'] = results['bbox'].apply(lambda x: [x['x'], x['y'], x['w'], x['h']])
    # Add x and y columns to sort by position
    results['x'] = results['bbox'].apply(lambda x: x[0])
    results['y'] = results['bbox'].apply(lambda x: x[1])
    # Sort by y first, then by x
    results = results.sort_values(by=['y', 'x']).reset_index(drop=True)
    # Remove temporary x and y columns
    results = results.drop(columns=['x', 'y'])

    return results

def draw_bboxes(image: Image.Image, results_df: pd.DataFrame) -> Image.Image:
    copy = image.copy()
    draw = ImageDraw.Draw(copy)
    font = ImageFont.load_default()

    for idx, row in results_df.iterrows():
        bbox = row['bbox']
        label = row['label']
        word = row['word']

        x, y, w, h = bbox

        # Draw the bounding box
        draw.rectangle([x, y, x + w, y + h], outline="red", width=2)

        # Draw the label text
        text = f"{label}: {word}"
        text_bbox = draw.textbbox((x, y), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        draw.rectangle([x, y - text_height, x + text_width, y], fill="red")
        draw.text((x, y - text_height), text, fill="white", font=font)

    return copy


# Streamlit app
def main(config: dict) -> None:
    model_config = config['model']
    model_path = model_config['path']
    processor_config = config['processor']
    processor_path = processor_config['path']
    ocr_config = config['ocr']
    lang = ocr_config['lang']

    model, processor = load_resources(model_path, processor_path)

    st.title("LayoutLMv3 detectAR App")
    st.write("Upload an image to analyze the invoice content.")

    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Process Document"):
            with st.spinner("Processing..."):
                # Extract bounding boxes using OCR
                ocr_result = extract_bounding_boxes(image, lang)

                words = []
                boxes = []
                img_width, img_height = image.size

                for _, row in ocr_result.iterrows():
                    text = row['text']
                    # Normalize bboxes to 0-1000 scale as LayoutLMv3 expects
                    box = normalize_bbox(row, img_width, img_height)
                    words.append(text)
                    boxes.append(box)

                # Prepare inputs for the model
                encoding = processor(
                    image,
                    words,
                    boxes=boxes,
                    truncation=True,
                    max_length=INPUT_MAX_LENGTH,
                    stride=INPUT_STRIDE,
                    return_overflowing_tokens=True,
                    return_offsets_mapping=True,
                )

                # Perform inference
                results_df = infer(model, encoding, words, img_width, img_height)

                st.write("Inference Results:")
                st.dataframe(results_df)

                csv = results_df.to_csv(index=False, sep=';', encoding='utf-16')
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name='results.csv',
                    mime='text/csv'
                )

                # Draw bounding boxes on the image
                annotated_image = draw_bboxes(image.copy(), results_df)
                st.image(annotated_image, caption="Annotated Image", use_container_width=True)
                image_ext = get_file_ext(image.filename)
                st.download_button(
                    label="Download Annotated Image",
                    data=annotated_image.tobytes(),
                    file_name=f'annotated_image.{image_ext}',
                    mime=f'image/{image_ext}'
                )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python app.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    config = get_json_config(config_path)

    main(config)
