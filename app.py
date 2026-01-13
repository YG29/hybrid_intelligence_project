import os
from datetime import datetime
import streamlit as st
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import random_split
from torchvision import transforms

from src import dataset
from src import configure

@st.cache_data
def load_predictions(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def get_top5_from_row(row: pd.Series):
    top5 = []
    for k in range(1, 6):
        idx = int(row[f"top{k}_index"])
        name = row.get(f"top{k}_name", str(idx))
        prob = float(row[f"top{k}_prob"])
        top5.append((k, idx, name, prob))
    return top5


def init_session_state(df: pd.DataFrame):
    num_rows = len(df)

    if "row_idx" not in st.session_state:
        st.session_state.row_idx = 0

    if "annotations" not in st.session_state:
        # If we already have an annotations file, load it and skip labeled images
        if os.path.exists(configure.ANNOTATIONS_CSV):
            st.session_state.annotations = pd.read_csv(configure.ANNOTATIONS_CSV)
            labeled_paths = set(st.session_state.annotations["image_path"].tolist())

            st.session_state.row_idx = 0
            while st.session_state.row_idx < num_rows:
                path = df.iloc[st.session_state.row_idx]["image_path"]
                if path in labeled_paths:
                    st.session_state.row_idx += 1
                else:
                    break
        else:
            st.session_state.annotations = pd.DataFrame()

    if st.session_state.row_idx >= num_rows:
        st.session_state.row_idx = num_rows - 1


def save_annotations_to_disk():
    st.session_state.annotations.to_csv(configure.ANNOTATIONS_CSV, index=False)


def record_annotation(row: pd.Series,
                      image_path: str,
                      chosen_label_idx: int,
                      chosen_label_name: str,
                      chosen_label_prob: float,
                      accepted_top1: bool):
    true_idx = int(row["true_label_index"])
    true_name = row.get("true_label_name", str(true_idx))
    top1_idx = int(row["top1_index"])
    top1_name = row.get("top1_name", str(top1_idx))
    top1_prob = float(row["top1_prob"])

    annotation = {
        "image_path": image_path,
        "true_label_index": true_idx,
        "true_label_name": true_name,
        "top1_index": top1_idx,
        "top1_name": top1_name,
        "top1_prob": top1_prob,
        "human_label_index": int(chosen_label_idx),
        "human_label_name": chosen_label_name,
        "human_label_prob": chosen_label_prob,
        "accepted_top1": bool(accepted_top1),
        "model_top1_correct": bool(top1_idx == true_idx),
        "timestamp": datetime.utcnow().isoformat(),
    }

    st.session_state.annotations = pd.concat(
        [st.session_state.annotations, pd.DataFrame([annotation])],
        ignore_index=True,
    )
    save_annotations_to_disk()

## Coded with help of ChatGPT
# -------------------
# STREAMLIT UI
# -------------------

st.set_page_config(page_title="GTSRB Human Audit", layout="wide")

st.title("Human Audit of First-Round Predictions")
st.write(
    "For each image, you see the model's top-5 predictions. "
    "Use **Tick** to accept the model's top-1 prediction, or **Cross** to choose a different one."
)

# Load predictions CSV
if not os.path.exists(configure.PREDICTIONS_CSV):
    st.error(f"Predictions CSV not found: {configure.PREDICTIONS_CSV}")
    st.stop()

df = load_predictions(configure.PREDICTIONS_CSV)
num_rows = len(df)

if num_rows == 0:
    st.warning("No rows found in predictions CSV.")
    st.stop()

# Build full list of labels (id -> name) from the CSV for the dropdown
gtsrb_names = dataset.gtsrb_class_names
all_label_options = [
    f"{idx}: {name}"
    for idx, name in sorted(gtsrb_names.items(), key=lambda x: x[0])
]
full_label_placeholder = "(none / not needed)"

# Init session state (row index, existing annotations)
init_session_state(df)

if st.session_state.row_idx >= num_rows:
    st.success("All images have been annotated 🎉")
    st.stop()

row_idx = st.session_state.row_idx
row = df.iloc[row_idx]

image_path = row["image_path"]

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader(f"Image {row_idx + 1} / {num_rows}")
    try:
        img = load_image(image_path)
        st.image(img, caption=image_path, width=350) # change the width to make the image look better
    except Exception as e:
        st.error(f"Could not load image path: {image_path}\nError: {e}")

    # st.markdown(
    #     f"**True label** (for analysis): "
    #     f"`{row.get('true_label_name', row['true_label_index'])}`"
    # )

with col_right:
    st.subheader("Model's Top-5 Predictions")

    top5 = get_top5_from_row(row)

    # Show as radio list for Cross case
    options = [f"{name} (p={prob:.5f})" for (_, _, name, prob) in top5]
    default_index = 0  # by default highlight top1

    selected_option = st.radio(
        "If the model is wrong, choose the correct class:",
        options=options,
        index=default_index,
    )

    # map back selection
    for (k, idx, name, prob), opt in zip(top5, options):
        if opt == selected_option:
            selected_k, selected_idx, selected_name, selected_prob = k, idx, name, prob
            break

    # NEW: full-label dropdown as backup
    st.markdown("If the correct class is **not** in the top-5, choose it here:")
    full_label_choice = st.selectbox(
        "Full label list (fallback):",
        options=[full_label_placeholder] + all_label_options,
        index=0,
    )

    # Parse full-label choice if used
    full_label_idx = None
    full_label_name = None
    if full_label_choice != full_label_placeholder:
        # format: "idx: name"
        idx_str, name_str = full_label_choice.split(":", 1)
        full_label_idx = int(idx_str.strip())
        full_label_name = name_str.strip()

    # Visual highlight of final choice (current selection)
    st.markdown(
        f"**Current chosen class:** :blue[`{selected_name}`] "
        f"(prob = {selected_prob:.5f})"
    )

    st.markdown("### Actions")

    col_tick, col_cross = st.columns(2)

    with col_tick:
        if st.button("✅ Accept model's top-1"):
            # Final choice: top-1 prediction
            k1, idx1, name1, prob1 = top5[0]
            record_annotation(
                row=row,
                image_path=image_path,
                chosen_label_idx=idx1,
                chosen_label_name=name1,
                chosen_label_prob=prob1,
                accepted_top1=True,
            )
            st.session_state.row_idx += 1
            st.rerun()

    with col_cross:
        if st.button("❌ Use selected class"):
            # Decide whether to use the full-label dropdown or the top-5 radio
            if full_label_idx is not None:
                final_idx = full_label_idx
                final_name = full_label_name
                # No prob info available beyond top-5; use 0.0 as placeholder
                final_prob = 0.0
            else:
                final_idx = selected_idx
                final_name = selected_name
                final_prob = selected_prob

            record_annotation(
                row=row,
                image_path=image_path,
                chosen_label_idx=final_idx,
                chosen_label_name=final_name,
                chosen_label_prob=final_prob,
                accepted_top1=False,
            )
            st.session_state.row_idx += 1
            st.rerun()

st.markdown("---")
st.caption(
    f"Annotations are saved incrementally to `{configure.ANNOTATIONS_CSV}`. "
    "You can stop and continue later — progress is tracked by image path."
)

