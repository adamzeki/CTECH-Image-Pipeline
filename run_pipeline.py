import json
import gc
import cv2
import torch
import json
from pathlib import Path
from ultralytics import YOLO
from vllm import LLM, SamplingParams
from tqdm import tqdm

# --- Setting config ---
DEFAULT_CONFIG = {
    "YOLO_MODEL_PATH": 'yolo26n.pt',
    "LLM_MODEL_PATH": "google/gemma-3-4b-it",
    "DETECTED_CLASSES": ["person", "car", "bus", "motorcycle", "bicycle"],
    "TARGET_RESOLUTION": [640, 640],
    "ENABLE_PREPROCESSING": True,
    "PROCESSED_DIR_NAME": "processed",
    "INPUT_DIR": "../inputs/input_images",
    "OUTPUT_DIR": "./outputs/pipeline_results",
}

CONFIG_FILE = Path("config.json")

def load_config(config_path: Path):
    config = DEFAULT_CONFIG.copy()
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                external_config = json.load(f)
                config.update(external_config)
                print(f"Załadowano konfigurację z pliku: {config_path}")
        except Exception as e:
            print(f"Błąd podczas wczytywania pliku konfiguracyjnego: {e}. Używam domyślnych.")
    else:
        print("Plik konfiguracyjny nie istnieje. Używam ustawień domyślnych.")
    return config

config=load_config(CONFIG_FILE)

# --- Config ---
YOLO_MODEL_PATH = config["YOLO_MODEL_PATH"]
LLM_MODEL_PATH = config["LLM_MODEL_PATH"]

DETECTED_CLASSES = config["DETECTED_CLASSES"]  
INPUT_DIR = Path(config["INPUT_DIR"])
OUTPUT_DIR = Path(config["OUTPUT_DIR"])

# --- Preprocessing config ---
TARGET_RESOLUTION = tuple(config["TARGET_RESOLUTION"])
ENABLE_PREPROCESSING = config["ENABLE_PREPROCESSING"]
PROCESSED_DIR_NAME =  config["PROCESSED_DIR_NAME"]
PROCESSED_DIR = INPUT_DIR / PROCESSED_DIR_NAME

# Global sampling params for vLLM
sampling_params = SamplingParams(temperature=0.01, max_tokens=1024)


def clear_vram():
    """Forces Python and PyTorch to release all unassigned GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def preprocess_image(image_path: Path, output_dir: Path, target_size=TARGET_RESOLUTION) -> Path:
    """
    Resizes image to target_size while maintaining aspect ratio (letterboxing).
    Saves the result to output_dir and returns the new path.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return image_path

    h, w = img.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    top = (target_size[1] - new_h) // 2
    bottom = target_size[1] - new_h - top
    left = (target_size[0] - new_w) // 2
    right = target_size[0] - new_w - left
    
    padded_img = cv2.copyMakeBorder(
        resized, top, bottom, left, right, 
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    processed_path = output_dir / image_path.name
    cv2.imwrite(str(processed_path), padded_img)
    
    return processed_path

def process_image_with_yolo(image_path, model):
    """Run YOLO and extract objects/coordinates using the provided model."""
    results = model.predict(str(image_path), verbose=False)
    detected_objects = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]
            
            detected_objects.append({
                "label": class_name,
                "bbox": [x1, y1, x2, y2]
            })
            
    return detected_objects

def generate_description(detected_objects, llm_instance):
    """Format data and pass it to the provided vLLM instance."""
    yolo_output_str = json.dumps(detected_objects, indent=2)
    
    system_prompt = """
    You are an AI assistant that analyzes object detection data. 
    You will be provided with a JSON list of detected objects and their bounding box coordinates.
    
    Write a descriptive summary of every object in the scene. 
    You MUST output object descriptions in your response strictly in the following format:
    {
        object: "The detected object",
        location: "The location of the object in the image, based on the bounding box coordinates (e.g., 'top-left', 'center', 'bottom-right')",
    }
    """
    
    full_prompt = f"{system_prompt}\n\nDetected Objects:\n{yolo_output_str}"
    
    try:
        outputs = llm_instance.generate([full_prompt], sampling_params, use_tqdm=False)
        llm_response = outputs[0].outputs[0].text.strip()
        
        return llm_response
        
    except Exception as e:
        print(f"Error during LLM generation: {e}")
        return None

if __name__ == "__main__":
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    image_files = list(INPUT_DIR.glob("*.jpg")) + list(INPUT_DIR.glob("*.png"))
    
    if not image_files:
        print(f"No images found in '{INPUT_DIR}'. Please add some and run again.")
        exit()

    # --- Preprocessing ---
    target_images = []
    
    if ENABLE_PREPROCESSING:
        print("\nPreprocessing images...")
        for img_path in tqdm(image_files):
            processed_path = preprocess_image(img_path, PROCESSED_DIR)
            target_images.append(processed_path)
    else:
        target_images = image_files

    # --- YOLO Detection ---
    print("\n--- YOLO Detection ---")
    print("Loading YOLO model...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    
    # Optional: Filter for specific classes
    # yolo_model.set_classes(DETECTED_CLASSES) 

    all_detected_data = {}

    for img_path in tqdm(target_images):
        print(f"Detecting objects in {img_path.name}...")
        objects = process_image_with_yolo(img_path, yolo_model)
        
        if objects:
            all_detected_data[img_path] = objects
        else:
            print(f"  -> No objects detected in {img_path.name}. Skipping.")

    print("Unloading YOLO to free VRAM...")
    del yolo_model
    clear_vram()

    if not all_detected_data:
        print("No objects detected in any of the images. Pipeline finished.")
        exit()

    # --- LLM Generation ---
    print("\n--- LLM Generation ---")
    print(f"Loading {LLM_MODEL_PATH}...")
    llm = LLM(model=LLM_MODEL_PATH, gpu_memory_utilization=0.75)
    
    print("\nGenerating descriptions...")
    for img_path, objects in tqdm(all_detected_data.items()):
        final_output = generate_description(objects, llm)
        
        if final_output:
            output_file = OUTPUT_DIR / f"{img_path.stem}.txt"
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(final_output)
                

    print(f"\nPipeline execution complete. Results saved to {OUTPUT_DIR}")