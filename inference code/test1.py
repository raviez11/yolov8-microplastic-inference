import argparse
import os
from ultralytics import YOLO

def run_inference(model_path, source, dest):
    # Load model
    model = YOLO(model_path)

    # Ensure destination folder exists
    os.makedirs(dest, exist_ok=True)

    # Run inference
    results = model.predict(
        source=source,
        save=True,
        project=dest,
        name="predictions",  # results will be saved in dest/predictions/
        exist_ok=True        # overwrite if already exists
    )

    print(f"[INFO] Inference complete. Results saved in: {os.path.join(dest, 'predictions')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Microplastic Detection")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to image or folder of images")
    parser.add_argument("--dest", type=str, required=True,
                        help="Destination folder to save results")

    args = parser.parse_args()

    # 🔹 Set your trained model path here
    model_path = r"C:\Users\ASUS\Downloads\runs\content\runs\detect\train\weights\best.pt"
    run_inference(model_path, args.source, args.dest)


# python C:\Users\ASUS\Downloads\runs\content\runs\inference_code\test1.py --source C:\Users\ASUS\Downloads\runs\content\runs\inference_code\image\a--82-_jpg.rf.19555a247947eee0814df82cd1f007ae.jpg --dest C:\Users\ASUS\Downloads\runs\content\runs\inference_code\dest 