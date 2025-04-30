# model.py

import torch
import numpy as np
import os
import cv2
from PIL import Image
import supervision as sv
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForCausalLM

class GroundedSAM2Florence2:
    def __init__(self):
        print("🔥 [Model] __init__ started", flush=True)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        print(f"📸 [Model] Device set to {self.device}", flush=True)

        try:
            self.florence2_model_id = "microsoft/Florence-2-large"
            print("🔵 [Model] Loading Florence-2 model...", flush=True)
            self.florence2_model = AutoModelForCausalLM.from_pretrained(
                self.florence2_model_id,
                trust_remote_code=True,
                torch_dtype='auto'
            ).eval().to(self.device)
            print("✅ [Model] Florence-2 model loaded", flush=True)

            self.florence2_processor = AutoProcessor.from_pretrained(
                self.florence2_model_id,
                trust_remote_code=True
            )
            print("✅ [Model] Florence-2 processor loaded", flush=True)
        except Exception as e:
            print(f"❌ [Model] Error loading Florence-2: {str(e)}", flush=True)
            raise e

        try:
            print("🔵 [Model] Loading SAM2 model...", flush=True)
            self.sam2_checkpoint = "./sam2_hiera_large.pt"
            self.sam2_config = "./configs/sam2/sam2_hiera_l.yaml"
            self.sam2_model = build_sam2(self.sam2_config, self.sam2_checkpoint, device=self.device)
            self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
            print("✅ [Model] SAM2 model loaded", flush=True)
        except Exception as e:
            print(f"❌ [Model] Error loading SAM2: {str(e)}", flush=True)
            raise e

        print("🏁 [Model] __init__ completed", flush=True)

    def predict(self, image_bytes, caption_type="caption"):
        print("🔵 [Model] predict() called", flush=True)

        try:
            image = Image.open(image_bytes).convert("RGB")
            print("✅ [Model] Image loaded and converted to RGB", flush=True)
        except Exception as e:
            print(f"❌ [Model] Error loading image: {str(e)}", flush=True)
            raise e

        try:
            task_prompt = {
                "caption": "<CAPTION>",
                "detailed_caption": "<DETAILED_CAPTION>",
                "more_detailed_caption": "<MORE_DETAILED_CAPTION>"
            }[caption_type]
            print(f"📝 [Model] Task prompt selected: {task_prompt}", flush=True)
        except Exception as e:
            print(f"❌ [Model] Error selecting task prompt: {str(e)}", flush=True)
            raise e

        try:
            print("🏃 [Model] Running Florence2 captioning...", flush=True)
            caption_results = self.run_florence2(task_prompt, None, self.florence2_model,self.florence2_processor,image)
            print("✅ [Model] Florence2 captioning done", flush=True)
        except Exception as e:
            print(f"❌ [Model] Error during Florence2 captioning: {str(e)}", flush=True)
            raise e

        text_input = caption_results[task_prompt]

        try:
            print("🏃 [Model] Running Florence2 grounding...", flush=True)
            grounding_results = self.run_florence2('<CAPTION_TO_PHRASE_GROUNDING>',text_input, self.florence2_model, self.florence2_processor, image)
            grounding_results = grounding_results['<CAPTION_TO_PHRASE_GROUNDING>']
            print("✅ [Model] Florence2 grounding done", flush=True)
        except Exception as e:
            print(f"❌ [Model] Error during grounding: {str(e)}", flush=True)
            raise e

        try:
            input_boxes = np.array(grounding_results["bboxes"])
            class_names = grounding_results["labels"]
            print(f"📦 [Model] Got {len(class_names)} boxes", flush=True)
        except Exception as e:
            print(f"❌ [Model] Error parsing grounding results: {str(e)}", flush=True)
            raise e

        try:
            print("🏃 [Model] Running SAM2 predictor...", flush=True)
            self.sam2_predictor.set_image(np.array(image))
            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
            print("✅ [Model] SAM2 prediction done", flush=True)
        except Exception as e:
            print(f"❌ [Model] Error during SAM2 prediction: {str(e)}", flush=True)
            raise e

        output = {
            "class_names": class_names,
            "bboxes": input_boxes.tolist()
        }
        print("🏁 [Model] predict() finished", flush=True)
        return output
    

    def run_florence2(self, task_prompt, text_input, model, processor, image):
        assert model is not None, "You should pass the init florence-2 model here"
        assert processor is not None, "You should set florence-2 processor here"

        device = model.device

        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input
        
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch.float16)
        generated_ids = model.generate(
        input_ids=inputs["input_ids"].to(device),
        pixel_values=inputs["pixel_values"].to(device),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(image.width, image.height)
        )
        return parsed_answer
