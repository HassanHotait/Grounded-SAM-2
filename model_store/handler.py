# handler.py

import io
import torch
from model import GroundedSAM2Florence2
from ts.torch_handler.base_handler import BaseHandler
from PIL import Image
import sys
import logging

class GroundedSAM2Florence2Handler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False
        self.model = None

    def initialize(self, context):
        sys.stdout = open('/home/appuser/Grounded-SAM-2/logs/model_log.log', 'a')
        sys.stderr = open('/home/appuser/Grounded-SAM-2/logs/model_log.log', 'a')

        print("🔥 [Handler] initialize() called", flush=True)

        try:
            self.model = GroundedSAM2Florence2()
            print("✅ [Handler] GroundedSAM2Florence2() created", flush=True)
            self.initialized = True
        except Exception as e:
            print(f"❌ [Handler] Error during initialize: {str(e)}", flush=True)
            raise e

    def preprocess(self, data):
        print("📥 [Handler] preprocess() called", flush=True)
        try:
            image_input = data[0].get("data") or data[0].get("body")
            image_bytes = io.BytesIO(image_input)
            print("✅ [Handler] Image data extracted", flush=True)
            return image_bytes
        except Exception as e:
            print(f"❌ [Handler] Error in preprocess: {str(e)}", flush=True)
            raise e

    def inference(self, image_bytes):
        print("🤖 [Handler] inference() called", flush=True)
        try:
            output = self.model.predict(image_bytes)
            print("✅ [Handler] inference() completed", flush=True)
            return output
        except Exception as e:
            print(f"❌ [Handler] Error in inference: {str(e)}", flush=True)
            raise e

    def postprocess(self, inference_output):
        print("📤 [Handler] postprocess() called", flush=True)
        return [inference_output]
