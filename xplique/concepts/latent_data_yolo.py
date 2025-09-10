from typing import Callable, Optional, Tuple, List
import torch
from torch import Tensor
from .holistic_craft_object_detection import *


class LatentDataYolo(LatentData):

    def __init__(self, preds, im, im0s):
        self.preds = preds  # a list of [torch.Tensor, [torch.Tensor, torch.Tensor, torch.Tensor]]
        self.im = im        # a torch.Tensor
        self.im0s = im0s    # a torch.Tensor
        
        # # typically
        # preds[0].shape: torch.Size([1, 84, 8400])
        # preds[1][0].shape: torch.Size([1, 144, 80, 80])
        # preds[1][1].shape: torch.Size([1, 144, 40, 40])
        # preds[1][2].shape: torch.Size([1, 144, 20, 20])
        # im shape: torch.Size([1, 3, 640, 640])
        # im0s shape: torch.Size([1, 3, 640, 640])

    def print_shapes(self):
        print("preds shape:", self.preds.shape if isinstance(self.preds, torch.Tensor) else "list of length "+str(len(self.preds)))
        print("preds[0].shape:", self.preds[0].shape)
        print("preds[1][0].shape:", self.preds[1][0].shape)
        print("preds[1][1].shape:", self.preds[1][1].shape)
        print("preds[1][2].shape:", self.preds[1][2].shape)
        print("im shape:", self.im.shape)
        print("im0s shape:", self.im0s.shape if isinstance(self.im0s, torch.Tensor) else "list of length "+str(len(self.im0s)))

    def get_activations(self):
        activations = self.preds
        is_4d = len(activations.shape) == 4
        if is_4d: 
            # torch -> tensorflow/numpy
            # activations: (N, C, H, W) -> (N, H, W, C)
            activations = activations.permute(0, 2, 3, 1)
        return activations

    def set_activations(self, values):
        values = values.permute(0, 3, 1, 2)        
        self.preds = values

    def __getitem__(self, index):
        if isinstance(index, int):
            index = slice(index, index + 1)        
        # Slice the features and pos using the given index
        sliced_preds0 = self.preds[0][index]
        sliced_preds1 = [fm[index] for fm in self.preds[1]]
        sliced_preds = [sliced_preds0, sliced_preds1]
        sliced_im = self.im[index]
        if isinstance(self.im0s, torch.Tensor):
            sliced_im0s = self.im0s[index]
        else:
            sliced_im0s = [self.im0s[i] for i in range(len(self.im0s))][index]
        return LatentDataYolo(sliced_preds, sliced_im, sliced_im0s)

    @classmethod
    def aggregate(self, *latent_data_list: 'LatentDataYolo') -> 'LatentDataYolo':
        # Aggregate several LatentDataYolo instances
            
        if not latent_data_list:
            raise ValueError("empty latent_data_list")
        
        first = latent_data_list[0]
        # Aggregate preds
        # preds[0] : main output tensor
        aggregated_preds0 = torch.cat([data.preds[0] for data in latent_data_list], dim=0)
        
        # preds[1] (feature maps list)
        aggregated_preds1 = []
        for i in range(len(first.preds[1])):
            aggregated_feature_map = torch.cat([data.preds[1][i] for data in latent_data_list], dim=0)
            aggregated_preds1.append(aggregated_feature_map)

        aggregated_preds = [aggregated_preds0, aggregated_preds1]
        
        # Images (im)
        aggregated_im = torch.cat([data.im for data in latent_data_list], dim=0)
        
        # Original images (im0s)
        # if isinstance(first.im0s, torch.Tensor):
        aggregated_im0s = torch.cat([data.im0s for data in latent_data_list], dim=0)
        # else:
        #     # Si im0s est une liste, fusionner les listes
        #     aggregated_im0s = []
        #     for data in latent_data_list:
        #         if isinstance(data.im0s, list):
        #             aggregated_im0s.extend(data.im0s)
        #         else:
        #             aggregated_im0s.append(data.im0s)

        return LatentDataYolo(aggregated_preds, aggregated_im, aggregated_im0s)
        



import types

from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple
from torch import Tensor
import torchvision
from xplique.concepts.latent_extractor import TorchLatentExtractor
from xplique.concepts.latent_extractor import YoloBoxFormatter,  BasicYoloBoxFormatter

# Yolo:
# model.predict()
# -> model.predictor.stream_inference()
#
#
#

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops
from ultralytics.utils import DEFAULT_CFG, LOGGER, MACOS, WINDOWS, callbacks, colorstr, ops
from ultralytics.utils.torch_utils import select_device, smart_inference_mode

from pathlib import Path
import types

def buildTorchYoloLatentExtractor(model: Callable, nb_classes: int) -> 'TorchLatentExtractor':

    # Create predictor
    def make_predictor(model):

        @smart_inference_mode()
        def stream_inference_g(self, source=None, model=None, *args, **kwargs):
            print("stream_inference_g called")
            # Setup model
            if not self.model:
                self.setup_model(model)

            results = []
            with self._lock:  # for thread-safe inference
                # Setup source every time predict is called
                self.setup_source(source if source is not None else self.args.source)

                # Check if save_dir/ label file exists
                if self.args.save or self.args.save_txt:
                    (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

                # Warmup model
                if not self.done_warmup:
                    self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
                    self.done_warmup = True

                self.seen, self.windows, self.batch = 0, [], None
                self.run_callbacks("on_predict_start")
                for self.batch in self.dataset:
                    self.run_callbacks("on_predict_batch_start")
                    paths, im0s, s = self.batch

                    # Preprocess
                    im = self.preprocess(im0s)

                    # Inference
                    preds = self.inference(im, *args, **kwargs)
                    print("preds len:", len(preds))
                    # yield from [preds] if isinstance(preds, torch.Tensor) else preds  # yield embedding tensors
                    latent_data = LatentDataYolo(preds, im, im0s)
                    latent_data.print_shapes()
                    yield latent_data
            
        @smart_inference_mode()
        def stream_inference_h(self, latent_data: LatentDataYolo):
            print("stream_inference_h called")
            # Process preds
            # for (preds, im, im0s) in latent_data:
            # print("Processing pred:", preds)
            self.results = self.postprocess(latent_data.preds, latent_data.im, latent_data.im0s)
            yield from self.results

        # Build default predictor
        custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict", "rect": True}  # method defaults
        args = {**model.overrides, **custom}  # highest priority args on the right

        predictor = model._smart_load("predictor")(overrides=args, _callbacks=model.callbacks)
        predictor.setup_model(model=model.model, verbose=False)
        
        # Add g and h methods to predictor

        # model.predictor.stream_inference_g = types.MethodType(stream_inference_g, model.predictor)
        # model.predictor.stream_inference_h = types.MethodType(stream_inference_h, model.predictor)
        predictor.stream_inference_g = types.MethodType(stream_inference_g, predictor)
        predictor.stream_inference_h = types.MethodType(stream_inference_h, predictor)
        
        return predictor

    # if not model.predictor:
    #     raise ValueError('no predictor in model !')

    custom_predictor = make_predictor(model)

    def g(self, samples) -> LatentDataYolo:
        self.predictor = custom_predictor
        latent_data_list = list(self.predictor.stream_inference_g(source=samples, model=self.model))
        return LatentDataYolo.aggregate(*latent_data_list)

    def h(self, latent_data: LatentDataYolo) -> Tensor:
        self.predictor = custom_predictor
        results = list(self.predictor.stream_inference_h(latent_data))
        return results # todo: list or tensor ?
    

    model.g = types.MethodType(g, model)
    model.h = types.MethodType(h, model)

    # processed_formatter = YoloBoxFormatter(nb_classes=nb_classes)
    processed_formatter = BasicYoloBoxFormatter(nb_classes=nb_classes)
    latent_extractor = TorchLatentExtractor(model, model.g, model.h, latent_data_class=LatentDataYolo, output_formatter=processed_formatter, batch_size=1)
    return latent_extractor



def buildTorchYoloLatentExtractorWithGradients(model: Callable, nb_classes: int) -> 'TorchLatentExtractor':

    # Create predictor
    def make_predictor(model):

        @smart_inference_mode()
        def stream_inference_g(self, source=None, model=None, *args, **kwargs):
            print("stream_inference_g called")
            # Setup model
            if not self.model:
                self.setup_model(model)

            results = []
            with self._lock:  # for thread-safe inference
                # Setup source every time predict is called
                self.setup_source(source if source is not None else self.args.source)

                # Check if save_dir/ label file exists
                if self.args.save or self.args.save_txt:
                    (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

                # Warmup model
                if not self.done_warmup:
                    self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
                    self.done_warmup = True

                self.seen, self.windows, self.batch = 0, [], None
                self.run_callbacks("on_predict_start")
                for self.batch in self.dataset:
                    self.run_callbacks("on_predict_batch_start")
                    paths, im0s, s = self.batch

                    # Preprocess
                    im = self.preprocess(im0s)

                    # Inference
                    preds = self.inference(im, *args, **kwargs)
                    print("preds len:", len(preds))
                    # yield from [preds] if isinstance(preds, torch.Tensor) else preds  # yield embedding tensors
                    latent_data = LatentDataYolo(preds, im, im0s)
                    latent_data.print_shapes()
                    yield latent_data
            
        @smart_inference_mode()
        def stream_inference_h(self, latent_data: LatentDataYolo):
            print("stream_inference_h called")
            # Process preds
            # for (preds, im, im0s) in latent_data:
            # print("Processing pred:", preds)
            # self.results = self.postprocess(latent_data.preds, latent_data.im, latent_data.im0s)
            self.results = latent_data.preds
            yield from self.results

        # Build default predictor
        custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict", "rect": True}  # method defaults
        args = {**model.overrides, **custom}  # highest priority args on the right

        predictor = model._smart_load("predictor")(overrides=args, _callbacks=model.callbacks)
        predictor.setup_model(model=model.model, verbose=False)
        
        # Add g and h methods to predictor

        # model.predictor.stream_inference_g = types.MethodType(stream_inference_g, model.predictor)
        # model.predictor.stream_inference_h = types.MethodType(stream_inference_h, model.predictor)
        predictor.stream_inference_g = types.MethodType(stream_inference_g, predictor)
        predictor.stream_inference_h = types.MethodType(stream_inference_h, predictor)
        
        return predictor

    # if not model.predictor:
    #     raise ValueError('no predictor in model !')

    custom_predictor = make_predictor(model)

    def g(self, samples) -> LatentDataYolo:
        self.predictor = custom_predictor
        latent_data_list = list(self.predictor.stream_inference_g(source=samples, model=self.model))
        return LatentDataYolo.aggregate(*latent_data_list)

    def h(self, latent_data: LatentDataYolo) -> Tensor:
        self.predictor = custom_predictor
        results = list(self.predictor.stream_inference_h(latent_data))
        return results # todo: list or tensor ?
    

    model.g = types.MethodType(g, model)
    model.h = types.MethodType(h, model)

    processed_formatter = YoloBoxFormatter(nb_classes=nb_classes)
    latent_extractor = TorchLatentExtractor(model, model.g, model.h, latent_data_class=LatentDataYolo, output_formatter=processed_formatter, batch_size=1)
    return latent_extractor
