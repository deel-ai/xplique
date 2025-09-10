from typing import Callable, List
from math import ceil
from abc import ABC, abstractmethod

import torch
import tensorflow as tf

# from xplique.utils_functions.object_detection.common.box_manager import NumpyBoxManager
# from xplique.utils_functions.object_detection.torch.box_manager import TorchBoxManager, TorchBoxCoordinatesTranslator

from xplique.utils_functions.object_detection.common.box_formatter import XpliqueBoxFormatter


class LatentData(ABC):

    @abstractmethod
    def get_activations(self):
        raise NotImplementedError("get_activations method must be implemented by subclasses")

    @abstractmethod
    def set_activations(self, values):
        raise NotImplementedError("set_activations method must be implemented by subclasses")

    @abstractmethod
    def aggregate(self, *latent_data_list: 'LatentData') -> 'LatentData':
        raise NotImplementedError("aggregate method must be implemented by subclasses")

    @classmethod
    def aggregate_class(cls, *latent_data_list: 'LatentData') -> 'LatentData':
        if not latent_data_list:
            raise ValueError("latent_data_list cannot be empty")
        if len(latent_data_list) == 1:
            return latent_data_list[0]
        
        first_element = latent_data_list[0]
        if not all(isinstance(data, type(first_element)) for data in latent_data_list):
            raise TypeError("All elements must be instances of the same subclass of LatentData")

        # return type(first_element).aggregate(*latent_data_list)
        return first_element.aggregate(*latent_data_list[1:])

class TorchLatentData(LatentData):

    def __init__(self, features: List, pos: List[torch.Tensor]): # -> pas de pos ici, ça devrait etre dans Detr uniquement
        self.features = features
        self.pos = pos

    @abstractmethod
    def detach(self) -> 'LatentData':
        raise NotImplementedError("detach method must be implemented by subclasses")


class LatentExtractor:
    def __init__(self, model: Callable,
                       input_to_latent_model: Callable,
                       latent_to_logit_model: Callable,
                       latent_data_class=LatentData,
                       output_formatter: XpliqueBoxFormatter = None,
                       batch_size: int = 8):
        self.model = model
        self.input_to_latent_model = input_to_latent_model
        self.latent_to_logit_model = latent_to_logit_model
        self.latent_data_class = latent_data_class
        self.output_formatter = output_formatter
        self.batch_size = batch_size
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def input_to_latent(self, inputs) -> LatentData:
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def latent_to_logit(self, latent_data: LatentData):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def forward(self, samples):
        latent_data = self.input_to_latent(samples)
        return self.latent_to_logit(latent_data)

class TorchLatentExtractor(LatentExtractor):

    def __init__(self, model: Callable,
                       input_to_latent_model: Callable,
                       latent_to_logit_model: Callable,
                       latent_data_class=LatentData,
                       output_formatter: XpliqueBoxFormatter = None,
                       batch_size: int = 8,
                       device: str='cuda'):
        super().__init__(model, input_to_latent_model, latent_to_logit_model, latent_data_class, output_formatter, batch_size)
        self.model = self.model.to(device)
        self.device = device
        self.training = self.model.training
    
    def eval(self):
        self.model.eval()
        return self
 
    def to(self, device):
        self.model.to(device)
        return self

    def zero_grad(self):
        self.model.zero_grad()
        return self

    def forward(self, samples):
        latent_data = self.input_to_latent_model(samples)
        outputs = self.latent_to_logit_model(latent_data)
        if self.output_formatter:
            outputs = self.output_formatter(outputs)
        return outputs

    def forward_batched(self, samples):
        results = []
        for latent_data in self._input_to_latent_generator(samples):
            output = self.latent_to_logit_model(latent_data)
            if self.output_formatter:
                output = self.output_formatter(output)
            results.append(output)
        results = torch.cat(results, dim=0)
        # results = np.concatenate(results, axis=0)
        return results

    ## input -> latent
    # no batch
    def input_to_latent(self, inputs) -> LatentData:
        # inputs: (N, C, H, W)
        if len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(0)
        latent_data = self.input_to_latent_model(inputs)
        return latent_data
    
    # batched
    def input_to_latent_batched(self, inputs, resize=None) -> LatentData:
        # TODO: faire un for + yield pour chaque batch
        latent_data_list = list(self._input_to_latent_generator(inputs, resize))
        return LatentData.aggregate_class(*latent_data_list)
    

    
    def _input_to_latent_generator(self, inputs, resize=None):
        # inputs: (N, C, H, W)
        if len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(0) # add an extra dim in case we get only 1 image to predict

        # dataset: (N, C, H, W)
        nb_batchs = ceil(len(inputs) / self.batch_size)
        start_ids = [i*self.batch_size for i in range(nb_batchs)]
        
        print(f"Total batches: {nb_batchs}, Batch size: {self.batch_size}, Dataset size: {len(inputs)}")
        with torch.no_grad():
            for i in start_ids:
                i_end = min(i + self.batch_size, len(inputs))
                batch = inputs[i:i_end].to(self.device)

                if resize:
                    batch = torch.nn.functional.interpolate(batch, size=resize, mode='bilinear',
                                                            align_corners=False)

                latent_data = self.input_to_latent_model(batch)
                del batch
                # check ici !
                latent_data = latent_data.to(self.device)
                torch.cuda.empty_cache()
                yield latent_data

    ## latent -> output
    # no batch
    def latent_to_logit(self, latent_data):
        output = self.latent_to_logit_model(latent_data)
        if self.output_formatter:
            output = self.output_formatter(output)
        return output
    
    # batched
    def latent_to_logit_batched(self, latent_data: LatentData, no_grad=True):
        if no_grad:
            with torch.no_grad():
                output_list = list(self._latent_to_logit_generator(latent_data))
        else:
            output_list = list(self._latent_to_logit_generator(latent_data))
        result = torch.cat(output_list, dim=0)
        return result

    def _latent_to_logit_generator(self, latent_data: LatentData):
        # dataset: (N, C, H, W)
        nb_batchs = ceil(len(latent_data) / self.batch_size)
        start_ids = [i*self.batch_size for i in range(nb_batchs)]

        #with torch.no_grad():
        for i in start_ids:
            batch = latent_data[i:i+self.batch_size].to(self.device)
            boxes_scores_labels = self.latent_to_logit_model(batch)
            del batch
            if self.output_formatter:
                # boxes_scores_labels = [self.output_formatter(output) for output in boxes_scores_labels]
                boxes_scores_labels = self.output_formatter(boxes_scores_labels)
            yield boxes_scores_labels

    # def batch_latent_inference(self,
    #                             model: torch.nn.Module,
    #                             dataset: torch.Tensor,
    #                             batch_size: int = 128,
    #                             resize: Optional[int] = None,
    #                             device: str='cuda') -> LatentData:
    #     """
    #     Returns
    #     -------
    #     activations
    #         The latent activations of shape (n_samples, channels, height, width).
    #     """
    #     # dataset: (N, C, H, W)
    #     nb_batchs = ceil(len(dataset) / batch_size)
    #     start_ids = [i*batch_size for i in range(nb_batchs)]
        
    #     # print(f"Total batches: {nb_batchs}, Batch size: {batch_size}, Dataset size: {len(dataset)}")
            
    #     with torch.no_grad():
    #         for i in start_ids:
    #             i_end = min(i + batch_size, len(dataset))
    #             # print(f"Processing batch from {i} to {i_end}...")
    #             # print(f"_batch_inference for batch {i} to {i_end} (max is {len(dataset)})")
    #             batch = dataset[i:i_end].to(device)

    #             if resize:
    #                 batch = torch.nn.functional.interpolate(batch, size=resize, mode='bilinear',
    #                                                         align_corners=False)

    #             latent_data = model(batch)
    #             del batch
    #             # check ici !
    #             latent_data = latent_data.to('cpu')
    #             torch.cuda.empty_cache()
    #             # print(f"Yielding latent data of len: {len(latent_data)}")
    #             yield latent_data

    # def latent_predict(self, inputs: torch.Tensor, resize=None) -> LatentData:
    #     # inputs: (N, C, H, W)
    #     if len(inputs.shape) == 3:
    #         inputs = inputs.unsqueeze(0) # add an extra dim in case we get only 1 image to predict
    #     latent_data_list = list(self.batch_latent_inference(self.input_to_latent_model, inputs,
    #                                             self.batch_size, resize, device=self.device))
    #     return LatentData.aggregate_class(*latent_data_list)




    # def batch_output_inference(self, model: torch.nn.Module,
    #                             dataset: LatentData,
    #                             batch_size: int = 128,
    #                             resize: Optional[int] = None,
    #                             device: str='cuda',
    #                             no_grad: bool = True):
    #     # dataset: (N, C, H, W)
    #     nb_batchs = ceil(len(dataset) / batch_size)
    #     start_ids = [i*batch_size for i in range(nb_batchs)]

    #     context = torch.no_grad() if no_grad else nullcontext()
    #     with context:
    #         for i in start_ids:
    #             # print("\t\t [_batch_logit_inference] selecting the activation batch from latent data ...")
    #             batch = dataset[i:i+batch_size].to(device)
    #             boxes_scores_labels = model(batch) # keys: 'pred_logits', 'pred_boxes' / fcos: list of ['boxes', 'scores', 'labels']
    #             del batch
    #             yield boxes_scores_labels



class TfLatentExtractor(LatentExtractor):

    def __init__(self, model: Callable,
                       input_to_latent_model: Callable,
                       latent_to_logit_model: Callable,
                       latent_data_class=LatentData,
                       output_formatter: XpliqueBoxFormatter = None,
                       batch_size: int = 8):
        super().__init__(model, input_to_latent_model, latent_to_logit_model, latent_data_class, output_formatter, batch_size)
    
    def forward(self, samples):
        latent_data = self.input_to_latent_model(samples)
        outputs = self.latent_to_logit_model(latent_data)
        if self.output_formatter:
            outputs = self.output_formatter(outputs)
        return outputs

    def forward_batched(self, samples):
        results = []
        for latent_data in self._input_to_latent_generator(samples):
            output = self.latent_to_logit_model(latent_data)
            if self.output_formatter:
                output = self.output_formatter(output)
            results.append(output)
        results = tf.concat(results, axis=0)
        return results

    ## input -> latent
    # no batch
    def input_to_latent(self, inputs) -> LatentData:
        # inputs: (N, C, H, W)
        if len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(0)
        latent_data = self.input_to_latent_model(inputs)
        return latent_data
    
    # batched
    def input_to_latent_batched(self, inputs, resize=None) -> LatentData:
        # TODO: faire un for + yield pour chaque batch
        latent_data_list = list(self._input_to_latent_generator(inputs, resize))
        print(f"[TfLatentExtractor] Number of latent data batches: {len(latent_data_list)}")
        return LatentData.aggregate_class(*latent_data_list)

    def _input_to_latent_generator(self, inputs, resize=None):
        # inputs: (N, C, H, W)
        if len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(0) # add an extra dim in case we get only 1 image to predict

        # dataset: (N, C, H, W)
        nb_batchs = ceil(len(inputs) / self.batch_size)
        start_ids = [i*self.batch_size for i in range(nb_batchs)]
        
        print(f"Total batches: {nb_batchs}, Batch size: {self.batch_size}, Dataset size: {len(inputs)}")
        with torch.no_grad():
            for i in start_ids:
                i_end = min(i + self.batch_size, len(inputs))
                batch = inputs[i:i_end] # .to(self.device) # diff avec pytorch

                if resize:
                    batch = tf.image.resize(batch, size=resize)

                print(f"[TfLatentExtractor] Calling input_to_latent_model(batch)...")
                latent_data = self.input_to_latent_model(batch)
                del batch
                # check ici !
                yield latent_data

    ## latent -> output
    # no batch
    def latent_to_logit(self, latent_data):
        output = self.latent_to_logit_model(latent_data)
        if self.output_formatter:
            output = self.output_formatter(output)
        return output
    
    # batched
    def latent_to_logit_batched(self, latent_data: LatentData):
        output_list = list(self._latent_to_logit_generator(latent_data))
        result = tf.concat(output_list, axis=0)
        return result

    def _latent_to_logit_generator(self, latent_data: LatentData):
        # dataset: (N, C, H, W)
        nb_batchs = ceil(len(latent_data) / self.batch_size)
        start_ids = [i*self.batch_size for i in range(nb_batchs)]

        for i in start_ids:
            batch = latent_data[i:i+self.batch_size]
            boxes_scores_labels = self.latent_to_logit_model(batch)
            del batch
            if self.output_formatter:
                boxes_scores_labels = self.output_formatter(boxes_scores_labels)
            yield boxes_scores_labels

