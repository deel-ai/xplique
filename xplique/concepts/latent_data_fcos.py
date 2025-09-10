import torch, torchvision
from .holistic_craft_object_detection import *

class LatentDataFcos(LatentData):
    index_activations = -1
    def __init__(self,
                 images: torchvision.models.detection.image_list.ImageList, # ImageList contains 'tensors(tensor)' and 'image_sizes(list of int)'
                 original_image_sizes: List,
                 resnet_features: OrderedDict):
        self.images = images
        self.original_image_sizes = original_image_sizes
        self.resnet_features = resnet_features
    
    def __len__(self) -> int:
        last_key = list(self.resnet_features.keys())[self.index_activations]
        return self.resnet_features[last_key].shape[0]
        
    def detach(self):
        for key, value in self.resnet_features.items():
            self.resnet_features[key] = value.detach()
    
    def get_activations(self):
        last_key = list(self.resnet_features.keys())[self.index_activations]
        activations = self.resnet_features[last_key]# return the features of the last resnet layer
        # ici
        is_4d = len(activations.shape) == 4
        if is_4d: 
            # torch -> tensorflow/numpy
            # activations: (N, C, H, W) -> (N, H, W, C)            
            activations = activations.permute(0, 2, 3, 1)
        return activations

    def set_activations(self, values):
        
        # tensorflow/numpy -> torch
        # activations: (N, H, W, C) -> (N, C, H, W)
        # ici
        values = values.permute(0, 3, 1, 2) 
        
        last_key = list(self.resnet_features.keys())[self.index_activations]
        if type(values) == torch.Tensor:
            self.resnet_features[last_key] = values.clone().detach()
        elif type(values) == np.ndarray:
            self.resnet_features[last_key] = torch.from_numpy(values)
        else:
            raise Exception("unknown values type")
        # self.resnet_features[last_key] = values.clone().detach()
 
    def to(self, device: torch.device) -> 'LatentData':
        images = torchvision.models.detection.image_list.ImageList(
            self.images.tensors.to(device),
            self.images.image_sizes
        )
        resnet_features = OrderedDict()

        for key, value in self.resnet_features.items():
            resnet_features[key] = value.to(device)

        return LatentDataFcos(images, self.original_image_sizes, resnet_features)

    @classmethod
    def aggregate(self, *latent_data_list: 'LatentDataFcos') -> 'LatentDataFcos':
        if not latent_data_list:
            raise ValueError("latent_data_list can not be empty")

        
        # gather tensors for ImageList property 1
        tensors = torch.cat([ld.images.tensors for ld in latent_data_list])
        # gather image sizes for ImageList property 2
        image_sizes = []
        for ld in latent_data_list:
            image_sizes.extend(ld.images.image_sizes)
        # create aggregated ImageList
        images = torchvision.models.detection.image_list.ImageList(tensors, image_sizes)
        
        # create aggregated image sizes
        original_image_sizes = [size for ld in latent_data_list for size in ld.original_image_sizes]
        
        # create aggregated OrderedDict
        resnet_features = OrderedDict()
        for key in latent_data_list[0].resnet_features.keys():
            # print(f"key: {key}")
            # li = [ld.resnet_features[key].detach().cpu().numpy() for ld in latent_data_list]
            li = [ld.resnet_features[key] for ld in latent_data_list]
            # print(f"nb resnet features for this key: {len(li)}")
            # merged_dict_values = torch.cat(li)
            # print(f"merging {len(li)} elements of shape {li[0].shape} into a single tensor")
            try:
                # merged_dict_values = np.concatenate(li, 0)
                merged_dict_values = torch.cat(li)           
                # print(f"shape of final values tensor: {merged_dict_values.shape}")
                resnet_features[key] = merged_dict_values
            finally:
                # print("end of merging dict")
                pass
            # cleanup
            

        # print("finished merging the 3 properties")
        # return cls(images, original_image_sizes, resnet_features)
        return LatentDataFcos(images, original_image_sizes, resnet_features)

    def __getitem__(self, index):
        if isinstance(index, int):
            index = slice(index, index + 1)
        
        # images & original_image_sizes have the same small size
        nb_imageslist = len(self.images.image_sizes)
        start = index.start % nb_imageslist if index.start is not None else None
        stop = start + (index.stop - index.start)
        step = index.step
        small_index = slice(start, stop, step)
        
        images_subset = torchvision.models.detection.image_list.ImageList(
            self.images.tensors[small_index],
            self.images.image_sizes[small_index]
        )
        original_image_sizes_subset = self.original_image_sizes[small_index]
        
        # this dict can have a different size, larger than images & original_image_sizes (resnet_features -> patches)
        # BUT only for the last key !
        resnet_features_subset = OrderedDict()
        last_key = list(self.resnet_features.keys())[-1]
        for key in self.resnet_features.keys():
            if key != last_key:
                resnet_features_subset[key] = self.resnet_features[key][small_index]
            else:
                # last key -> here we need to specifically select the right index
                resnet_features_subset[key] = self.resnet_features[key][index]

        return LatentDataFcos(images_subset, original_image_sizes_subset, resnet_features_subset)

    def __setitem__(self, index, value):
        if not isinstance(value, LatentData):
            raise TypeError("value must be a LatentData object")

        if isinstance(index, int):
            index = slice(index, index + 1)

        # images & original_image_sizes have the same small size
        nb_imageslist = len(self.images.image_sizes)
        start = index.start % nb_imageslist if index.start is not None else None
        stop = start + (index.stop - index.start)
        step = index.step
        small_index = slice(start, stop, step)
        
        self.images.tensors[small_index] = value.images.tensors
        self.images.image_sizes[small_index] = value.images.image_sizes
        self.original_image_sizes[small_index] = value.original_image_sizes

        last_key = list(self.resnet_features.keys())[-1]
        for key in self.resnet_features.keys():
            if key != last_key:
                self.resnet_features[key][small_index] = value.resnet_features[key]
            else:
                self.resnet_features[key][index] = value.resnet_features[key]

    def pdim(self):
        print(f"latent data dimensions (CRAFT operating on activations layer {self.index_activations}):")
        print("images.tensors len:\t", len(self.images.tensors))
        print("images.tensors[0]:\t", self.images.tensors[0].shape)
        print("images.tensors[1]:\t", self.images.tensors[1].shape)
        print("original_image_sizes:\t", self.original_image_sizes)
        print("resnet_features['0']:\t", self.resnet_features['0'].shape)
        print("resnet_features['1']:\t", self.resnet_features['1'].shape)
        print("resnet_features['2']:\t", self.resnet_features['2'].shape)


class FcosWrapper(ObjectDetectionModelWrapper):

    def output_predict(self, latent_data: LatentDataFcos, resize=None) -> List[ObjectDetectionOutput]:
        output = self.batch_output_inference(self.latent_to_logit_model, latent_data,
                                  self.batch_size, resize,  device=self.device)
        
        output_list = list(output)
        dict_list = flatten_list(output_list)
        dict_list = [dict_to_cpu(d) for d in dict_list]
        del output_list
        torch.cuda.empty_cache()
        
        output_fcos = []
        for i in range(len(dict_list)):
            scores = dict_list[i]['scores']
            labels = dict_list[i]['labels']
            raw_boxes = dict_list[i]['boxes']
            output_fcos.append(ObjectDetectionOutput(None, scores, labels, raw_boxes))
        return output_fcos
    
    def predict_score_for_class(self, latent_data: LatentDataFcos, class_id:int, top_k:int, no_grad:bool = True):
        context = torch.no_grad() if no_grad else nullcontext()
        latent_data = latent_data.to(self.device)
        with context:
            boxes_scores_labels = self.latent_to_logit_model(latent_data)            
            probas_for_our_class = []
            for out in boxes_scores_labels:
                keep = out['labels'].cpu() == class_id
                scores = out['scores'].cpu()[keep]
                scores.sort(descending=True)[:top_k]
                probas_for_our_class.append(scores.mean())
            probas_for_our_class = torch.stack(probas_for_our_class)
            del latent_data, boxes_scores_labels
            torch.cuda.empty_cache()
            return probas_for_our_class
        