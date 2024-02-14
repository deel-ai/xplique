import numpy as np
import torch
import torch.nn as nn

from torch.nn import MSELoss
from transformers import RobertaPreTrainedModel, RobertaModel
from transformers import RobertaTokenizerFast
from xplique.concepts import CockatielTorch as Cockatiel
from xplique.commons.torch_operations import NlpPreprocessor


class FakeImdbClassifier(torch.nn.Module):
    def __init__(self, nb_classes):
        super().__init__()
        self.nb_classes = nb_classes

    def features(self, **kwargs):
        start_value = 0
        return torch.arange(start_value,
                            start_value + kwargs['input_ids'].shape[0] * self.nb_classes).\
            reshape(kwargs['input_ids'].shape[0], self.nb_classes)

    def classifier(self, latent):
        # Simulates a classifier, returns alternatively [0, 1] and [1, 0]
        # as result
        n = len(latent)
        pattern = torch.cat([torch.tensor([0, 1]), torch.tensor([1, 0])])
        ypred = torch.cat([pattern] * n).reshape(n * 2, -1)
        return ypred

    def forward(self, **kwargs):
        return self.classifier(self.features(**kwargs))


class ImdbPreprocessor(NlpPreprocessor):
    def preprocess(self, inputs: np.ndarray, labels: np.ndarray):
        preprocessed_inputs = self.tokenize(samples=inputs.tolist())
        preprocessed_labels = torch.Tensor(np.array(
            labels.tolist()) == 'positive').int().to(self.device)
        return preprocessed_inputs, preprocessed_labels


def test_shape():
    """Ensure the output shape is correct"""

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    pretrained_model_path = "wrmurray/roberta-base-finetuned-imdb"
    tokenizer = RobertaTokenizerFast.from_pretrained(pretrained_model_path)
    nb_classes = 2
    model = FakeImdbClassifier(nb_classes=nb_classes)
    model = model.eval()

    imdb_preprocessor = ImdbPreprocessor(tokenizer,
                                         device,
                                         padding="max_length",
                                         max_length=512,
                                         truncation=True,
                                         return_tensors='pt')

    input_to_latent_model = model.features
    latent_to_logit_model = model.classifier
    number_of_concepts = 20
    cockatiel_explainer_pos = Cockatiel(input_to_latent_model=input_to_latent_model,
                                        latent_to_logit_model=latent_to_logit_model,
                                        preprocessor=imdb_preprocessor,
                                        number_of_concepts=number_of_concepts,
                                        batch_size=64,
                                        device=device)

    data = ["Absolutely riveting from beginning to end. Test 1 2 3. Another test.",
            "But it's a great movie.",
            "This is the best movie of the year to date.",
            "The movie is excellent."]
    crops, crops_u, concept_bank_w \
        = cockatiel_explainer_pos.fit(inputs=data,
                                      class_id=1, alpha_w=0)
    assert len(crops) == 6
    assert crops_u.shape == (6, number_of_concepts)
    assert concept_bank_w.shape == (number_of_concepts, nb_classes)

    global_importance_pos = cockatiel_explainer_pos.estimate_importance()
    assert len(global_importance_pos) == number_of_concepts


class CustomRobertaForSequenceClassification(RobertaPreTrainedModel):
    """
    A custom RoBERTa model using a custom fully-connected head with a non-negative layer on which
    we can compute the NMF.

    Parameters
    ----------
    config
        An object indicating the hidden layer size, the presence and amount of dropout for the
        classification head.
    """
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 1
        self.config = config

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, 2)

        self.roberta = RobertaModel(config, add_pooling_layer=False)

        self.mse_loss = MSELoss()

        self.post_init()

    def classifier_features(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.relu(x)
        return x

    def classifier_end_model(self, x):
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

    def classifier(self, x):
        x = self.classifier_features(x)
        x = self.classifier_end_model(x)
        return x

    def features(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        # chain RobertaModel and the classifier features ending with a ReLU
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0][:, 0, :]
        return self.classifier_features(sequence_output)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        activations = self.features(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = self.classifier_end_model(activations)
        return logits


def test_classifier():

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    pretrained_model_path = "wrmurray/roberta-base-finetuned-imdb"
    tokenizer = RobertaTokenizerFast.from_pretrained(pretrained_model_path)
    model = CustomRobertaForSequenceClassification.from_pretrained(
        pretrained_model_path).to(device)
    model = model.eval()

    imdb_preprocessor = ImdbPreprocessor(tokenizer,
                                         device,
                                         padding="max_length",
                                         max_length=512,
                                         truncation=True,
                                         return_tensors='pt')

    input_to_latent_model = model.features
    latent_to_logit_model = model.classifier
    number_of_concepts = 2
    cockatiel_explainer_pos = Cockatiel(input_to_latent_model=input_to_latent_model,
                                        latent_to_logit_model=latent_to_logit_model,
                                        preprocessor=imdb_preprocessor,
                                        number_of_concepts=number_of_concepts,
                                        batch_size=64,
                                        device=device)

    data = ["Absolutely great riveting from beginning to end. It was great. Great great great.",
            "But it's a great movie.",
            "I liked this film.",
            "I enjoyed the scenario of this movie."]  # sentence with 'great', othes not
    crops, crops_u, concept_bank_w \
        = cockatiel_explainer_pos.fit(inputs=data,
                                      class_id=1, alpha_w=0)
    assert len(crops) == 6
    assert crops_u.shape == (6, number_of_concepts)
    assert concept_bank_w.shape[0] == number_of_concepts

    global_importance_pos = cockatiel_explainer_pos.estimate_importance()
    assert len(global_importance_pos) == number_of_concepts
    most_important_concepts_ids = global_importance_pos.argsort()

    nb_excerpts = 2
    nb_most_important_concepts = 2

    best_sentences_per_concept = cockatiel_explainer_pos.get_best_excerpts_per_concept(
        nb_excerpts=nb_excerpts, nb_most_important_concepts=nb_most_important_concepts)
    assert np.all(best_sentences_per_concept[most_important_concepts_ids[0]] == [
        'I enjoyed the scenario of this movie.', 'I liked this film.'])
    assert np.all(best_sentences_per_concept[most_important_concepts_ids[1]] == [
        'Great great great.', 'It was great.'])
