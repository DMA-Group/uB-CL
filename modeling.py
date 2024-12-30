import torch
import torch.nn as nn
import torch.distributed as dist
import os
from transformers.models.mpnet.modeling_mpnet import MPNetPreTrainedModel, MPNetModel, MPNetLMHead
from transformers.file_utils import ModelOutput
from dataclasses import dataclass
import logging
from transformers import AutoTokenizer
from transformers.file_utils import WEIGHTS_NAME, is_torch_tpu_available

logger = logging.getLogger(__name__)

@dataclass
class CLModelOutput(ModelOutput):
    loss: torch.FloatTensor = None
    hidden_states: torch.FloatTensor = None
    attentions: torch.FloatTensor = None

class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "avg"]

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state

        if self.pooler_type in ['cls']:
            return last_hidden[:, 0]  # 直接返回cls
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        else:
            raise NotImplementedError

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features):
        x = self.dense(features)
        x = self.activation(x)

        return x



class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=-1)  # 计算两个张量之间的余弦相似度

    def forward(self, x, y):
        return self.cos(x, y)


class AdaptiveuBCLLoss(nn.Module):
    def __init__(self, initial_lambda=10.0, lambda_shape=(1,), hard_negative=False):
        super(AdaptiveuBCLLoss, self).__init__()
        self.sim = Similarity()
        initial_lambda_matrix = torch.full(lambda_shape, initial_lambda, dtype=torch.float32)
        self.lambda_ = nn.Parameter(initial_lambda_matrix)
        self.hard_negative = hard_negative

    def forward(self, output):
        if not self.hard_negative:
            z1, z2 = output[:, 0], output[:, 1]
            cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
            pos = cos_sim.diag().reshape(-1, 1)
            temp_res = self.lambda_ * (cos_sim - pos)
            loss = torch.log(torch.sum(torch.exp(temp_res), dim=1, keepdim=True))
            loss = torch.mean(loss)
        else:
            query = output[:, 0]
            answer = output[:, 1:]
            cos_sim = self.sim(query.unsqueeze(1), answer)
            # 第一列是pos
            pos = cos_sim[:, 0].reshape(-1, 1)
            temp_res = self.lambda_ * (cos_sim - pos)
            # temp_res = 20 * (cos_sim - pos)
            loss = torch.log(torch.sum(torch.exp(temp_res), dim=1, keepdim=True))
            loss = torch.mean(loss)

        return loss

class SimCSELoss(nn.Module):
    def __init__(self, temperature=0.05, hard_negative=False):
        super(SimCSELoss, self).__init__()
        self.sim = Similarity()
        self.hard_negative = hard_negative
        self.temperature = temperature

    def forward(self, output):
        if not self.hard_negative:
            z1, z2 = output[:, 0], output[:, 1]
            cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
            labels = torch.arange(cos_sim.size(0)).long().to(z1.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(cos_sim / self.temperature, labels)
        else:
            query = output[:, 0]
            answer = output[:, 1:]
            cos_sim = self.sim(query.unsqueeze(1), answer)
            loss_fct = nn.CrossEntropyLoss()
            labels = torch.zeros(cos_sim.size(0), dtype=torch.long, device=cos_sim.device)
            loss = loss_fct(cos_sim, labels)

        return loss


class MyMPNetModel(MPNetModel):
    def save_pretrained(self, save_directory, state_dict=None):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.

        Arguments:
            save_directory (:obj:`str` or :obj:`os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
        """
        if os.path.isfile(save_directory):
            logger.error("Provided path ({}) should be a directory, not a file".format(save_directory))
            return
        os.makedirs(save_directory, exist_ok=True)

        # Only save the model itself if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self

        # Attach architecture to the config
        model_to_save.config.architectures = [model_to_save.__class__.__name__]

        # state_dict = model_to_save.state_dict()

        # Handle the case where some state_dict keys shouldn't be saved
        if self._keys_to_ignore_on_save is not None:
            state_dict = {k: v for k, v in state_dict.items() if k not in self._keys_to_ignore_on_save}

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)

        if getattr(self.config, "xla_device", False) and is_torch_tpu_available():
            import torch_xla.core.xla_model as xm

            if xm.is_master_ordinal():
                # Save configuration file
                model_to_save.config.save_pretrained(save_directory)
            # xm.save takes care of saving only from master
            xm.save(state_dict, output_model_file)
        else:
            model_to_save.config.save_pretrained(save_directory)
            torch.save(state_dict, output_model_file)

        logger.info("Model weights saved in {}".format(output_model_file))

class MPNetForCL(MPNetPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        if self.model_args.model_type == "mpnet":
            self.mpnet = MyMPNetModel(config, add_pooling_layer=False)
        else:
            raise NotImplementedError

        self.pool_type = self.model_args.pool_type
        self.pooler = Pooler(self.model_args.pool_type)
        if self.model_args.pool_type == "cls":
            self.mlp = MLPLayer(config)
        self.init_weights()

        self.sparse_linear = nn.Linear(in_features=self.mpnet.config.hidden_size, out_features=1)
        # self.loss_fct = AdaptiveuBCLLoss(self.model_args.initial_lambda, self.model_args.lambda_shape)
        # self.loss_fct_sparse = AdaptiveuBCLLoss(self.model_args.initial_lambda, self.model_args.lambda_shape)
        if self.model_args.loss_type == "adaptive":
            self.loss_fct = AdaptiveuBCLLoss(self.model_args.initial_lambda, self.model_args.lambda_shape, self.model_args.hard_negative)
            if self.model_args.use_sparse:
                self.loss_fct_sparse = AdaptiveuBCLLoss(self.model_args.initial_lambda, self.model_args.lambda_shape, self.model_args.hard_negative)
        elif self.model_args.loss_type == "simcse":
            self.loss_fct = SimCSELoss(self.model_args.temperature, self.model_args.hard_negative)
            if self.model_args.use_sparse:
                self.loss_fct_sparse = SimCSELoss(self.model_args.temperature, self.model_args.hard_negative)
        else:
            raise NotImplementedError
        self.vocab_size = self.mpnet.config.vocab_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_args.model_name_or_path)
        if os.path.exists(os.path.join(self.model_args.model_name_or_path, 'sparse_linear.pt')):
            logger.info('loading existing sparse_linear---------')
            self.load_pooler(model_dir=self.model_args.model_name_or_path)
        else:
            logger.info(
                'The parameters of sparse linear is new initialize. Make sure the model is loaded for training, not inferencing')


    def sparse_embedding(self, hidden_state, input_ids, return_embedding: bool = True):
        token_weights = torch.relu(self.sparse_linear(hidden_state))
        if not return_embedding: return token_weights

        sparse_embedding = torch.zeros(input_ids.size(0), input_ids.size(1), self.vocab_size,
                                       dtype=token_weights.dtype,
                                       device=token_weights.device)
        sparse_embedding = torch.scatter(sparse_embedding, dim=-1, index=input_ids.unsqueeze(-1), src=token_weights)

        unused_tokens = [self.tokenizer.cls_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id,
                         self.tokenizer.unk_token_id]
        sparse_embedding = torch.max(sparse_embedding, dim=1).values
        sparse_embedding[:, unused_tokens] *= 0.
        return sparse_embedding

    def encode(self, features):
        if features is None:
            return None
        batch_size = features["input_ids"].size(0)
        num_sent = features["input_ids"].size(1)
        features["input_ids"] = features["input_ids"].view((-1, features["input_ids"].size(-1)))
        features["attention_mask"] = features["attention_mask"].view((-1, features["attention_mask"].size(-1)))
        if "token_type_ids" in features:
            features["token_type_ids"] = features["token_type_ids"].view((-1, features["token_type_ids"].size(-1)))
        outputs = self.mpnet(**features)
        pooler_output = self.pooler(features["attention_mask"], outputs)
        pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1)))  # [batch_size, num_sent, hidden_size] cls or avg 之后会少一个维度
        if self.pool_type == "cls":
            pooler_output = self.mlp(pooler_output)
        # s1, s2 = pooler_output[:, 0], pooler_output[:, 1]
        if self.model_args.use_sparse:
            sparse_vecs = self.sparse_embedding(outputs.last_hidden_state, features['input_ids'])
            sparse_vecs = sparse_vecs.view((batch_size, num_sent, sparse_vecs.size(-1)))  # [batch_size, num_sent, vocab_size]
            # sparse_s1 = sparse_vecs[:, 0]
            # sparse_s2 = sparse_vecs[:, 1]

        else:
            sparse_vecs = None
        return pooler_output, sparse_vecs, outputs


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
            return_dict=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        pooler_output, sparse_vecs, outputs= self.encode(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "position_ids": position_ids,
                "head_mask": head_mask,
                "inputs_embeds": inputs_embeds,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict
            }
        )
        loss = self.loss_fct(pooler_output)
        if self.model_args.use_sparse:
            loss_sparse = self.loss_fct_sparse(sparse_vecs)
            loss = (loss + 0.1 * loss_sparse) / 2   # change the weight
        if not return_dict:
            return loss
        return CLModelOutput(
            loss=loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def save(self, output_dir: str):
        def _trans_state_dict(state_dict):
            state_dict = type(state_dict)(
                {k: v.clone().cpu()
                 for k,
                 v in state_dict.items()})
            return state_dict

        self.mpnet.save_pretrained(output_dir, state_dict=_trans_state_dict(self.mpnet.state_dict()))
        if self.model_args.use_sparse:
            torch.save(_trans_state_dict(self.sparse_linear.state_dict()), os.path.join(output_dir, 'sparse_linear.pt'))



    def load_pooler(self, model_dir):
        sparse_state_dict = torch.load(os.path.join(model_dir, 'sparse_linear.pt'), map_location='cpu')
        self.sparse_linear.load_state_dict(sparse_state_dict)

    # def save(self, output_dir: str):
    #     # self.model.save_pretrained(output_dir)
    #     state_dict = self.model.state_dict()
    #     state_dict = type(state_dict)(
    #         {k: v.clone().cpu()
    #          for k,
    #          v in state_dict.items()})
    #     self.model.save_pretrained(output_dir, state_dict=state_dict)

    # def save_pretrained(self, **kwargs):
    #     self.tokenizer.save_pretrained(**kwargs)
    #     return self.model.save_pretrained(**kwargs)