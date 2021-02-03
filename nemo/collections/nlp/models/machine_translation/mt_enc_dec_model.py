# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import itertools
import json
import os
import pickle
import random
from pathlib import Path
import tarfile
import tempfile
from typing import Dict, List, Optional
from importlib_metadata import metadata

import numpy as np
import torch
import torch.utils.data as pt_data
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_only
from sacrebleu import corpus_bleu
from sacremoses import MosesDetokenizer, MosesPunctNormalizer, MosesTokenizer
import youtokentome as yttm

from nemo.collections.common.losses import SmoothedCrossEntropyLoss
from nemo.collections.common.metrics import GlobalAverageLossMetric
from nemo.collections.common.parts import transformer_weights_init
from nemo.collections.nlp.data import TarredTranslationDataset, TranslationDataset
from nemo.collections.nlp.models.enc_dec_nlp_model import EncDecNLPModel
from nemo.collections.nlp.models.machine_translation.mt_enc_dec_config import MTEncDecModelConfig
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.collections.nlp.modules.common.transformer import BeamSearchSequenceGenerator
from nemo.collections.nlp.modules.common.transformer.transformer import TransformerDecoderNM, TransformerEncoderNM
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.core.classes.common import typecheck
from nemo.utils import logging, model_utils


__all__ = ['MTEncDecModel']


class MTEncDecModel(EncDecNLPModel):
    """
    Encoder-decoder machine translation model.
    """

    def __init__(self, cfg: MTEncDecModelConfig, trainer: Trainer = None):
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        self.global_rank = 0
        self.world_size = 1
        if trainer is not None:
            self.global_rank = (trainer.node_rank * trainer.num_gpus) + trainer.local_rank
            self.world_size = trainer.num_nodes * trainer.num_gpus

        cfg = model_utils.maybe_update_config_version(cfg)

        # Train tokenizers if they don't exist
        if (
            cfg.encoder_tokenizer.get('tokenizer_model') is None
            or cfg.decoder_tokenizer.get('tokenizer_model') is None
        ):
            # train tokenizer model on training data
            encoder_tokenizer_model, decoder_tokenizer_model = self.train_tokenizers(
                out_dir=cfg.preproc_out_dir,
                src_fname=cfg.train_ds.src_file_name,
                tgt_fname=cfg.train_ds.tgt_file_name,
                shared_tokenizer=cfg.shared_tokenizer,
                encoder_tokenizer_vocab_size=cfg.encoder_tokenizer.vocab_size,
                decoder_tokenizer_vocab_size=cfg.decoder_tokenizer.vocab_size,
                encoder_tokenizer_name=cfg.encoder_tokenizer.tokenizer_name,
                decoder_tokenizer_name=cfg.decoder_tokenizer.tokenizer_name,
            )
        else:
            encoder_tokenizer_model = cfg.encoder_tokenizer.tokenizer_model
            decoder_tokenizer_model = cfg.decoder_tokenizer.tokenizer_model

        # Instaniate tokenizers and register to be saved with NeMo Model archive
        self.setup_enc_dec_tokenizers(
            encoder_tokenizer_name=cfg.encoder_tokenizer.tokenizer_name,
            encoder_tokenizer_model=encoder_tokenizer_model,
            encoder_bpe_dropout=cfg.encoder_tokenizer.get('bpe_dropout', 0.0),
            decoder_tokenizer_name=cfg.decoder_tokenizer.tokenizer_name,
            decoder_tokenizer_model=decoder_tokenizer_model,
            decoder_bpe_dropout=cfg.decoder_tokenizer.get('bpe_dropout', 0.0),
        )

        # If using tarred dataset for training, automatically create it if needed
        if hasattr(cfg, 'train_ds'):
            if cfg.train_ds.get('use_tarred_dataset'):
                if cfg.train_ds.get('tar_file_names') is None or cfg.train_ds.get('metadata_file_name') is None:
                    # Preprocess data and cache for use during training
                    logging.info(
                        f"Creating tarred dataset for src {cfg.train_ds.get('src_file_name')} and tgt {cfg.train_ds.get('tgt_file_name')}"
                    )
                    self.train_tar_files, self.train_metadata_file = self.preprocess_dataset(
                        clean=cfg.train_ds.clean,
                        src_fname=cfg.train_ds.get('src_file_name'),
                        tgt_fname=cfg.train_ds.get('tgt_file_name'),
                        out_dir=cfg.get('preproc_out_dir'),
                        encoder_tokenizer=self.encoder_tokenizer,
                        decoder_tokenizer=self.decoder_tokenizer,
                        max_seq_length=cfg.train_ds.max_seq_length,
                        tokens_in_batch=cfg.train_ds.tokens_in_batch,
                        lines_per_dataset_fragment=cfg.train_ds.get('lines_per_dataset_fragment'),
                        num_batches_per_tarfile=cfg.train_ds.get('num_batches_per_tarfile'),
                        min_seq_length=1,
                    )
                    logging.info(
                        f"Tarred dataset created at {self.train_tar_files} and metadata created at {self.train_metadata_file}"
                    )
                else:
                    self.train_tar_files = cfg.train_ds.get('tar_file_names')
                    self.train_metadata_file = cfg.train_ds.get('metadata_file_name')

        super().__init__(cfg=cfg, trainer=trainer)

        # TODO: use get_encoder function with support for HF and Megatron
        self.encoder = TransformerEncoderNM(
            vocab_size=self.encoder_vocab_size,
            hidden_size=cfg.encoder.hidden_size,
            num_layers=cfg.encoder.num_layers,
            inner_size=cfg.encoder.inner_size,
            max_sequence_length=cfg.encoder.max_sequence_length
            if hasattr(cfg.encoder, 'max_sequence_length')
            else 512,
            embedding_dropout=cfg.encoder.embedding_dropout if hasattr(cfg.encoder, 'embedding_dropout') else 0.0,
            learn_positional_encodings=cfg.encoder.learn_positional_encodings
            if hasattr(cfg.encoder, 'learn_positional_encodings')
            else False,
            num_attention_heads=cfg.encoder.num_attention_heads,
            ffn_dropout=cfg.encoder.ffn_dropout,
            attn_score_dropout=cfg.encoder.attn_score_dropout,
            attn_layer_dropout=cfg.encoder.attn_layer_dropout,
            hidden_act=cfg.encoder.hidden_act,
            mask_future=cfg.encoder.mask_future,
            pre_ln=cfg.encoder.pre_ln,
        )

        # TODO: user get_decoder function with support for HF and Megatron
        self.decoder = TransformerDecoderNM(
            vocab_size=self.decoder_vocab_size,
            hidden_size=cfg.decoder.hidden_size,
            num_layers=cfg.decoder.num_layers,
            inner_size=cfg.decoder.inner_size,
            max_sequence_length=cfg.decoder.max_sequence_length
            if hasattr(cfg.decoder, 'max_sequence_length')
            else 512,
            embedding_dropout=cfg.decoder.embedding_dropout if hasattr(cfg.decoder, 'embedding_dropout') else 0.0,
            learn_positional_encodings=cfg.decoder.learn_positional_encodings
            if hasattr(cfg.decoder, 'learn_positional_encodings')
            else False,
            num_attention_heads=cfg.decoder.num_attention_heads,
            ffn_dropout=cfg.decoder.ffn_dropout,
            attn_score_dropout=cfg.decoder.attn_score_dropout,
            attn_layer_dropout=cfg.decoder.attn_layer_dropout,
            hidden_act=cfg.decoder.hidden_act,
            pre_ln=cfg.decoder.pre_ln,
        )

        self.log_softmax = TokenClassifier(
            hidden_size=self.decoder.hidden_size,
            num_classes=self.decoder_vocab_size,
            activation=cfg.head.activation,
            log_softmax=cfg.head.log_softmax,
            dropout=cfg.head.dropout,
            use_transformer_init=cfg.head.use_transformer_init,
        )

        self.beam_search = BeamSearchSequenceGenerator(
            embedding=self.decoder.embedding,
            decoder=self.decoder.decoder,
            log_softmax=self.log_softmax,
            max_sequence_length=self.decoder.max_sequence_length,
            beam_size=cfg.beam_size,
            bos=self.decoder_tokenizer.bos_id,
            pad=self.decoder_tokenizer.pad_id,
            eos=self.decoder_tokenizer.eos_id,
            len_pen=cfg.len_pen,
            max_delta_length=cfg.max_generation_delta,
        )

        # tie weights of embedding and softmax matrices
        self.log_softmax.mlp.layer0.weight = self.decoder.embedding.token_embedding.weight

        # TODO: encoder and decoder with different hidden size?
        std_init_range = 1 / self.encoder.hidden_size ** 0.5
        self.apply(lambda module: transformer_weights_init(module, std_init_range))

        self.loss_fn = SmoothedCrossEntropyLoss(
            pad_id=self.decoder_tokenizer.pad_id, label_smoothing=cfg.label_smoothing
        )
        self.eval_loss = GlobalAverageLossMetric(dist_sync_on_step=False, take_avg_loss=True)

    def filter_predicted_ids(self, ids):
        ids[ids >= self.decoder_tokenizer.vocab_size] = self.decoder_tokenizer.unk_id
        return ids

    @typecheck()
    def forward(self, src, src_mask, tgt, tgt_mask):
        src_hiddens = self.encoder(src, src_mask)
        if tgt is not None:
            tgt_hiddens = self.decoder(tgt, tgt_mask, src_hiddens, src_mask)
            log_probs = self.log_softmax(hidden_states=tgt_hiddens)
        else:
            log_probs = None
        beam_results = None
        if not self.training:
            beam_results = self.beam_search(encoder_hidden_states=src_hiddens, encoder_input_mask=src_mask)
            beam_results = self.filter_predicted_ids(beam_results)
        return log_probs, beam_results

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        for i in range(len(batch)):
            if batch[i].ndim == 3:
                # Dataset returns already batched data and the first dimension of size 1 added by DataLoader
                # is excess.
                batch[i] = batch[i].squeeze(dim=0)
        src_ids, src_mask, tgt_ids, tgt_mask, labels = batch
        log_probs, _ = self(src_ids, src_mask, tgt_ids, tgt_mask)
        train_loss = self.loss_fn(log_probs=log_probs, labels=labels)
        # training_perplexity = self.training_perplexity(logits=log_probs)
        tensorboard_logs = {
            'train_loss': train_loss,
            'lr': self._optimizer.param_groups[0]['lr'],
            # "train_ppl": training_perplexity,
        }
        return {'loss': train_loss, 'log': tensorboard_logs}

    def eval_step(self, batch, batch_idx, mode):
        for i in range(len(batch)):
            if batch[i].ndim == 3:
                # Dataset returns already batched data and the first dimension of size 1 added by DataLoader
                # is excess.
                batch[i] = batch[i].squeeze(dim=0)
        src_ids, src_mask, tgt_ids, tgt_mask, labels = batch
        log_probs, beam_results = self(src_ids, src_mask, tgt_ids, tgt_mask)
        eval_loss = self.loss_fn(log_probs=log_probs, labels=labels)
        self.eval_loss(loss=eval_loss, num_measurements=log_probs.shape[0] * log_probs.shape[1])
        translations = [self.decoder_tokenizer.ids_to_text(tr) for tr in beam_results.cpu().numpy()]
        np_tgt = tgt_ids.cpu().numpy()
        ground_truths = [self.decoder_tokenizer.ids_to_text(tgt) for tgt in np_tgt]
        num_non_pad_tokens = np.not_equal(np_tgt, self.decoder_tokenizer.pad_id).sum().item()
        return {
            'translations': translations,
            'ground_truths': ground_truths,
            'num_non_pad_tokens': num_non_pad_tokens,
        }

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'test')

    @rank_zero_only
    def log_param_stats(self):
        for name, p in self.named_parameters():
            if p.requires_grad:
                self.trainer.logger.experiment.add_histogram(name + '_hist', p, global_step=self.global_step)
                self.trainer.logger.experiment.add_scalars(
                    name,
                    {'mean': p.mean(), 'stddev': p.std(), 'max': p.max(), 'min': p.min()},
                    global_step=self.global_step,
                )

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        return self.eval_step(batch, batch_idx, 'val')

    def eval_epoch_end(self, outputs, mode):
        eval_loss = self.eval_loss.compute()
        translations = list(itertools.chain(*[x['translations'] for x in outputs]))
        ground_truths = list(itertools.chain(*[x['ground_truths'] for x in outputs]))

        # TODO: add target language so detokenizer can be lang specific.
        detokenizer = MosesDetokenizer()
        translations = [detokenizer.detokenize(sent.split()) for sent in translations]
        ground_truths = [detokenizer.detokenize(sent.split()) for sent in ground_truths]
        assert len(translations) == len(ground_truths)
        sacre_bleu = corpus_bleu(translations, [ground_truths], tokenize="13a")
        dataset_name = "Validation" if mode == 'val' else "Test"
        logging.info(f"\n\n\n\n{dataset_name} set size: {len(translations)}")
        logging.info(f"{dataset_name} Sacre BLEU = {sacre_bleu.score}")
        logging.info(f"{dataset_name} TRANSLATION EXAMPLES:".upper())
        for i in range(0, 3):
            ind = random.randint(0, len(translations) - 1)
            logging.info("    " + '\u0332'.join(f"EXAMPLE {i}:"))
            logging.info(f"    Prediction:   {translations[ind]}")
            logging.info(f"    Ground Truth: {ground_truths[ind]}")

        ans = {f"{mode}_loss": eval_loss, f"{mode}_sacreBLEU": sacre_bleu.score}
        ans['log'] = dict(ans)
        return ans

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        self.log_dict(self.eval_epoch_end(outputs, 'val'))

    def test_epoch_end(self, outputs):
        return self.eval_epoch_end(outputs, 'test')

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config)

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config)

    def _setup_dataloader_from_config(self, cfg: DictConfig):
        if cfg.get("use_tarred_dataset", False):
            # tarred dataset only used for training data
            logging.info('Loading from tarred dataset %s' % (self.train_tar_file))
            dataset = TarredTranslationDataset(
                text_tar_filepaths=self.train_tar_file,
                metadata_path=self.train_metadata_file,
                encoder_tokenizer=self.encoder_tokenizer,
                decoder_tokenizer=self.decoder_tokenizer,
                shuffle_n=cfg.get("tar_shuffle_n", 100),
                shard_strategy=cfg.get("shard_strategy", "scatter"),
                global_rank=self.global_rank,
                world_size=self.world_size,
                reverse_lang_direction=cfg.get("reverse_lang_direction", False),
            )
            return torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=1,
                num_workers=cfg.get("num_workers", 2),
                pin_memory=cfg.get("pin_memory", False),
                drop_last=cfg.get("drop_last", False),
            )
        else:
            dataset = TranslationDataset(
                dataset_src=str(Path(cfg.src_file_name).expanduser()),
                dataset_tgt=str(Path(cfg.tgt_file_name).expanduser()),
                tokens_in_batch=cfg.tokens_in_batch,
                clean=cfg.get("clean", False),
                max_seq_length=cfg.get("max_seq_length", 512),
                min_seq_length=cfg.get("min_seq_length", 1),
                max_seq_length_diff=cfg.get("max_seq_length_diff", 512),
                max_seq_length_ratio=cfg.get("max_seq_length_ratio", 512),
                cache_ids=cfg.get("cache_ids", False),
                cache_data_per_node=cfg.get("cache_data_per_node", False),
                use_cache=cfg.get("use_cache", False),
                reverse_lang_direction=cfg.get("reverse_lang_direction", False),
            )
            dataset.batchify(self.encoder_tokenizer, self.decoder_tokenizer)
        if cfg.shuffle:
            sampler = pt_data.RandomSampler(dataset)
        else:
            sampler = pt_data.SequentialSampler(dataset)
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=cfg.get("num_workers", 2),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=cfg.get("drop_last", False),
        )

    @torch.no_grad()
    def translate(self, text: List[str], source_lang: str = None, target_lang: str = None) -> List[str]:
        """
        Translates list of sentences from source language to target language.
        Should be regular text, this method performs its own tokenization/de-tokenization
        Args:
            text: list of strings to translate
            source_lang: if not None, corresponding MosesTokenizer and MosesPunctNormalizer will be run
            target_lang: if not None, corresponding MosesDecokenizer will be run
        Returns:
            list of translated strings
        """
        mode = self.training
        if source_lang != "None":
            tokenizer = MosesTokenizer(lang=source_lang)
            normalizer = MosesPunctNormalizer(lang=source_lang)
        if target_lang != "None":
            detokenizer = MosesDetokenizer(lang=target_lang)
        try:
            self.eval()
            res = []
            for txt in text:
                if source_lang != "None":
                    txt = normalizer.normalize(txt)
                    txt = tokenizer.tokenize(txt, escape=False, return_str=True)
                ids = self.encoder_tokenizer.text_to_ids(txt)
                ids = [self.encoder_tokenizer.bos_id] + ids + [self.encoder_tokenizer.eos_id]
                src = torch.Tensor(ids).long().to(self._device).unsqueeze(0)
                src_mask = torch.ones_like(src)
                src_hiddens = self.encoder(input_ids=src, encoder_mask=src_mask)
                beam_results = self.beam_search(encoder_hidden_states=src_hiddens, encoder_input_mask=src_mask)
                beam_results = self.filter_predicted_ids(beam_results)
                translation_ids = beam_results.cpu()[0].numpy()
                translation = self.decoder_tokenizer.ids_to_text(translation_ids)
                if target_lang != "None":
                    translation = detokenizer.detokenize(translation.split())
                res.append(translation)
        finally:
            self.train(mode=mode)
        return res

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass

    # TODO: add local or global rank 0 decorator
    @rank_zero_only
    def preprocess_dataset(
        self,
        clean,
        src_fname,
        tgt_fname,
        out_dir,
        encoder_tokenizer,
        decoder_tokenizer,
        max_seq_length,
        min_seq_length,
        tokens_in_batch,
        lines_per_dataset_fragment,
        num_batches_per_tarfile,
    ):

        os.makedirs(out_dir, exist_ok=True)

        tar_file_ctr = 1
        num_files_in_tar = 0
        num_lines = 0
        shard_num = 0
        global_batch_ctr = 0
        tmp_f_src = tempfile.NamedTemporaryFile(delete=False, mode='w')
        tmp_f_tgt = tempfile.NamedTemporaryFile(delete=False, mode='w')
        tar_file_path = os.path.join(out_dir, 'batches.tokens.%d.%d.tar' % (tokens_in_batch, 1))
        metadata_path = os.path.join(out_dir, f'metadata.tokens.{tokens_in_batch}.json')
        if os.path.isfile(tar_file_path) and os.path.isfile(metadata_path):
            logging.info(
                f'Tarred dataset {tar_file_path} and metadata file {metadata_path} exists and will be used. Remove if reprocessing.'
            )
        else:
            tar_file_ptr = tarfile.open(tar_file_path, 'w')
            with open(src_fname, 'r') as f_src, open(tgt_fname) as f_tgt:
                for src_line, tgt_line in zip(f_src, f_tgt):
                    tmp_f_src.write(src_line)
                    tmp_f_tgt.write(tgt_line)
                    num_lines += 1

                    if num_lines == lines_per_dataset_fragment:
                        tmp_f_src.close()
                        tmp_f_tgt.close()
                        (
                            tar_file_ptr,
                            global_batch_ctr,
                            num_files_in_tar,
                            tar_file_ctr,
                        ) = self.write_batches_to_tarfiles(
                            src_fname=tmp_f_src.name,
                            tgt_fname=tmp_f_tgt.name,
                            num_tokens=tokens_in_batch,
                            encoder_tokenizer=encoder_tokenizer,
                            decoder_tokenizer=decoder_tokenizer,
                            num_files_in_tar=num_files_in_tar,
                            tar_file_ptr=tar_file_ptr,
                            tar_file_ctr=tar_file_ctr,
                            global_batch_ctr=global_batch_ctr,
                        )
                        tar_file_ptrs.append(tar_file_ptr)

                        num_lines = 0
                        shard_num += 1

                        os.remove(tmp_f_src.name)
                        os.remove(tmp_f_tgt.name)

                        tmp_f_src = tempfile.NamedTemporaryFile(delete=False, mode='w')
                        tmp_f_tgt = tempfile.NamedTemporaryFile(delete=False, mode='w')

            tmp_f_src.close()
            tmp_f_tgt.close()
            tar_file_ptr, global_batch_ctr, num_files_in_tar, tar_file_ctr = self.write_batches_to_tarfiles(
                out_dir=out_dir,
                num_batches_per_tarfile=num_batches_per_tarfile,
                clean=clean,
                max_seq_length=max_seq_length,
                min_seq_length=min_seq_length,
                src_fname=tmp_f_src.name,
                tgt_fname=tmp_f_tgt.name,
                num_tokens=tokens_in_batch,
                encoder_tokenizer=encoder_tokenizer,
                decoder_tokenizer=decoder_tokenizer,
                num_files_in_tar=num_files_in_tar,
                tar_file_ptr=tar_file_ptr,
                tar_file_ctr=tar_file_ctr,
                global_batch_ctr=global_batch_ctr,
            )
            tar_file_ptr.close()
            os.remove(tmp_f_src.name)
            os.remove(tmp_f_tgt.name)

            if num_files_in_tar != num_batches_per_tarfile:
                os.remove(os.path.join(out_dir, 'batches.tokens.%d.%d.tar' % (tokens_in_batch, tar_file_ctr)))
                global_batch_ctr -= num_files_in_tar
                print('Dropping %d batches because of overflow' % (num_files_in_tar))

            json.dump({'num_batches': global_batch_ctr}, open(metadata_path, 'w'))
        tar_file_paths = glob.glob(f'{out_dir}/batches.tokens.{tokens_in_batch}.*.tar')
        logging.info(f'tar_file_paths: {tar_file_paths}')
        logging.info(f'metadata_path: {metadata_path}')
        return tar_file_paths, metadata_path

    @rank_zero_only
    def train_tokenizers(
        self,
        out_dir,
        src_fname,
        tgt_fname,
        shared_tokenizer,
        encoder_tokenizer_name,
        encoder_tokenizer_vocab_size,
        decoder_tokenizer_name,
        decoder_tokenizer_vocab_size,
    ):
        # trains tokenizers if needed and
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        if encoder_tokenizer_name != 'yttm' or decoder_tokenizer_name != 'yttm':
            raise NotImplemented(f"Currently we only support yttm tokenizer.")

        if shared_tokenizer:
            encoder_tokenizer_model = os.path.join(
                out_dir, 'shared_tokenizer.%d.BPE.model' % (encoder_tokenizer_vocab_size)
            )
            decoder_tokenizer_model = encoder_tokenizer_model
            if os.path.isfile(encoder_tokenizer_model):
                logging.info(
                    f'Shared tokenizer model {encoder_tokenizer_model} already exists. Remove file if training a new tokenizer model.'
                )
            else:
                logging.info(f'Shared tokenizer model {encoder_tokenizer_model} not found. Training tokenizer model.')
                os.system('cat %s %s > %s' % (src_fname, tgt_fname, '/tmp/concat_dataset.txt'))
                yttm.BPE.train(
                    data='/tmp/concat_dataset.txt',
                    vocab_size=encoder_tokenizer_vocab_size,
                    model=os.path.join(out_dir, encoder_tokenizer_model),
                )
                os.remove('/tmp/concat_dataset.txt')
        else:
            encoder_tokenizer_model = os.path.join(
                out_dir, 'tokenizer.encoder.%d.BPE.model' % (encoder_tokenizer_vocab_size)
            )
            if os.path.isfile(encoder_tokenizer_model):
                logging.info(
                    f'Encoder tokenizer model {encoder_tokenizer_model} already exists. Remove file if training a new tokenizer model.'
                )
            else:
                logging.info(f'Encoder tokenizer model {encoder_tokenizer_model} not found. Training tokenizer model.')
                yttm.BPE.train(data=src_fname, vocab_size=encoder_tokenizer_vocab_size, model=encoder_tokenizer_model)

            decoder_tokenizer_model = os.path.join(
                out_dir, 'tokenizer.decoder.%d.BPE.model' % (decoder_tokenizer_vocab_size)
            )
            if os.path.isfile(decoder_tokenizer_model):
                logging.info(
                    f'Decoder tokenizer model {decoder_tokenizer_model} already exists. Remove file if training a new tokenizer model.'
                )
            else:
                logging.info(f'Decoder tokenizer model {decoder_tokenizer_model} not found. Training tokenizer model.')
                yttm.BPE.train(data=src_fname, vocab_size=decoder_tokenizer_vocab_size, model=decoder_tokenizer_model)

        return encoder_tokenizer_model, decoder_tokenizer_model

    @rank_zero_only
    def write_batches_to_tarfiles(
        self,
        out_dir,
        num_batches_per_tarfile,
        clean,
        max_seq_length,
        min_seq_length,
        src_fname,
        tgt_fname,
        num_tokens,
        encoder_tokenizer,
        decoder_tokenizer,
        num_files_in_tar,
        tar_file_ptr,
        tar_file_ctr,
        global_batch_ctr,
    ):
        """
        Writes current fragment of the overall parallel corpus to tarfiles by:
        (1) Creating a minibatches using a TranslationDataset object.
        (2) Writing each minibatch to a pickle file.
        (3) Adding pickle files to a tarfile until it reaches args.num_batches_per_tarfile.
        """

        dataset = TranslationDataset(
            dataset_src=src_fname,
            dataset_tgt=tgt_fname,
            tokens_in_batch=num_tokens,
            clean=clean,
            max_seq_length=max_seq_length,
            min_seq_length=min_seq_length,
            max_seq_length_diff=max_seq_length,
            max_seq_length_ratio=max_seq_length,
            cache_ids=False,
            cache_data_per_node=False,
            use_cache=False,
        )
        dataset.batchify(encoder_tokenizer, decoder_tokenizer)

        for _, batch in dataset.batches.items():
            global_batch_ctr += 1
            pickle.dump(batch, open(os.path.join(out_dir, 'batch-%d.pkl' % (global_batch_ctr)), 'wb'))

            if num_files_in_tar == num_batches_per_tarfile:
                tar_file_ctr += 1
                tar_file_ptr.close()
                tar_file_ptr = tarfile.open(
                    os.path.join(out_dir, 'batches.tokens.%d.%d.tar' % (num_tokens, tar_file_ctr)), 'w'
                )
                num_files_in_tar = 0

            tar_file_ptr.add(os.path.join(out_dir, 'batch-%d.pkl' % (global_batch_ctr)))
            num_files_in_tar += 1
            os.remove(os.path.join(out_dir, 'batch-%d.pkl' % (global_batch_ctr)))
        return tar_file_ptr, global_batch_ctr, num_files_in_tar, tar_file_ctr
