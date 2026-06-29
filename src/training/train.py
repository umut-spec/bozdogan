"""
QLoRA ile model fine-tuning
Vast.ai H200 NVL için optimize edilmiş
"""

import os
import sys
import yaml
import torch
from pathlib import Path

# A100'e ozel: tf32 matmul (~1.3x hiz, bedava, kaliteyi etkilemez)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


class FineTuner:
    def __init__(self, config_path: str = "configs/training_config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.model_name = self.config['model_name']
        self.output_dir = self.config['output_dir']

        print("=" * 60)
        print(f"MODEL: {self.model_name}")
        print(f"OUTPUT: {self.output_dir}")
        print("=" * 60)

    def load_model_and_tokenizer(self):
        """Model ve tokenizer yükle (4-bit QLoRA veya bf16 LoRA)"""

        print("\n📦 Model ve tokenizer yükleniyor...")

        use_4bit = self.config['quantization'].get('load_in_4bit', True)

        # Quantization config (sadece 4-bit istenirse)
        bnb_config = None
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.config['quantization']['bnb_4bit_compute_dtype']),
                bnb_4bit_use_double_quant=self.config['quantization']['bnb_4bit_use_double_quant'],
                bnb_4bit_quant_type=self.config['quantization']['bnb_4bit_quant_type']
            )

        # Model yükle. 80GB GPU'da bf16 LoRA (sikistirmasiz) cok daha hizli:
        # 4-bit dequant yuku yok. attn_implementation config'den okunur.
        attn_impl = self.config['training'].get('attn_implementation', 'sdpa')
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation=attn_impl
        )
        print(f"  Quantization: {'4-bit QLoRA' if use_4bit else 'bf16 LoRA (sikistirmasiz)'}")
        print(f"  Attention: {attn_impl}")

        # Tokenizer yükle
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Pad token ayarla
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model.config.pad_token_id = tokenizer.pad_token_id

        # Model'i k-bit training için hazırla (sadece 4-bit'te gerekli).
        # NOT: prepare_model_for_kbit_training varsayilan olarak gradient checkpointing'i
        # ACAR ve config'i ezer. Config'deki ayara uy.
        use_gc = self.config['training'].get('gradient_checkpointing', True)
        if use_4bit:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=use_gc
            )
        elif use_gc:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()

        print("✓ Model ve tokenizer yüklendi")
        print(f"  Device: {next(model.parameters()).device}")
        print(f"  Dtype: {next(model.parameters()).dtype}")

        return model, tokenizer

    def setup_lora(self, model):
        """LoRA konfigürasyonu"""

        print("\n🔧 LoRA ayarlanıyor...")

        lora_config = LoraConfig(
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['lora_alpha'],
            lora_dropout=self.config['lora']['lora_dropout'],
            target_modules=self.config['lora']['target_modules'],
            bias=self.config['lora']['bias'],
            task_type=self.config['lora']['task_type']
        )

        model = get_peft_model(model, lora_config)

        # Trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())

        print("✓ LoRA uygulandı")
        print(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")
        print(f"  All params: {all_params:,}")

        return model

    def load_datasets(self):
        """Veri setlerini yükle"""

        print("\n📚 Veri setleri yükleniyor...")

        data_files = {
            "train": self.config['data']['train_file'],
            "validation": self.config['data']['val_file']
        }

        dataset = load_dataset("json", data_files=data_files)

        print("✓ Veri setleri yüklendi")
        print(f"  Train: {len(dataset['train'])} örnekler")
        print(f"  Val: {len(dataset['validation'])} örnekler")

        return dataset

    def train(self):
        """Training pipeline"""

        # 1. Model ve tokenizer
        model, tokenizer = self.load_model_and_tokenizer()

        # 2. LoRA setup
        model = self.setup_lora(model)

        # 3. Dataset yükle
        dataset = self.load_datasets()

        # 4. Training arguments (SFTConfig - yeni TRL API)
        training_args = SFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=self.config['training']['num_train_epochs'],
            per_device_train_batch_size=self.config['training']['per_device_train_batch_size'],
            per_device_eval_batch_size=self.config['training']['per_device_eval_batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            learning_rate=self.config['training']['learning_rate'],
            lr_scheduler_type=self.config['training']['lr_scheduler_type'],
            warmup_ratio=self.config['training']['warmup_ratio'],
            weight_decay=self.config['training']['weight_decay'],
            max_grad_norm=self.config['training']['max_grad_norm'],
            logging_steps=self.config['training']['logging_steps'],
            save_strategy=self.config['training']['save_strategy'],
            save_steps=self.config['training']['save_steps'],
            save_total_limit=self.config['training']['save_total_limit'],
            eval_strategy=self.config['training']['evaluation_strategy'],
            eval_steps=self.config['training']['eval_steps'],
            optim=self.config['training']['optim'],
            gradient_checkpointing=self.config['training']['gradient_checkpointing'],
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_num_workers=self.config['training'].get('dataloader_num_workers', 4),
            dataloader_pin_memory=True,
            tf32=True,
            fp16=self.config['training']['fp16'],
            bf16=self.config['training']['bf16'],
            ddp_find_unused_parameters=self.config['training']['ddp_find_unused_parameters'],
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            # SFT-specific (yeni TRL API'de buraya taşındı)
            max_length=self.config['training']['max_seq_length'],
            packing=self.config['training'].get('packing', False),
        )

        # 5. Trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
        )

        print("\n🚀 Training başlıyor...\n")

        # 6. Train!
        trainer.train()

        print("\n✅ Training tamamlandı!")

        # 7. Model kaydet
        print(f"\n💾 Model kaydediliyor: {self.output_dir}")
        trainer.save_model(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)

        print("✓ Model kaydedildi")

        return trainer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    args = parser.parse_args()

    finetuner = FineTuner(config_path=args.config)
    trainer = finetuner.train()
