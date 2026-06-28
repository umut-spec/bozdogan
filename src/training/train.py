"""
QLoRA ile model fine-tuning
Vast.ai H200 NVL için optimize edilmiş
"""

import os
import sys
import yaml
import torch
from pathlib import Path
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
from trl import SFTTrainer


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
        """Model ve tokenizer yükle (4-bit quantization ile)"""

        print("\n📦 Model ve tokenizer yükleniyor...")

        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config['quantization']['load_in_4bit'],
            bnb_4bit_compute_dtype=getattr(torch, self.config['quantization']['bnb_4bit_compute_dtype']),
            bnb_4bit_use_double_quant=self.config['quantization']['bnb_4bit_use_double_quant'],
            bnb_4bit_quant_type=self.config['quantization']['bnb_4bit_quant_type']
        )

        # Model yükle
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )

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

        # Model'i k-bit training için hazırla
        model = prepare_model_for_kbit_training(model)

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

        # 4. Training arguments
        training_args = TrainingArguments(
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
            evaluation_strategy=self.config['training']['evaluation_strategy'],
            eval_steps=self.config['training']['eval_steps'],
            optim=self.config['training']['optim'],
            gradient_checkpointing=self.config['training']['gradient_checkpointing'],
            fp16=self.config['training']['fp16'],
            bf16=self.config['training']['bf16'],
            ddp_find_unused_parameters=self.config['training']['ddp_find_unused_parameters'],
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )

        # 5. Trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=tokenizer,
            dataset_text_field="text",  # JSONL'deki 'text' field'ı
            max_seq_length=self.config['training']['max_seq_length'],
            packing=False,  # Her örnek ayrı işlensin
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
