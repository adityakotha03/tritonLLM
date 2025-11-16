import logging
import time
import os
import json
from datetime import datetime

import dotenv
import pandas as pd
import tinker
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.utils import ml_log

dotenv.load_dotenv()

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)

# Configuration
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
LORA_RANK = 32
SAVE_EVERY = 50
MAX_LENGTH = 16384
TRAIN_ON_WHAT = renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES

# Paths and logging
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "kernelbook-training")
date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
LOG_PATH = f"/tmp/tinker-examples/kernelbook/{date_and_time}"
WANDB_NAME = f"kernelbook-qwen3-4b-{date_and_time}"


def main():
    """Main training function."""
    logger.info("Starting Kernelbook training script")

    wandb_key = os.getenv("WANDB_API_KEY")
    print(f"DEBUG: WANDB_API_KEY {'is SET' if wandb_key else 'is NOT SET'}")
    print(f"DEBUG: WANDB_PROJECT = {WANDB_PROJECT}")
    
    try:
        import wandb
        print(f"DEBUG: wandb package is installed (version: {wandb.__version__})")
    except ImportError:
        print("DEBUG: wandb package is NOT installed!!")
    
    # Setup logging
    ml_logger = ml_log.setup_logging(
        log_dir=LOG_PATH,
        wandb_project=WANDB_PROJECT,
        wandb_name=WANDB_NAME,
        config={
            "base_model": BASE_MODEL,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "lora_rank": LORA_RANK,
            "max_length": MAX_LENGTH,
            "train_on_what": TRAIN_ON_WHAT,
        },
        do_configure_logging_module=True,
    )
    
    # Get tokenizer and renderer
    logger.info(f"Loading model: {BASE_MODEL}")
    service_client = tinker.ServiceClient(api_key=os.getenv("TINKER_API_KEY"))
    training_client = service_client.create_lora_training_client(
        base_model=BASE_MODEL, rank=LORA_RANK
    )
    tokenizer = training_client.get_tokenizer()
    
    renderer_name = model_info.get_recommended_renderer_name(BASE_MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")
    
    # Load dataset
    logger.info("Loading kernelbook dataset...")
    dataset = pd.read_csv("kernelbook.csv")
    dataset['messages'] = dataset['messages'].apply(json.loads)
    logger.info(f"Loaded {len(dataset)} examples")
    
    n_train_batches = len(dataset) // BATCH_SIZE
    logger.info(f"Training for {n_train_batches} batches (full epoch)")
    
    # Save first example for later testing
    first_example = dataset['messages'].iloc[0]
    

    logger.info("Starting training loop...")
    start_time = time.time()
    
    for batch_idx in range(n_train_batches):
        batch_start_time = time.time()
        step = batch_idx
        
        # Save checkpoint periodically
        if step % SAVE_EVERY == 0 and step > 0:
            logger.info(f"Saving checkpoint at step {step}")
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"step_{step:06d}",
                log_path=LOG_PATH,
                kind="state",
                loop_state={"batch": batch_idx},
            )
        
        # Linear learning rate schedule (decay to 0)
        lr_mult = max(0.0, 1.0 - step / n_train_batches)
        current_lr = LEARNING_RATE * lr_mult
        adam_params = tinker.AdamParams(
            learning_rate=current_lr,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8
        )
        
        # Get training batch
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min((batch_idx + 1) * BATCH_SIZE, len(dataset))
        batch_rows = dataset.iloc[batch_start:batch_end]
        
        # Convert to datums
        batch = [
            conversation_to_datum(
                row['messages'],
                renderer,
                MAX_LENGTH,
                TRAIN_ON_WHAT,
            )
            for _, row in batch_rows.iterrows()
        ]
        
        # Training step
        fwd_bwd_future = training_client.forward_backward(batch, loss_fn="cross_entropy")
        optim_step_future = training_client.optim_step(adam_params)
        
        fwd_bwd_result = fwd_bwd_future.result()
        _optim_result = optim_step_future.result()
        
        # Compute metrics
        train_logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
        train_weights = [d.loss_fn_inputs["weights"] for d in batch]
        train_nll = compute_mean_nll(train_logprobs, train_weights)
        
        # Calculate stats
        num_tokens = sum(d.model_input.length for d in batch)
        batch_time = time.time() - batch_start_time
        tokens_per_sec = num_tokens / batch_time if batch_time > 0 else 0
        
        # Log metrics
        metrics = {
            "train_mean_nll": train_nll,
            "learning_rate": current_lr,
            "num_sequences": len(batch),
            "num_tokens": num_tokens,
            "tokens_per_sec": tokens_per_sec,
            "progress": step / n_train_batches,
            "time_per_batch": batch_time,
            "time_elapsed": time.time() - start_time,
        }
        ml_logger.log_metrics(metrics=metrics, step=step)
        
        # Print progress
        if step % 10 == 0:
            logger.info(
                f"Step {step}/{n_train_batches} | "
                f"Loss: {train_nll:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Tokens/sec: {tokens_per_sec:.0f} | "
                f"Progress: {100 * step / n_train_batches:.1f}%"
            )
    
    # Save final checkpoint
    logger.info("Saving final checkpoint...")
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=LOG_PATH,
        kind="both",
        loop_state={"batch": n_train_batches},
    )
    
    logger.info("Training completed!")
    total_time = time.time() - start_time
    logger.info(f"Total training time: {total_time / 60:.2f} minutes")
    
    logger.info("\n" + "="*80)
    logger.info("Generating sample output with trained model...")
    logger.info("="*80)
    
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name='kernelbook-model-Qwen3-4B-Instruct-2507'
    )
    
    # Use first example from dataset as test prompt
    # Remove the assistant message to get just system + user
    test_messages = [msg for msg in first_example if msg['role'] != 'assistant']
    
    logger.info("\nTest Input:")
    for msg in test_messages:
        logger.info(f"[{msg['role'].upper()}]: {msg['content'][:200]}...")
    
    # Build prompt for generation
    prompt = renderer.build_generation_prompt(test_messages, role="assistant")
    
    # Sample from the model
    params = tinker.SamplingParams(
        max_tokens=4096,
        temperature=0.0,  # Greedy sampling
        stop=renderer.get_stop_sequences()
    )
    
    future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1)
    result = future.result()
    
    logger.info("\nModel Output:")
    generated_text = tokenizer.decode(result.sequences[0].tokens)
    logger.info(generated_text)
    
    logger.info("\n" + "="*80)
    logger.info("Sample generation complete!")
    logger.info("="*80)
    

    ml_logger.close()
    logger.info(f"\nTraining logs saved to: {LOG_PATH}")
    if wandb_link := ml_logger.get_logger_url():
        logger.info(f"View results at: {wandb_link}")


if __name__ == "__main__":
    main()