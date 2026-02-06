#!/usr/bin/env python3
"""
CLI tool to run the enhanced agentic food aggregation workflow (V2).

New features:
- Rate limiting for free tier models
- Checkpoint/resume functionality
- Graceful interruption handling

Usage:
    # FREE tier (Gemini Flash)
    python aggregate_foods_v2.py --sample 1000 --model google/gemini-flash-1.5-8b --rate-limit 15 --use-llm

    # Paid tier (fast)
    python aggregate_foods_v2.py --sample 1000 --model deepseek/deepseek-r1-distill-qwen-32b --rate-limit 60 --use-llm

    # Resume from checkpoint
    python aggregate_foods_v2.py --resume checkpoints/checkpoint_iter_5.pkl
"""

import argparse
from pyfooda.agentic_aggregator_v2 import FoodAggregationAgentV2
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate similar food items using enhanced agentic workflow (V2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # FREE tier (patient, but $0 cost)
  python aggregate_foods_v2.py --sample 1000 --model google/gemini-flash-1.5-8b --rate-limit 15 --use-llm

  # BEST VALUE (fast and cheap)
  python aggregate_foods_v2.py --sample 1000 --model deepseek/deepseek-r1-distill-qwen-32b --rate-limit 60 --use-llm

  # Resume from checkpoint
  python aggregate_foods_v2.py --resume checkpoints/checkpoint_iter_5.pkl --use-llm

  # Full database (FREE, will take days)
  python aggregate_foods_v2.py --model google/gemini-flash-1.5-8b --rate-limit 15 --use-llm
        """
    )
    
    parser.add_argument(
        '--input',
        default='pyfooda/data/foods.csv',
        help='Input CSV file with food data'
    )
    
    parser.add_argument(
        '--output',
        default='pyfooda/data/foods_aggregated.csv',
        help='Output CSV file for aggregated data'
    )
    
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Number of items to sample (default: use all)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.85,
        help='Similarity threshold for merge candidates (default: 0.85)'
    )
    
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=10,
        help='Maximum iterations before stopping (default: 10)'
    )
    
    parser.add_argument(
        '--model',
        default='google/gemini-flash-1.5-8b',
        help='LLM model to use (default: google/gemini-flash-1.5-8b - FREE)'
    )
    
    parser.add_argument(
        '--rate-limit',
        type=int,
        default=15,
        help='Max API calls per minute (default: 15 for free tier, 0 = no limit)'
    )
    
    parser.add_argument(
        '--checkpoint-every',
        type=int,
        default=10,
        help='Save checkpoint every N merges (default: 10)'
    )
    
    parser.add_argument(
        '--checkpoint-dir',
        default='./checkpoints',
        help='Directory for checkpoint files (default: ./checkpoints)'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint file (e.g., checkpoints/checkpoint_iter_5.pkl)'
    )
    
    parser.add_argument(
        '--use-llm',
        action='store_true',
        default=False,
        help='Use LLM for merge validation (requires OPENROUTER_API_KEY)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Enhanced Agentic Food Aggregation Workflow (V2)")
    print("="*60)
    
    # Resume from checkpoint
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        agent = FoodAggregationAgentV2.load_checkpoint(
            args.resume,
            llm_model=args.model,
            rate_limit=args.rate_limit,
            checkpoint_dir=args.checkpoint_dir
        )
        
        # Continue workflow
        aggregated = agent.run_until_convergence(
            max_iterations=args.max_iterations,
            similarity_threshold=args.threshold,
            use_llm_validation=args.use_llm,
            checkpoint_every=args.checkpoint_every
        )
    
    else:
        # Fresh start
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
        print(f"Sample size: {args.sample or 'ALL'}")
        print(f"Model: {args.model}")
        print(f"Rate limit: {args.rate_limit} calls/min")
        print(f"Similarity threshold: {args.threshold}")
        print(f"Max iterations: {args.max_iterations}")
        print(f"LLM validation: {'YES' if args.use_llm else 'NO'}")
        print(f"Checkpoint dir: {args.checkpoint_dir}")
        print(f"Checkpoint frequency: Every {args.checkpoint_every} merges")
        print("="*60)
        
        # Load data
        df = pd.read_csv(args.input)
        
        if args.sample:
            df = df.sample(min(args.sample, len(df)), random_state=42)
            print(f"Sampled {len(df)} items")
        
        # Initialize agent
        agent = FoodAggregationAgentV2(
            df,
            llm_model=args.model,
            rate_limit=args.rate_limit,
            checkpoint_dir=args.checkpoint_dir
        )
        
        # Run workflow
        aggregated = agent.run_until_convergence(
            max_iterations=args.max_iterations,
            similarity_threshold=args.threshold,
            use_llm_validation=args.use_llm,
            checkpoint_every=args.checkpoint_every
        )
    
    # Save results
    agent.save_results(
        output_path=args.output,
        merge_history_path=args.output.replace('.csv', '_history.json')
    )
    
    print("\n✓ Workflow complete!")
    print(f"Checkpoints saved in: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
