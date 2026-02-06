#!/usr/bin/env python3
"""
Quick test of the agentic aggregation workflow on a small sample.

This script demonstrates:
1. Loading a sample of food items
2. Running the aggregation workflow
3. Inspecting the results
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyfooda.agentic_aggregator import FoodAggregationAgent
import pandas as pd


def test_small_sample():
    """Test on a tiny sample to verify the workflow works."""
    
    # Create a small test dataset
    test_data = {
        'foodName': [
            'ORGANIC APPLE, RAW',
            'APPLE, FRESH',
            'FUJI APPLE',
            'GRANNY SMITH APPLE',
            'APPLE PIE',
            'APPLE JUICE',
            'ORANGE, RAW',
            'ORANGE, FRESH',
            'ORANGE JUICE',
            'BANANA, RAW',
            'BANANA, FRESH',
        ],
        'food_category': [
            'Fruits', 'Fruits', 'Fruits', 'Fruits', 'Baked Goods', 'Beverages',
            'Fruits', 'Fruits', 'Beverages', 'Fruits', 'Fruits'
        ],
        'Energy': [52, 52, 52, 52, 237, 46, 47, 47, 45, 89, 89],
        'Carbohydrate': [13.8, 13.8, 13.8, 13.8, 34.0, 11.3, 11.7, 11.7, 10.4, 22.8, 22.8],
        'Protein': [0.3, 0.3, 0.3, 0.3, 2.0, 0.1, 0.9, 0.9, 0.7, 1.1, 1.1],
        'Total fat': [0.2, 0.2, 0.2, 0.2, 11.0, 0.1, 0.1, 0.1, 0.2, 0.3, 0.3],
        'Fiber': [2.4, 2.4, 2.4, 2.4, 1.6, 0.2, 2.4, 2.4, 0.2, 2.6, 2.6],
        'Sugars, Total': [10.4, 10.4, 10.4, 10.4, 18.0, 9.6, 9.4, 9.4, 8.4, 12.2, 12.2],
        'Sodium': [1, 1, 1, 1, 300, 5, 0, 0, 1, 1, 1],
        'Calcium': [6, 6, 6, 6, 20, 8, 40, 40, 11, 5, 5],
        'Iron': [0.1, 0.1, 0.1, 0.1, 0.9, 0.2, 0.1, 0.1, 0.4, 0.3, 0.3],
    }
    
    df = pd.DataFrame(test_data)
    
    print("="*60)
    print("Testing Agentic Food Aggregation")
    print("="*60)
    print(f"\nInitial foods ({len(df)} items):")
    for i, name in enumerate(df['foodName'], 1):
        print(f"  {i}. {name}")
    
    # Initialize agent
    agent = FoodAggregationAgent(
        df,
        llm_api_key=os.getenv('OPENROUTER_API_KEY')  # Will use heuristics if None
    )
    
    # Run one iteration
    print("\n" + "="*60)
    print("Running aggregation workflow...")
    print("="*60)
    
    merges = agent.run_iteration(
        similarity_threshold=0.80,  # Lower threshold for demo
        max_merges_per_iteration=10,
        use_llm_validation=bool(os.getenv('OPENROUTER_API_KEY'))
    )
    
    # Show results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nFinal foods ({len(agent.df)} items):")
    for i, name in enumerate(agent.df['foodName'], 1):
        print(f"  {i}. {name}")
    
    print(f"\nMerge history ({len(agent.merge_history)} merges):")
    for merge in agent.merge_history:
        merged = ' + '.join(merge['merged_items'])
        generic = merge['generic_name']
        print(f"  → {merged}")
        print(f"    = {generic}")
    
    print("\n" + "="*60)
    print(f"Reduction: {len(df) - len(agent.df)} items ({100 * (1 - len(agent.df)/len(df)):.1f}%)")
    print("="*60)
    
    return agent.df


def test_with_real_data():
    """Test on 100 items from the real database."""
    
    data_path = 'pyfooda/data/foods.csv'
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Please run from the repo root directory")
        return
    
    df = pd.read_csv(data_path).head(100)
    
    print("="*60)
    print("Testing on 100 Real Food Items")
    print("="*60)
    
    agent = FoodAggregationAgent(df)
    
    aggregated = agent.run_until_convergence(
        max_iterations=3,
        similarity_threshold=0.85,
        use_llm_validation=False  # Heuristics only for speed
    )
    
    print(f"\nSample of aggregated foods:")
    print(aggregated['foodName'].head(20).tolist())
    
    return aggregated


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test aggregation workflow')
    parser.add_argument('--real', action='store_true', help='Test on real data (100 items)')
    args = parser.parse_args()
    
    if args.real:
        test_with_real_data()
    else:
        test_small_sample()
