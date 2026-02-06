"""
Agentic Food Aggregation Workflow

This module implements an iterative agentic workflow to merge similar food items
into a lightweight, everyday-use nutrition database. The workflow uses LLM-based
similarity detection and iteratively merges items until a fixed point is reached.

Key concepts:
1. Agent identifies potential merges using semantic similarity + nutrient profiles
2. Each iteration proposes merges and validates them
3. Process continues until no more meaningful merges can be found
4. Outputs a lightweight database with representative items
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Set
import json
from tqdm import tqdm
import openai
import os


class FoodAggregationAgent:
    """
    Agent that iteratively merges similar food items into generic categories.
    Uses a combination of semantic embeddings and LLM validation.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        embedding_model: str = 'all-MiniLM-L6-v2',
        llm_api_key: str = None,
        llm_base_url: str = None,
        llm_model: str = "qwen/qwen-2.5-7b-instruct"
    ):
        """
        Initialize the aggregation agent.
        
        Args:
            df: DataFrame with food data
            embedding_model: SentenceTransformer model name
            llm_api_key: API key for LLM (OpenRouter or OpenAI)
            llm_base_url: Base URL for LLM API (for OpenRouter)
            llm_model: Model to use for validation
        """
        self.df = df.copy()
        self.original_df = df.copy()
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # LLM configuration
        self.llm_api_key = llm_api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.llm_base_url = llm_base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.llm_model = llm_model
        
        # Track merges
        self.merge_history = []
        self.iteration = 0
        
        # Nutrient columns for similarity
        self.nutrient_cols = [
            'Energy', 'Carbohydrate', 'Protein', 'Total fat',
            'Fiber', 'Sugars, Total', 'Sodium', 'Calcium', 'Iron'
        ]
        
    def _compute_embeddings(self, food_names: List[str]) -> np.ndarray:
        """Compute sentence embeddings for food names."""
        return self.embedding_model.encode(food_names, show_progress_bar=False)
    
    def _normalize_nutrients(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize nutrient values for comparison."""
        normalized = df.copy()
        for col in self.nutrient_cols:
            if col in df.columns:
                vals = df[col].fillna(0)
                if vals.std() > 0:
                    normalized[col] = (vals - vals.mean()) / vals.std()
                else:
                    normalized[col] = 0
        return normalized
    
    def _compute_similarity_matrix(
        self,
        embeddings: np.ndarray,
        nutrient_data: pd.DataFrame
    ) -> np.ndarray:
        """
        Compute combined similarity matrix using:
        - Semantic similarity (70%)
        - Nutrient profile similarity (30%)
        """
        # Semantic similarity
        semantic_sim = cosine_similarity(embeddings)
        
        # Nutrient similarity
        nutrient_matrix = nutrient_data[self.nutrient_cols].fillna(0).values
        nutrient_sim = cosine_similarity(nutrient_matrix)
        
        # Weighted combination
        combined_sim = 0.7 * semantic_sim + 0.3 * nutrient_sim
        
        return combined_sim
    
    def _ask_llm_to_validate_merge(self, food1: str, food2: str) -> Tuple[bool, str]:
        """
        Ask LLM if two food items should be merged into a generic category.
        
        Returns:
            (should_merge: bool, generic_name: str)
        """
        if not self.llm_api_key:
            # Fallback: merge if names are very similar
            return len(set(food1.lower().split()) & set(food2.lower().split())) >= 2, food1
        
        prompt = f"""You are a nutrition database expert. Decide if these two food items should be merged into one generic category.

Food 1: {food1}
Food 2: {food2}

Rules:
- Merge if they represent the same everyday food (e.g., "ORGANIC APPLE" + "APPLE, RAW" → "Apple")
- Don't merge if they're fundamentally different (e.g., "APPLE PIE" vs "APPLE")
- Prefer simple, everyday names for generic categories

Respond in JSON format:
{{
  "should_merge": true/false,
  "generic_name": "Simple generic name if merging, or empty string if not",
  "reason": "Brief explanation"
}}

Respond ONLY with valid JSON, no other text."""

        try:
            client = openai.OpenAI(
                api_key=self.llm_api_key,
                base_url=self.llm_base_url
            )
            
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON
            if content.startswith('```'):
                content = '\n'.join(content.split('\n')[1:-1])
            
            result = json.loads(content)
            return result.get("should_merge", False), result.get("generic_name", food1)
        
        except Exception as e:
            print(f"LLM validation failed: {e}")
            # Fallback to heuristic
            return len(set(food1.lower().split()) & set(food2.lower().split())) >= 2, food1
    
    def _find_merge_candidates(
        self,
        similarity_threshold: float = 0.85,
        max_candidates: int = 100
    ) -> List[Tuple[int, int, float]]:
        """
        Find pairs of food items that are candidates for merging.
        
        Returns:
            List of (idx1, idx2, similarity_score) tuples
        """
        # Get current food names
        food_names = self.df['foodName'].fillna('').tolist()
        
        # Compute embeddings
        embeddings = self._compute_embeddings(food_names)
        
        # Normalize nutrients
        normalized_df = self._normalize_nutrients(self.df)
        
        # Compute similarity matrix
        sim_matrix = self._compute_similarity_matrix(embeddings, normalized_df)
        
        # Find candidate pairs
        candidates = []
        n = len(food_names)
        
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= similarity_threshold:
                    candidates.append((i, j, sim_matrix[i, j]))
        
        # Sort by similarity (highest first)
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        return candidates[:max_candidates]
    
    def _merge_foods(self, idx1: int, idx2: int, generic_name: str) -> None:
        """
        Merge two food items by:
        1. Averaging their nutrient values
        2. Using the generic name
        3. Removing the second item
        """
        # Average nutrients
        for col in self.nutrient_cols:
            if col in self.df.columns:
                val1 = self.df.iloc[idx1][col]
                val2 = self.df.iloc[idx2][col]
                
                # Handle NaN
                if pd.isna(val1) and pd.isna(val2):
                    avg = np.nan
                elif pd.isna(val1):
                    avg = val2
                elif pd.isna(val2):
                    avg = val1
                else:
                    avg = (val1 + val2) / 2
                
                self.df.at[self.df.index[idx1], col] = avg
        
        # Update name
        self.df.at[self.df.index[idx1], 'foodName'] = generic_name
        
        # Track merge
        self.merge_history.append({
            'iteration': self.iteration,
            'merged_items': [
                self.original_df.iloc[idx1]['foodName'],
                self.original_df.iloc[idx2]['foodName']
            ],
            'generic_name': generic_name
        })
        
        # Remove second item
        self.df = self.df.drop(self.df.index[idx2]).reset_index(drop=True)
    
    def run_iteration(
        self,
        similarity_threshold: float = 0.85,
        max_merges_per_iteration: int = 50,
        use_llm_validation: bool = True
    ) -> int:
        """
        Run one iteration of the aggregation workflow.
        
        Returns:
            Number of merges performed
        """
        self.iteration += 1
        print(f"\n{'='*60}")
        print(f"Iteration {self.iteration}")
        print(f"Current database size: {len(self.df)} items")
        print(f"{'='*60}")
        
        # Find candidates
        candidates = self._find_merge_candidates(
            similarity_threshold=similarity_threshold
        )
        
        if not candidates:
            print("No merge candidates found. Stopping.")
            return 0
        
        print(f"Found {len(candidates)} potential merge candidates")
        
        # Validate and merge
        merges_performed = 0
        processed_indices = set()
        
        for idx1, idx2, sim_score in tqdm(candidates[:max_merges_per_iteration], desc="Validating merges"):
            # Skip if either index was already merged
            if idx1 in processed_indices or idx2 in processed_indices:
                continue
            
            food1 = self.df.iloc[idx1]['foodName']
            food2 = self.df.iloc[idx2]['foodName']
            
            # Validate merge
            if use_llm_validation:
                should_merge, generic_name = self._ask_llm_to_validate_merge(food1, food2)
            else:
                # Heuristic: merge if names share significant words
                should_merge = len(set(food1.lower().split()) & set(food2.lower().split())) >= 2
                generic_name = food1  # Use first name as generic
            
            if should_merge:
                print(f"  ✓ Merging: '{food1}' + '{food2}' → '{generic_name}'")
                self._merge_foods(idx1, idx2, generic_name)
                processed_indices.add(idx1)
                processed_indices.add(idx2)
                merges_performed += 1
        
        print(f"\nMerges performed: {merges_performed}")
        print(f"New database size: {len(self.df)} items")
        
        return merges_performed
    
    def run_until_convergence(
        self,
        max_iterations: int = 10,
        similarity_threshold: float = 0.85,
        max_merges_per_iteration: int = 50,
        use_llm_validation: bool = True
    ) -> pd.DataFrame:
        """
        Run the aggregation workflow until convergence (no more merges found).
        
        Returns:
            Final aggregated DataFrame
        """
        print("Starting agentic food aggregation workflow")
        print(f"Initial database size: {len(self.df)} items")
        
        for i in range(max_iterations):
            merges = self.run_iteration(
                similarity_threshold=similarity_threshold,
                max_merges_per_iteration=max_merges_per_iteration,
                use_llm_validation=use_llm_validation
            )
            
            if merges == 0:
                print(f"\n✓ Convergence reached after {self.iteration} iterations")
                break
        else:
            print(f"\n⚠ Max iterations ({max_iterations}) reached")
        
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Original size: {len(self.original_df)} items")
        print(f"Final size: {len(self.df)} items")
        print(f"Reduction: {len(self.original_df) - len(self.df)} items ({100 * (1 - len(self.df)/len(self.original_df)):.1f}%)")
        print(f"Total merges: {len(self.merge_history)}")
        
        return self.df
    
    def save_results(self, output_path: str, merge_history_path: str = None):
        """Save aggregated database and merge history."""
        self.df.to_csv(output_path, index=False)
        print(f"\n✓ Saved aggregated database to {output_path}")
        
        if merge_history_path:
            with open(merge_history_path, 'w') as f:
                json.dump(self.merge_history, f, indent=2)
            print(f"✓ Saved merge history to {merge_history_path}")


def run_aggregation_workflow(
    input_csv: str = 'pyfooda/data/foods.csv',
    output_csv: str = 'pyfooda/data/foods_aggregated.csv',
    sample_size: int = 1000,
    similarity_threshold: float = 0.85,
    max_iterations: int = 10,
    use_llm: bool = True
):
    """
    Convenience function to run the full aggregation workflow.
    
    Args:
        input_csv: Path to input food database
        output_csv: Path to save aggregated database
        sample_size: Number of items to sample (None = use all)
        similarity_threshold: Minimum similarity for merge candidates
        max_iterations: Max iterations before stopping
        use_llm: Whether to use LLM for validation
    """
    # Load data
    df = pd.read_csv(input_csv)
    
    # Sample if requested
    if sample_size:
        df = df.sample(min(sample_size, len(df)), random_state=42)
    
    print(f"Loaded {len(df)} food items")
    
    # Initialize agent
    agent = FoodAggregationAgent(df)
    
    # Run workflow
    aggregated_df = agent.run_until_convergence(
        max_iterations=max_iterations,
        similarity_threshold=similarity_threshold,
        use_llm_validation=use_llm
    )
    
    # Save results
    agent.save_results(
        output_path=output_csv,
        merge_history_path=output_csv.replace('.csv', '_history.json')
    )
    
    return aggregated_df


if __name__ == "__main__":
    # Example usage
    run_aggregation_workflow(
        sample_size=500,  # Start with 500 items for testing
        similarity_threshold=0.85,
        max_iterations=5,
        use_llm=True  # Set to False if no API key available
    )
