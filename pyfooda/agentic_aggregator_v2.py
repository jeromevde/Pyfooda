"""
Agentic Food Aggregation Workflow V2

Enhanced version with:
- Rate limiting for free tier models
- Checkpoint/resume functionality
- Progress tracking and ETA
- Graceful interruption handling
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Set, Optional
import json
from tqdm import tqdm
import openai
import os
import time
import pickle
from pathlib import Path
from datetime import datetime, timedelta


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, calls_per_minute: int = 15):
        self.calls_per_minute = calls_per_minute
        self.calls = []
        self.min_interval = 60.0 / calls_per_minute if calls_per_minute > 0 else 0
    
    def wait_if_needed(self):
        """Wait if we've exceeded the rate limit."""
        if self.calls_per_minute <= 0:
            return  # No rate limiting
        
        now = time.time()
        
        # Remove calls older than 1 minute
        self.calls = [t for t in self.calls if now - t < 60]
        
        # If we've hit the limit, wait
        if len(self.calls) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.calls[0]) + 0.1  # Small buffer
            if sleep_time > 0:
                time.sleep(sleep_time)
                now = time.time()
                self.calls = [t for t in self.calls if now - t < 60]
        
        # Also enforce minimum interval between calls
        if self.calls and now - self.calls[-1] < self.min_interval:
            time.sleep(self.min_interval - (now - self.calls[-1]))
        
        self.calls.append(time.time())


class FoodAggregationAgentV2:
    """
    Enhanced agent with rate limiting and checkpoint/resume.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        embedding_model: str = 'all-MiniLM-L6-v2',
        llm_api_key: str = None,
        llm_base_url: str = None,
        llm_model: str = "google/gemini-flash-1.5-8b",  # Default to free model
        rate_limit: int = 15,  # calls per minute
        checkpoint_dir: str = "./checkpoints"
    ):
        """
        Initialize the aggregation agent.
        
        Args:
            df: DataFrame with food data
            embedding_model: SentenceTransformer model name
            llm_api_key: API key for LLM (OpenRouter or OpenAI)
            llm_base_url: Base URL for LLM API
            llm_model: Model to use for validation
            rate_limit: Max API calls per minute (0 = no limit)
            checkpoint_dir: Directory to save checkpoints
        """
        self.df = df.copy()
        self.original_df = df.copy()
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # LLM configuration
        self.llm_api_key = llm_api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.llm_base_url = llm_base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.llm_model = llm_model
        
        # Rate limiting
        self.rate_limiter = RateLimiter(rate_limit)
        
        # Checkpoint management
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Track merges and progress
        self.merge_history = []
        self.iteration = 0
        self.total_api_calls = 0
        self.start_time = None
        
        # Nutrient columns for similarity
        self.nutrient_cols = [
            'Energy', 'Carbohydrate', 'Protein', 'Total fat',
            'Fiber', 'Sugars, Total', 'Sodium', 'Calcium', 'Iron'
        ]
    
    def save_checkpoint(self, iteration: int):
        """Save current state to checkpoint file."""
        checkpoint = {
            'df': self.df,
            'merge_history': self.merge_history,
            'iteration': iteration,
            'total_api_calls': self.total_api_calls,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_iter_{iteration}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Also save merge history as JSON for easy inspection
        history_path = self.checkpoint_dir / f"merge_history_iter_{iteration}.json"
        with open(history_path, 'w') as f:
            json.dump(self.merge_history, f, indent=2)
        
        print(f"💾 Checkpoint saved: {checkpoint_path.name}")
    
    @classmethod
    def load_checkpoint(cls, checkpoint_path: str, **kwargs):
        """Load state from checkpoint file."""
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Create agent with loaded data
        agent = cls(checkpoint['df'], **kwargs)
        agent.df = checkpoint['df']
        agent.merge_history = checkpoint['merge_history']
        agent.iteration = checkpoint['iteration']
        agent.total_api_calls = checkpoint['total_api_calls']
        
        print(f"📂 Loaded checkpoint from iteration {checkpoint['iteration']}")
        print(f"   Timestamp: {checkpoint['timestamp']}")
        print(f"   Current DB size: {len(agent.df)} items")
        print(f"   Merges so far: {len(agent.merge_history)}")
        
        return agent
    
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
        """Compute combined similarity matrix."""
        semantic_sim = cosine_similarity(embeddings)
        nutrient_matrix = nutrient_data[self.nutrient_cols].fillna(0).values
        nutrient_sim = cosine_similarity(nutrient_matrix)
        combined_sim = 0.7 * semantic_sim + 0.3 * nutrient_sim
        return combined_sim
    
    def _ask_llm_to_validate_merge(self, food1: str, food2: str) -> Tuple[bool, str]:
        """Ask LLM if two foods should be merged."""
        if not self.llm_api_key:
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
            # Rate limiting
            self.rate_limiter.wait_if_needed()
            
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
            
            self.total_api_calls += 1
            
            content = response.choices[0].message.content.strip()
            if content.startswith('```'):
                content = '\n'.join(content.split('\n')[1:-1])
            
            result = json.loads(content)
            return result.get("should_merge", False), result.get("generic_name", food1)
        
        except Exception as e:
            print(f"⚠ LLM validation failed: {e}")
            return len(set(food1.lower().split()) & set(food2.lower().split())) >= 2, food1
    
    def _find_merge_candidates(
        self,
        similarity_threshold: float = 0.85,
        max_candidates: int = 100
    ) -> List[Tuple[int, int, float]]:
        """Find pairs of food items that are candidates for merging."""
        food_names = self.df['foodName'].fillna('').tolist()
        embeddings = self._compute_embeddings(food_names)
        normalized_df = self._normalize_nutrients(self.df)
        sim_matrix = self._compute_similarity_matrix(embeddings, normalized_df)
        
        candidates = []
        n = len(food_names)
        
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= similarity_threshold:
                    candidates.append((i, j, sim_matrix[i, j]))
        
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[:max_candidates]
    
    def _merge_foods(self, idx1: int, idx2: int, generic_name: str) -> None:
        """Merge two food items."""
        for col in self.nutrient_cols:
            if col in self.df.columns:
                val1 = self.df.iloc[idx1][col]
                val2 = self.df.iloc[idx2][col]
                
                if pd.isna(val1) and pd.isna(val2):
                    avg = np.nan
                elif pd.isna(val1):
                    avg = val2
                elif pd.isna(val2):
                    avg = val1
                else:
                    avg = (val1 + val2) / 2
                
                self.df.at[self.df.index[idx1], col] = avg
        
        self.df.at[self.df.index[idx1], 'foodName'] = generic_name
        
        self.merge_history.append({
            'iteration': self.iteration,
            'merged_items': [
                self.original_df.iloc[idx1]['foodName'],
                self.original_df.iloc[idx2]['foodName']
            ],
            'generic_name': generic_name
        })
        
        self.df = self.df.drop(self.df.index[idx2]).reset_index(drop=True)
    
    def run_iteration(
        self,
        similarity_threshold: float = 0.85,
        max_merges_per_iteration: int = 50,
        use_llm_validation: bool = True,
        checkpoint_every: int = 10
    ) -> int:
        """Run one iteration of the aggregation workflow."""
        self.iteration += 1
        
        if self.start_time is None:
            self.start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Iteration {self.iteration}")
        print(f"Current database size: {len(self.df)} items")
        print(f"Total API calls: {self.total_api_calls}")
        
        # ETA calculation
        if self.total_api_calls > 0 and self.rate_limiter.calls_per_minute > 0:
            elapsed = time.time() - self.start_time
            rate = self.total_api_calls / elapsed * 60  # calls per minute
            print(f"Current API rate: {rate:.1f} calls/min")
        
        print(f"{'='*60}")
        
        candidates = self._find_merge_candidates(
            similarity_threshold=similarity_threshold
        )
        
        if not candidates:
            print("No merge candidates found. Stopping.")
            return 0
        
        print(f"Found {len(candidates)} potential merge candidates")
        
        merges_performed = 0
        processed_indices = set()
        
        pbar = tqdm(candidates[:max_merges_per_iteration], desc="Validating merges")
        
        try:
            for idx1, idx2, sim_score in pbar:
                if idx1 in processed_indices or idx2 in processed_indices:
                    continue
                
                food1 = self.df.iloc[idx1]['foodName']
                food2 = self.df.iloc[idx2]['foodName']
                
                if use_llm_validation:
                    should_merge, generic_name = self._ask_llm_to_validate_merge(food1, food2)
                else:
                    should_merge = len(set(food1.lower().split()) & set(food2.lower().split())) >= 2
                    generic_name = food1
                
                if should_merge:
                    pbar.set_postfix_str(f"Merged → {generic_name[:30]}")
                    self._merge_foods(idx1, idx2, generic_name)
                    processed_indices.add(idx1)
                    processed_indices.add(idx2)
                    merges_performed += 1
                    
                    # Checkpoint periodically
                    if merges_performed % checkpoint_every == 0:
                        self.save_checkpoint(self.iteration)
        
        except KeyboardInterrupt:
            print("\n⚠ Interrupted! Saving checkpoint...")
            self.save_checkpoint(self.iteration)
            raise
        
        print(f"\nMerges performed: {merges_performed}")
        print(f"New database size: {len(self.df)} items")
        
        # Save checkpoint at end of iteration
        self.save_checkpoint(self.iteration)
        
        return merges_performed
    
    def run_until_convergence(
        self,
        max_iterations: int = 10,
        similarity_threshold: float = 0.85,
        max_merges_per_iteration: int = 50,
        use_llm_validation: bool = True,
        checkpoint_every: int = 10
    ) -> pd.DataFrame:
        """Run until convergence with checkpointing."""
        print("Starting agentic food aggregation workflow (V2)")
        print(f"Initial database size: {len(self.df)} items")
        print(f"Model: {self.llm_model}")
        print(f"Rate limit: {self.rate_limiter.calls_per_minute} calls/min")
        print(f"Checkpoints: Every {checkpoint_every} merges")
        
        try:
            for i in range(max_iterations):
                merges = self.run_iteration(
                    similarity_threshold=similarity_threshold,
                    max_merges_per_iteration=max_merges_per_iteration,
                    use_llm_validation=use_llm_validation,
                    checkpoint_every=checkpoint_every
                )
                
                if merges == 0:
                    print(f"\n✓ Convergence reached after {self.iteration} iterations")
                    break
        
        except KeyboardInterrupt:
            print("\n⚠ Workflow interrupted by user")
            print("You can resume from the last checkpoint using:")
            print(f"  agent = FoodAggregationAgentV2.load_checkpoint('{self.checkpoint_dir}/checkpoint_iter_{self.iteration}.pkl')")
        
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Original size: {len(self.original_df)} items")
        print(f"Final size: {len(self.df)} items")
        print(f"Reduction: {len(self.original_df) - len(self.df)} items ({100 * (1 - len(self.df)/len(self.original_df)):.1f}%)")
        print(f"Total merges: {len(self.merge_history)}")
        print(f"Total API calls: {self.total_api_calls}")
        
        return self.df
    
    def save_results(self, output_path: str, merge_history_path: str = None):
        """Save aggregated database and merge history."""
        self.df.to_csv(output_path, index=False)
        print(f"\n✓ Saved aggregated database to {output_path}")
        
        if merge_history_path:
            with open(merge_history_path, 'w') as f:
                json.dump(self.merge_history, f, indent=2)
            print(f"✓ Saved merge history to {merge_history_path}")
