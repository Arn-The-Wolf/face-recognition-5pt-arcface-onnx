#!/usr/bin/env python3
"""
Face recognition evaluation module
Implements threshold evaluation as specified in the document
"""

import cv2
import numpy as np
import os
import json
from itertools import combinations
import matplotlib.pyplot as plt

class FaceRecognitionEvaluator:
    """Face recognition system evaluator"""
    
    def __init__(self, data_dir="data"):
        """
        Initialize evaluator
        
        Args:
            data_dir: Base data directory
        """
        self.data_dir = data_dir
        self.enroll_dir = os.path.join(data_dir, "enroll")
        self.db_dir = os.path.join(data_dir, "db")
        
        print("Face recognition evaluator initialized")
        
    def load_enrolled_crops(self):
        """
        Load all aligned face crops from enrollment directory
        
        Returns:
            dict: {person_name: [list of aligned face images]}
        """
        crops_by_person = {}
        
        if not os.path.exists(self.enroll_dir):
            print(f"Enrollment directory not found: {self.enroll_dir}")
            return crops_by_person
            
        for person_name in os.listdir(self.enroll_dir):
            person_dir = os.path.join(self.enroll_dir, person_name)
            if not os.path.isdir(person_dir):
                continue
                
            crops = []
            for filename in os.listdir(person_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    crop_path = os.path.join(person_dir, filename)
                    crop = cv2.imread(crop_path)
                    if crop is not None:
                        crops.append(crop)
                        
            if crops:
                crops_by_person[person_name] = crops
                print(f"Loaded {len(crops)} crops for {person_name}")
                
        return crops_by_person
        
    def extract_embeddings_from_crops(self, crops_by_person):
        """
        Extract embeddings from face crops
        
        Args:
            crops_by_person: Dictionary of crops by person
            
        Returns:
            dict: {person_name: [list of embeddings]}
        """
        try:
            from embed import ArcFaceEmbedder
            
            model_path = "models/embedder_arcface.onnx"
            if not os.path.exists(model_path):
                print(f"ERROR: ArcFace model not found at {model_path}")
                return {}
                
            embedder = ArcFaceEmbedder(model_path)
            embeddings_by_person = {}
            
            for person_name, crops in crops_by_person.items():
                embeddings = []
                for crop in crops:
                    # Crops should already be aligned to 112x112
                    embedding = embedder.extract_embedding(crop)
                    if embedding is not None:
                        embeddings.append(embedding)
                        
                if embeddings:
                    embeddings_by_person[person_name] = embeddings
                    print(f"Extracted {len(embeddings)} embeddings for {person_name}")
                    
            return embeddings_by_person
            
        except Exception as e:
            print(f"Embedding extraction failed: {e}")
            return {}
            
    def compute_distances(self, embeddings_by_person):
        """
        Compute genuine and impostor distances
        
        Args:
            embeddings_by_person: Dictionary of embeddings by person
            
        Returns:
            tuple: (genuine_distances, impostor_distances)
        """
        genuine_distances = []
        impostor_distances = []
        
        # Compute genuine distances (same person)
        for person_name, embeddings in embeddings_by_person.items():
            if len(embeddings) < 2:
                continue
                
            # All pairs within same person
            for emb1, emb2 in combinations(embeddings, 2):
                # Convert cosine similarity to distance
                similarity = np.dot(emb1, emb2)
                distance = 1.0 - similarity
                genuine_distances.append(distance)
                
        # Compute impostor distances (different persons)
        person_names = list(embeddings_by_person.keys())
        for i, person1 in enumerate(person_names):
            for j, person2 in enumerate(person_names[i+1:], i+1):
                embeddings1 = embeddings_by_person[person1]
                embeddings2 = embeddings_by_person[person2]
                
                # Compare all pairs between different persons
                for emb1 in embeddings1:
                    for emb2 in embeddings2:
                        similarity = np.dot(emb1, emb2)
                        distance = 1.0 - similarity
                        impostor_distances.append(distance)
                        
        return np.array(genuine_distances), np.array(impostor_distances)
        
    def compute_far_frr(self, genuine_distances, impostor_distances, thresholds):
        """
        Compute False Accept Rate (FAR) and False Reject Rate (FRR)
        
        Args:
            genuine_distances: Array of genuine distances
            impostor_distances: Array of impostor distances
            thresholds: Array of threshold values
            
        Returns:
            tuple: (far_values, frr_values)
        """
        far_values = []
        frr_values = []
        
        for threshold in thresholds:
            # FAR: fraction of impostor pairs accepted (distance < threshold)
            if len(impostor_distances) > 0:
                far = np.sum(impostor_distances < threshold) / len(impostor_distances)
            else:
                far = 0.0
                
            # FRR: fraction of genuine pairs rejected (distance >= threshold)
            if len(genuine_distances) > 0:
                frr = np.sum(genuine_distances >= threshold) / len(genuine_distances)
            else:
                frr = 0.0
                
            far_values.append(far)
            frr_values.append(frr)
            
        return np.array(far_values), np.array(frr_values)
        
    def find_threshold_for_far(self, genuine_distances, impostor_distances, target_far=0.01):
        """
        Find threshold that achieves target FAR
        
        Args:
            genuine_distances: Array of genuine distances
            impostor_distances: Array of impostor distances
            target_far: Target False Accept Rate (default 1%)
            
        Returns:
            tuple: (threshold, actual_far, frr_at_threshold)
        """
        if len(impostor_distances) == 0:
            return 0.5, 0.0, 0.0
            
        # Sort impostor distances
        sorted_impostors = np.sort(impostor_distances)
        
        # Find threshold for target FAR
        far_index = int(target_far * len(sorted_impostors))
        if far_index >= len(sorted_impostors):
            threshold = sorted_impostors[-1] + 0.1
        else:
            threshold = sorted_impostors[far_index]
            
        # Compute actual FAR and FRR at this threshold
        actual_far = np.sum(impostor_distances < threshold) / len(impostor_distances)
        
        if len(genuine_distances) > 0:
            frr = np.sum(genuine_distances >= threshold) / len(genuine_distances)
        else:
            frr = 0.0
            
        return threshold, actual_far, frr
        
    def print_evaluation_results(self, genuine_distances, impostor_distances):
        """Print comprehensive evaluation results"""
        
        print("\n=== Face Recognition Evaluation Results ===")
        
        # Distribution summaries
        print(f"\nGenuine distances ({len(genuine_distances)} pairs):")
        if len(genuine_distances) > 0:
            print(f"  Mean: {np.mean(genuine_distances):.4f}")
            print(f"  Std:  {np.std(genuine_distances):.4f}")
            print(f"  Min:  {np.min(genuine_distances):.4f}")
            print(f"  Max:  {np.max(genuine_distances):.4f}")
        else:
            print("  No genuine pairs found")
            
        print(f"\nImpostor distances ({len(impostor_distances)} pairs):")
        if len(impostor_distances) > 0:
            print(f"  Mean: {np.mean(impostor_distances):.4f}")
            print(f"  Std:  {np.std(impostor_distances):.4f}")
            print(f"  Min:  {np.min(impostor_distances):.4f}")
            print(f"  Max:  {np.max(impostor_distances):.4f}")
        else:
            print("  No impostor pairs found")
            
        # FAR/FRR sweep table
        print(f"\n=== FAR/FRR Sweep Table ===")
        thresholds = np.linspace(0.0, 1.0, 21)
        far_values, frr_values = self.compute_far_frr(genuine_distances, impostor_distances, thresholds)
        
        print(f"{'Threshold':<10} {'FAR':<8} {'FRR':<8} {'EER Diff':<10}")
        print("-" * 40)
        
        for i, (thresh, far, frr) in enumerate(zip(thresholds, far_values, frr_values)):
            eer_diff = abs(far - frr)
            print(f"{thresh:<10.3f} {far:<8.4f} {frr:<8.4f} {eer_diff:<10.4f}")
            
        # Find EER (Equal Error Rate)
        eer_idx = np.argmin(np.abs(far_values - frr_values))
        eer_threshold = thresholds[eer_idx]
        eer_value = (far_values[eer_idx] + frr_values[eer_idx]) / 2
        
        print(f"\nEqual Error Rate (EER): {eer_value:.4f} at threshold {eer_threshold:.3f}")
        
        # Suggested threshold for FAR = 1%
        threshold_1pct, actual_far, frr_at_1pct = self.find_threshold_for_far(
            genuine_distances, impostor_distances, target_far=0.01
        )
        
        print(f"\n=== Suggested Operating Point ===")
        print(f"For FAR = 1%:")
        print(f"  Threshold: {threshold_1pct:.4f}")
        print(f"  Actual FAR: {actual_far:.4f} ({actual_far*100:.2f}%)")
        print(f"  FRR: {frr_at_1pct:.4f} ({frr_at_1pct*100:.2f}%)")
        
        return {
            'eer_threshold': eer_threshold,
            'eer_value': eer_value,
            'suggested_threshold': threshold_1pct,
            'far_at_suggested': actual_far,
            'frr_at_suggested': frr_at_1pct
        }
        
    def plot_distributions(self, genuine_distances, impostor_distances, save_path=None):
        """Plot distance distributions"""
        try:
            plt.figure(figsize=(12, 8))
            
            # Plot histograms
            plt.subplot(2, 2, 1)
            if len(genuine_distances) > 0:
                plt.hist(genuine_distances, bins=50, alpha=0.7, label='Genuine', color='green')
            if len(impostor_distances) > 0:
                plt.hist(impostor_distances, bins=50, alpha=0.7, label='Impostor', color='red')
            plt.xlabel('Distance')
            plt.ylabel('Frequency')
            plt.title('Distance Distributions')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot FAR/FRR curves
            plt.subplot(2, 2, 2)
            thresholds = np.linspace(0.0, 1.0, 100)
            far_values, frr_values = self.compute_far_frr(genuine_distances, impostor_distances, thresholds)
            
            plt.plot(thresholds, far_values, label='FAR', color='red')
            plt.plot(thresholds, frr_values, label='FRR', color='blue')
            plt.xlabel('Threshold')
            plt.ylabel('Error Rate')
            plt.title('FAR/FRR Curves')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot ROC curve
            plt.subplot(2, 2, 3)
            plt.plot(far_values, 1 - frr_values, color='blue')
            plt.xlabel('False Accept Rate')
            plt.ylabel('True Accept Rate')
            plt.title('ROC Curve')
            plt.grid(True, alpha=0.3)
            
            # Plot DET curve (log scale)
            plt.subplot(2, 2, 4)
            plt.loglog(far_values[far_values > 0], frr_values[far_values > 0], color='purple')
            plt.xlabel('False Accept Rate')
            plt.ylabel('False Reject Rate')
            plt.title('DET Curve')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Plots saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("Matplotlib not available, skipping plots")
        except Exception as e:
            print(f"Plotting failed: {e}")
            
    def run_evaluation(self):
        """Run complete evaluation pipeline"""
        print("Starting face recognition evaluation...")
        
        # Load crops
        crops_by_person = self.load_enrolled_crops()
        if not crops_by_person:
            print("No enrolled face crops found")
            return
            
        print(f"Found crops for {len(crops_by_person)} people")
        
        # Extract embeddings
        embeddings_by_person = self.extract_embeddings_from_crops(crops_by_person)
        if not embeddings_by_person:
            print("No embeddings extracted")
            return
            
        # Compute distances
        genuine_distances, impostor_distances = self.compute_distances(embeddings_by_person)
        
        # Print results
        results = self.print_evaluation_results(genuine_distances, impostor_distances)
        
        # Plot distributions
        self.plot_distributions(genuine_distances, impostor_distances, 'evaluation_plots.png')
        
        return results

def main():
    """Main evaluation interface"""
    evaluator = FaceRecognitionEvaluator()
    
    print("=== Face Recognition Evaluation ===")
    print("This will evaluate the recognition system using enrolled face crops")
    print("Make sure you have enrolled people with multiple samples first")
    
    input("Press Enter to start evaluation...")
    
    results = evaluator.run_evaluation()
    
    if results:
        print(f"\nEvaluation completed successfully!")
        print(f"Recommended threshold: {results['suggested_threshold']:.4f}")

if __name__ == "__main__":
    main()