import os

# Fixes openblas warning, from creator's github:
# For systems using OpenBLAS, I highly recommend setting 'export OPENBLAS_NUM_THREADS=1'.
# This disables its internal multithreading ability, which leads to substantial speedups for this package. 
# Likewise for Intel MKL, setting 'export MKL_NUM_THREADS=1' should also be set.
# https://github.com/benfred/implicit/
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import random
import time
import numpy as np
import pandas as pd
from mtg_database import MTGDatabase
from als_recommender import ALSRecommender

class ALSRecommenderEvaluator:
    def __init__(self, db_config='db_config.ini', num_decks=50, random_seed=29):
        self.db = MTGDatabase(db_config)
        self.num_decks = num_decks
        self.random_seed = random_seed
        self.basic_lands = ['Island', 'Swamp', 'Mountain', 'Plains', 'Forest']
        random.seed(self.random_seed)
        
    def is_basic_land(self, card_name):
        return card_name in self.basic_lands
    
    def is_land_card(self, card):
        if isinstance(card, dict):
            if 'type_line' in card and card['type_line'] and 'Land' in card['type_line']:
                return True
            card_id = card['card_id']
        else:
            card_id = card
        
        try:
            card_types = self.db.get_card_types(card_id)
            return any(t['type_name'] == 'Land' for t in card_types)
        except Exception as e:
            print(f"Error checking if card {card_id} is a land: {str(e)}")
            return False
        
    def split_deck(self, deck_id, seed_percentage=0.4):

        deck_info = self.db.get_deck_info(deck_id)
        if not deck_info:
            raise ValueError(f"Could not find deck id: {deck_id}")

        # get ids for commander and other cards
        commander_id = deck_info['primary_commander_id']
        all_cards = self.db.get_deck_cards(deck_id)
        
        # filter out lands and commander
        filtered_cards = [
            card for card in all_cards 
            if not self.is_land_card(card) and card['card_id'] != commander_id
        ]
        
        # randomly select seed cards and extract card ids
        seed_count = max(1, int(len(filtered_cards) * seed_percentage))
        random.shuffle(filtered_cards)
        seed_cards = filtered_cards[:seed_count]
        target_cards = filtered_cards[seed_count:]
        seed_card_ids = [card['card_id'] for card in seed_cards]
        target_card_ids = [card['card_id'] for card in target_cards]
        
        return seed_card_ids, target_card_ids, commander_id
    
    def evaluate_recommender(self, recommender, commander_id, seed_cards, target_cards, limit=100):

        start_time = time.time()
        
        # get recs
        recommendations = recommender.get_recommendations(
            commander_id, 
            existing_cards=seed_cards, 
            limit=limit
        )
        
        recommendation_time = time.time() - start_time
        
        # filter out lands
        non_land_recommendations = [
            card for card in recommendations 
            if not self.is_land_card(card)
        ]
        
        recommended_ids = [card['card_id'] for card in non_land_recommendations]
        
        # calc perfomance metrics
        true_positives = set(recommended_ids).intersection(set(target_cards))
        precision = len(true_positives)/ len(recommended_ids) if recommended_ids else 0
        recall = len(true_positives) / len(target_cards) if target_cards else 0
        f1 = 2*(precision * recall) / (precision + recall) if (precision + recall)>0 else 0
        
        precision_at_k = {}
        for k in [5, 10, 20, 50]:
            if k <= len(recommended_ids):
                hits_at_k = len(set(recommended_ids[:k]).intersection(set(target_cards)))
                precision_at_k[k] = hits_at_k / k
            else:
                precision_at_k[k] = 0
        
        reciprocal_ranks = []
        for target_id in target_cards:
            if target_id in recommended_ids:
                rank = recommended_ids.index(target_id) + 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0)
        
        mrr = sum(reciprocal_ranks) / len(target_cards) if target_cards else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'precision_at_k': precision_at_k,
            'mrr': mrr,
            'true_positives': len(true_positives),
            'recommendations': len(recommended_ids),
            'target_cards': len(target_cards),
            'time_taken': recommendation_time
        }
    
    def evaluate_params(self, 
                        factors_list=[100, 200],
                        regularization_list=[0.1, 1, 2],
                        alpha_list=[10, 20, 30],
                        iterations_list=[15, 20, 25],
                        seed_percentages=[0.0, 0.2, 0.4],
                        recommendation_limit=100,
                        model_deck_limit=1000):
        
        all_decks = self.get_all_deck_ids()
        
        # gets a sample of decks to evaluate
        if len(all_decks) > self.num_decks:
            eval_decks = random.sample(all_decks, self.num_decks)
        else:
            eval_decks = all_decks
            
        results = {}
            
        # split the decks based on seed percentages
        deck_splits = {}
        for seed_pct in seed_percentages:
            print(f"\n##### Preparing decks with seed perc: {seed_pct*100:.1f}% #####")
            
            deck_splits[seed_pct] = []
            # splitting each deck into target and seed cards
            for i, deck_id in enumerate(eval_decks):
                print(f"Splitting deck {i+1}/{len(eval_decks)}: {deck_id}")
                
                # Split the deck into seed and target
                seed_cards, target_cards, commander_id = self.split_deck(deck_id, seed_pct)
                if len(target_cards) < 5:
                    print(f"Skipping deck {deck_id} - too few target cards ({len(target_cards)})")
                    continue
                
                deck_splits[seed_pct].append({
                    'deck_id': deck_id,
                    'seed_cards': seed_cards,
                    'target_cards': target_cards,
                    'commander_id': commander_id
                })
        
        # test parameter configurations
        for factors in factors_list:
            for reg in regularization_list:
                for alpha in alpha_list:
                    for iterations in iterations_list:
                        config_key = f"factors={factors},reg={reg},alpha={alpha},iter={iterations}"
                        results[config_key] = {}
                        
                        print(f"\n##### Testing config: {config_key} #####")

                        # initialize recommender and build model w config set
                        recommender = ALSRecommender(self.db)
                        recommender.num_factors = factors
                        recommender.regularization = reg
                        recommender.alpha = alpha
                        recommender.iterations = iterations
                        recommender.build_model(deck_limit=model_deck_limit)
                        
                        # test each seed percentage against the model
                        for seed_pct in seed_percentages:
                            print(f"\n~~~ Evaluating with seed percentage: {seed_pct*100:.1f}% ~~~")
                            
                            split_decks = deck_splits[seed_pct]
                            eval_results = []
                            
                            for i, deck_data in enumerate(split_decks):
                                deck_id = deck_data['deck_id']
                                print(f"Evaluating deck {i+1}/{len(split_decks)}: {deck_id}")
                                
                                seed_cards = deck_data['seed_cards']
                                target_cards = deck_data['target_cards']
                                commander_id = deck_data['commander_id']
                                
                                metrics = self.evaluate_recommender(
                                    recommender,
                                    commander_id, 
                                    seed_cards, 
                                    target_cards,
                                    limit=recommendation_limit
                                )
                                eval_results.append({
                                    'deck_id': deck_id,
                                    'metrics': metrics
                                })
                            
                            avg_metrics = self.calc_metric_avg([r['metrics'] for r in eval_results])
                            results[config_key][seed_pct] = avg_metrics
                            
                            print(f"Results for {config_key} with {seed_pct*100:.1f}% seed:")
                            print(f"-- F1: {avg_metrics['f1']:.4f}")
                            print(f"-- MRR: {avg_metrics['mrr']:.4f}")
                            print(f"-- P@5: {avg_metrics['precision_at_k'][5]:.4f}")
                            print(f"-- Time: {avg_metrics['time_taken']:.4f}s")
                            
        return results
    
    def calc_metric_avg(self, metrics_list):
        if not metrics_list:
            return {}
            
        avg_metrics = {
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'mrr': 0,
            'precision_at_k': {5: 0, 10: 0, 20: 0, 50: 0},
            'true_positives': 0,
            'recommendations': 0,
            'target_cards': 0,
            'time_taken': 0
        }
        
        # sum for each metric
        for metrics in metrics_list:
            for key in ['precision', 'recall', 'f1', 'mrr', 'true_positives', 'recommendations', 'target_cards', 'time_taken']:
                avg_metrics[key] += metrics[key]
            
            for k in avg_metrics['precision_at_k'].keys():
                avg_metrics['precision_at_k'][k] += metrics['precision_at_k'][k]
        
        # calculating avg for each metric
        num_entries = len(metrics_list)
        for key in ['precision', 'recall', 'f1', 'mrr', 'true_positives', 'recommendations', 'target_cards', 'time_taken']:
            avg_metrics[key] /= num_entries
        
        for k in avg_metrics['precision_at_k'].keys():
            avg_metrics['precision_at_k'][k] /= num_entries
        
        return avg_metrics
    
    def get_all_deck_ids(self):
        with self.db.get_connection() as conn:
            from sqlalchemy import text
            query = text("SELECT deck_id FROM decks")
            result = conn.execute(query)
            return [row[0] for row in result]

    def output_evaluation_results(self, results):

        print("\n##### ALS PARAMETER RESULTS #####")
        
        configurations = list(results.keys())
        seed_percentages = list(results[configurations[0]].keys())
        
        f1_data = []
        mrr_data = []
        p5_data = []
        p10_data = []
        p20_data = []
        p50_data = []
        time_data = []
        
        for config in configurations:
            params = dict(param.split('=') for param in config.split(','))
            factors = int(params['factors'])
            reg = float(params['reg'])
            alpha = int(params['alpha'])
            iterations = int(params['iter']) if 'iter' in params else 20
            
            f1_row = {'Factors': factors, 'Reg': reg, 'Alpha': alpha, 'Iter': iterations}
            mrr_row = {'Factors': factors, 'Reg': reg, 'Alpha': alpha, 'Iter': iterations}
            p5_row = {'Factors': factors, 'Reg': reg, 'Alpha': alpha, 'Iter': iterations}
            p10_row = {'Factors': factors, 'Reg': reg, 'Alpha': alpha, 'Iter': iterations}
            p20_row = {'Factors': factors, 'Reg': reg, 'Alpha': alpha, 'Iter': iterations}
            p50_row = {'Factors': factors, 'Reg': reg, 'Alpha': alpha, 'Iter': iterations}
            time_row = {'Factors': factors, 'Reg': reg, 'Alpha': alpha, 'Iter': iterations}
            
            for seed_pct in seed_percentages:
                metrics = results[config][seed_pct]
                f1_row[f'Seed {seed_pct*100:.0f}%'] = metrics['f1']
                mrr_row[f'Seed {seed_pct*100:.0f}%'] = metrics['mrr']
                p5_row[f'Seed {seed_pct*100:.0f}%'] = metrics['precision_at_k'][5]
                p10_row[f'Seed {seed_pct*100:.0f}%'] = metrics['precision_at_k'][10]
                p20_row[f'Seed {seed_pct*100:.0f}%'] = metrics['precision_at_k'][20]
                p50_row[f'Seed {seed_pct*100:.0f}%'] = metrics['precision_at_k'][50]
                time_row[f'Seed {seed_pct*100:.0f}%'] = metrics['time_taken']
            
            f1_data.append(f1_row)
            mrr_data.append(mrr_row)
            p5_data.append(p5_row)
            p10_data.append(p10_row)
            p20_data.append(p20_row)
            p50_data.append(p50_row)
            time_data.append(time_row)
        
        f1_df = pd.DataFrame(f1_data)
        mrr_df = pd.DataFrame(mrr_data)
        p5_df = pd.DataFrame(p5_data)
        p10_df = pd.DataFrame(p10_data)
        p20_df = pd.DataFrame(p20_data)
        p50_df = pd.DataFrame(p50_data)
        time_df = pd.DataFrame(time_data)
        
        print("\nF1 Score Summary:")
        print(f1_df.to_string(index=False))
        print("\nMRR Summary:")
        print(mrr_df.to_string(index=False))
        print("\nPrecision@5 Summary:")
        print(p5_df.to_string(index=False))
        print("\nPrecision@10 Summary:")
        print(p10_df.to_string(index=False))
        print("\nPrecision@20 Summary:")
        print(p20_df.to_string(index=False))
        print("\nPrecision@50 Summary:")
        print(p50_df.to_string(index=False))
        print("\nExecution Time Summary:")
        print(time_df.to_string(index=False))
        
        # finds best config for each metric
        print("\n##### BEST CONFIGURATIONS #####")
        
        for seed_pct in seed_percentages:
            seed_str = f'Seed {seed_pct*100:.0f}%'
            
            best_f1_idx = f1_df[seed_str].idxmax()
            best_f1_config = f1_df.iloc[best_f1_idx]
            
            best_mrr_idx = mrr_df[seed_str].idxmax()
            best_mrr_config = mrr_df.iloc[best_mrr_idx]
            
            best_p5_idx = p5_df[seed_str].idxmax()
            best_p5_config = p5_df.iloc[best_p5_idx]

            best_p10_idx = p10_df[seed_str].idxmax()
            best_p10_config = p10_df.iloc[best_p10_idx]
                        
            best_p20_idx = p20_df[seed_str].idxmax()
            best_p20_config = p20_df.iloc[best_p20_idx]
                        
            best_p50_idx = p50_df[seed_str].idxmax()
            best_p50_config = p50_df.iloc[best_p50_idx]
            
            print(f"\nBest for {seed_str}:")
            print(f"Best F1 Score: {best_f1_config[seed_str]:.4f} with Factors={best_f1_config['Factors']}, Reg={best_f1_config['Reg']}, Alpha={best_f1_config['Alpha']}, Iter={best_f1_config['Iter']}")
            print(f"Best MRR: {best_mrr_config[seed_str]:.4f} with Factors={best_mrr_config['Factors']}, Reg={best_mrr_config['Reg']}, Alpha={best_mrr_config['Alpha']}, Iter={best_mrr_config['Iter']}")
            print(f"Best P@5: {best_p5_config[seed_str]:.4f} with Factors={best_p5_config['Factors']}, Reg={best_p5_config['Reg']}, Alpha={best_p5_config['Alpha']}, Iter={best_p5_config['Iter']}")
            print(f"Best P@10: {best_p10_config[seed_str]:.4f} with Factors={best_p10_config['Factors']}, Reg={best_p10_config['Reg']}, Alpha={best_p10_config['Alpha']}, Iter={best_p10_config['Iter']}")
            print(f"Best P@20: {best_p20_config[seed_str]:.4f} with Factors={best_p20_config['Factors']}, Reg={best_p20_config['Reg']}, Alpha={best_p20_config['Alpha']}, Iter={best_p20_config['Iter']}")
            print(f"Best P@50: {best_p50_config[seed_str]:.4f} with Factors={best_p50_config['Factors']}, Reg={best_p50_config['Reg']}, Alpha={best_p50_config['Alpha']}, Iter={best_p50_config['Iter']}")
        

def main():
    evaluator = ALSRecommenderEvaluator(num_decks=10)

    # tests multiple configurations
    results = evaluator.evaluate_params(
        factors_list=[250],
        regularization_list=[0.5, 1, 2],
        alpha_list=[10, 20, 30],
        iterations_list=[20, 25],
        seed_percentages=[0.4],
        recommendation_limit=100,
        model_deck_limit=1000
    )

    evaluator.output_evaluation_results(results)

if __name__ == "__main__":
    main()