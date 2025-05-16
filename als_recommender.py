import os

# Fixes openblas warning, from creator's github:
# For systems using OpenBLAS, I highly recommend setting 'export OPENBLAS_NUM_THREADS=1'.
# This disables its internal multithreading ability, which leads to substantial
# speedups for this package.
# Likewise for Intel MKL, setting 'export MKL_NUM_THREADS=1' should also be set.
# https://github.com/benfred/implicit/
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from mtg_database import MTGDatabase
import pickle

class ALSRecommender:
    def __init__(self, db):
        self.db = db
        self.basic_lands = ['Island', 'Swamp', 'Mountain', 'Plains', 'Forest']
        
        # default parameters
        self.num_factors = 150
        self.regularization = 0.01
        self.alpha = 40
        self.iterations = 20
        self.als_model = None
        self.user_factors = None
        self.card_ids = None
        self.card_id_to_index = None
        self.card_popularity = {}
        self.color_identity_cache = {}
        self.model_built = False
    
    def is_basic_land(self, card):
        if card['card_name'] in self.basic_lands:
            return True
        
        return False
    
    def build_model(self, deck_limit=1000, deck_ids=None):
        print(f"Building model with {self.num_factors} factors, reg={self.regularization}, alpha={self.alpha}")
        
        # Get deck IDs to use
        if deck_ids is not None:
            all_deck_ids = deck_ids[:deck_limit] if deck_limit < len(deck_ids) else deck_ids
            print(f"Using {len(all_deck_ids)} decks for model building")
        else:
            # Get random deck IDs from database
            all_deck_ids = self._get_all_deck_ids(limit=deck_limit)
            print(f"Using {len(all_deck_ids)} random decks for model building")
        
        self.deck_ids = all_deck_ids
        
        # get card ids for selected decks
        all_cards = set()
        deck_cards_map = {}
        
        for deck_id in all_deck_ids:
            deck_cards = self.db.get_deck_cards(deck_id)
            deck_cards_map[deck_id] = deck_cards
            
            for card in deck_cards:
                card_id = card['card_id']
                if not self.is_basic_land(card):
                    all_cards.add(card_id)
                    
                    # Track card popularity
                    if card_id not in self.card_popularity:
                        self.card_popularity[card_id] = {'count': 0, 'decks': 0}
                    
                    self.card_popularity[card_id]['count'] += card.get('count', 1)
                    self.card_popularity[card_id]['decks'] += 1
        
        # store card IDs and create mapping
        self.card_ids = list(all_cards)
        self.card_id_to_index = {card_id: i for i, card_id in enumerate(self.card_ids)}
        
        # build s deck-card matrix
        rows, cols, data = [], [], []
        
        for deck_idx, deck_id in enumerate(all_deck_ids):
            deck_cards = deck_cards_map[deck_id]
            for card in deck_cards:
                card_id = card['card_id']
                if not self.is_basic_land(card) and card_id in self.card_id_to_index:
                    card_idx = self.card_id_to_index[card_id]
                    confidence = card.get('count', 1) * self.alpha
                    rows.append(deck_idx)
                    cols.append(card_idx)
                    data.append(confidence)
        
        # Create a sparse matrix
        matrix_shape = (len(all_deck_ids), len(self.card_ids))
        deck_card_matrix = csr_matrix((data, (rows, cols)), shape=matrix_shape)
        
        # fit als model
        self.als_model = AlternatingLeastSquares(
            factors=self.num_factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=29
        )
        
        self.als_model.fit(deck_card_matrix)
        
        # store card and deck factors
        self.card_factors = self.als_model.item_factors
        self.deck_factors = self.als_model.user_factors
        
        # map deck id to index
        self.deck_id_to_index = {deck_id: i for i, deck_id in enumerate(all_deck_ids)}
        
        # sort cards by popularity for fallback recommendations
        self.sorted_cards_by_popularity = sorted(
            self.card_popularity.items(),
            key=lambda x: x[1]['decks'],
            reverse=True
        )

        self.model_built = True
        print("ALS MODEL BUILT")


    def save_model(self, file_path="models/als_final_model.pkl"):
        if not self.model_built:
            print("Cannot save model, model not built")
            return False
        
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            model_data = {
                'als_model': self.als_model,
                'card_factors': self.card_factors,
                'deck_factors': self.deck_factors,
                'card_ids': self.card_ids,
                'card_id_to_index': self.card_id_to_index,
                'card_popularity': self.card_popularity,
                'deck_id_to_index': self.deck_id_to_index,
                'deck_ids': self.deck_ids,
                'sorted_cards_by_popularity': self.sorted_cards_by_popularity,
                'color_identity_cache': self.color_identity_cache,
                'num_factors': self.num_factors,
                'regularization': self.regularization,
                'alpha': self.alpha,
                'iterations': self.iterations
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)
                
            print(f"Model saved to {file_path}")
            return True
            
        except Exception as e:
            print(f"Error saving model to {file_path}: {str(e)}")
            return False


    def load_model(self, file_path="models/als_final_model.pkl"):
        try:
            if not os.path.exists(file_path):
                print(f"Model file {file_path} not found")
                return False
                
            print(f"Loading model from {file_path}")
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.als_model = model_data.get('als_model')
            self.card_factors = model_data.get('card_factors')
            self.deck_factors = model_data.get('deck_factors')
            self.card_ids = model_data.get('card_ids')
            self.card_id_to_index = model_data.get('card_id_to_index')
            self.card_popularity = model_data.get('card_popularity')
            self.deck_id_to_index = model_data.get('deck_id_to_index')
            self.deck_ids = model_data.get('deck_ids')
            self.sorted_cards_by_popularity = model_data.get('sorted_cards_by_popularity')
            self.color_identity_cache = model_data.get('color_identity_cache', {})
            self.num_factors = model_data.get('num_factors', self.num_factors)
            self.regularization = model_data.get('regularization', self.regularization)
            self.alpha = model_data.get('alpha', self.alpha)
            self.iterations = model_data.get('iterations', self.iterations)
            
            self.model_built = True
            print(f"Model loaded: factors={self.num_factors}, reg={self.regularization}, alpha={self.alpha}, iter={self.iterations}")
            return True            
        except Exception as e:
            print(f"Error loading model from {file_path}: {str(e)}")
            return False


    def _get_all_deck_ids(self, limit=1000):
        with self.db.get_connection() as conn:
            from sqlalchemy import text
            query = text(f"SELECT deck_id FROM decks LIMIT {limit}")
            result = conn.execute(query)
            return [row[0] for row in result]


    def _get_card_color_identity(self, card_id):
        if card_id in self.color_identity_cache:
            return self.color_identity_cache[card_id]
        color_identity_data = self.db.get_card_color_identity(card_id)
        color_identity = [c['color_id'] for c in color_identity_data]
        # caches the color identity to prevent having to requery DB every time
        self.color_identity_cache[card_id] = color_identity
        return color_identity


    def _predict_card_scores(self, deck_vector):
        # use dot product to get card scores
        scores = deck_vector.dot(self.card_factors.T)
        card_scores_dict = {self.card_ids[i]: float(score) for i, score in enumerate(scores)}
        
        return card_scores_dict


    def _get_deck_vector_from_cards(self, existing_cards, commander_id):
        # return nothing if no commander or card data
        if not existing_cards and commander_id not in self.card_id_to_index:
            return np.zeros(self.num_factors)
        
        # Find decks with the same commander
        similar_decks = self.db.get_decks_by_commander(commander_id)
        
        # Check if we have this commander in our model
        if similar_decks and any(deck['deck_id'] in self.deck_id_to_index for deck in similar_decks):
            # Average the deck vectors for decks with this commander
            vectors = []
            for deck in similar_decks:
                deck_id = deck['deck_id']
                if deck_id in self.deck_id_to_index:
                    deck_idx = self.deck_id_to_index[deck_id]
                    vectors.append(self.deck_factors[deck_idx])
            
            if vectors:
                base_vector = np.mean(vectors, axis=0)
            else:
                base_vector = np.zeros(self.num_factors)
        else:
            base_vector = np.zeros(self.num_factors)
        
        # if we have existing cards in our model, adjust the vector
        known_cards = [card_id for card_id in existing_cards if card_id in self.card_id_to_index]
        if known_cards:
            # average the card factors for the existing cards
            card_vectors = []
            for card_id in known_cards:
                card_idx = self.card_id_to_index[card_id]
                card_vectors.append(self.card_factors[card_idx])
            
            if card_vectors:
                card_vector = np.mean(card_vectors, axis=0)
                
                # combine with base vector (weighted average)
                if np.any(base_vector != 0):
                    # Use 30% from similar decks, 70% from cards if we have both
                    combined_vector = 0.3 * base_vector + 0.7 * card_vector
                else:
                    combined_vector = card_vector
                    
                return combined_vector
        
        return base_vector


    def get_recommendations(self, commander_id, existing_cards=None, limit=20):
        if existing_cards is None:
            existing_cards = []
        if not self.model_built:
            self.build_model()
        # get commander color identity for filtering
        commander_colors = self._get_card_color_identity(commander_id)
        # get als recs
        als_recommendations = self._get_als_recs(
            commander_id, existing_cards, commander_colors, limit
        )

        all_recommendations = als_recommendations    
        # sort recs by score
        sorted_recommendations = sorted(
            all_recommendations, 
            key=lambda x: x['score'], 
            reverse=True
        )[:limit]
        return sorted_recommendations


    def _get_als_recs(self, commander_id, existing_cards, commander_colors, limit):
        # get deck vector
        deck_vector = self._get_deck_vector_from_cards(existing_cards, commander_id)

        if np.all(deck_vector == 0):
            return []
        
        # predict card scores
        card_scores = self._predict_card_scores(deck_vector)
        
        # filter out cards already in the deck list and the commander
        filtered_scores = {
            card_id: score for card_id, score in card_scores.items()
            if card_id not in existing_cards and card_id != commander_id
        }
        
        # get cards data and filter by color identity
        recommendations = []
        for card_id, score in sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True):
            if len(recommendations) >= limit * 2:
                break
                
            card_data = self.db.get_card_by_id(card_id)
            
            if not card_data or self.is_basic_land(card_data):
                continue

            card_colors = self._get_card_color_identity(card_id)

            if card_colors and any(color not in commander_colors and color != 'C' for color in card_colors):
                continue

            card_data['score'] = float(score)
            recommendations.append(card_data)
        
        return recommendations


def create_final_model():
    db = MTGDatabase()
    
    recommender = ALSRecommender(db)
    recommender.num_factors = 600     
    recommender.regularization = 2.25 
    recommender.alpha = 10            
    recommender.iterations = 25       
    
    print("Building final ALS model")
    recommender.build_model(deck_limit=200000)

    print("Saving model")
    recommender.save_model("models/als_final_model.pkl")
    
    print(f"Model built with {len(recommender.card_ids)} unique cards.")
    print(f"Model built with {len(recommender.deck_ids)} decks.")
    
    db.close()
    print("DB connection closed.")

if __name__ == "__main__":
    create_final_model()