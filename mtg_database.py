import configparser
from sqlalchemy import create_engine, text
from contextlib import contextmanager

class MTGDatabase:
    def __init__(self, config_file='db_config.ini'):
        try:
            self.config = configparser.ConfigParser()
            self.config.read(config_file)
            
            cnx = f"mysql+mysqlconnector://{self.config['mysql']['user']}:{self.config['mysql']['password']}@{self.config['mysql']['host']}/{self.config['mysql']['database']}"
            self.engine = create_engine(cnx)
            print("DB connection successful")
        except Exception as e:
            print(f"Error connecting to DB: {str(e)}")
            raise
    
    
    @contextmanager
    def get_connection(self):
        connection = self.engine.connect()
        try:
            yield connection
        finally:
            connection.close()
    

    def close(self):
        if hasattr(self, 'engine') and self.engine:
            self.engine.dispose()
            print("DB connection closed.")
    

    def get_card_by_id(self, card_id):
        with self.get_connection() as conn:
            query = text("SELECT * FROM cards WHERE card_id = :card_id")
            result = conn.execute(query, {"card_id": card_id})
            row = result.fetchone()
            if row:
                return dict(zip(result.keys(), row))
            return None
    

    def get_card_by_name(self, card_name):
        with self.get_connection() as conn:
            query = text("SELECT * FROM cards WHERE card_name = :card_name")
            result = conn.execute(query, {"card_name": card_name})
            row = result.fetchone()
            if row:
                return dict(zip(result.keys(), row))
            return None
    

    def get_deck_cards(self, deck_id):
        with self.get_connection() as conn:
            query = text("""
                SELECT c.*, dc.count
                FROM deck_cards dc
                JOIN cards c ON dc.card_id = c.card_id
                WHERE dc.deck_id = :deck_id
            """)
            result = conn.execute(query, {"deck_id": deck_id})
            rows = result.fetchall()
            return [dict(zip(result.keys(), row)) for row in rows]
    
    
    def get_card_colors(self, card_id):
        with self.get_connection() as conn:
            query = text("""
                SELECT c.color_id, c.color_name
                FROM card_colors cc
                JOIN colors c ON cc.color_id = c.color_id
                WHERE cc.card_id = :card_id
            """)
            result = conn.execute(query, {"card_id": card_id})
            rows = result.fetchall()
            return [dict(zip(result.keys(), row)) for row in rows]
    

    def get_card_color_identity(self, card_id):
        with self.get_connection() as conn:
            query = text("""
                SELECT c.color_id, c.color_name
                FROM card_color_identity cci
                JOIN colors c ON cci.color_id = c.color_id
                WHERE cci.card_id = :card_id
            """)
            result = conn.execute(query, {"card_id": card_id})
            rows = result.fetchall()
            return [dict(zip(result.keys(), row)) for row in rows]
       

    def get_commander_info(self, card_id):
        with self.get_connection() as conn:
            query = text("SELECT * FROM commanders WHERE card_id = :card_id")
            result = conn.execute(query, {"card_id": card_id})
            row = result.fetchone()
            if row:
                return dict(zip(result.keys(), row))
            return None
    

    def get_deck_info(self, deck_id):
        with self.get_connection() as conn:
            query = text("""
                SELECT d.*, 
                       pc.card_name as primary_commander_name,
                       sc.card_name as secondary_commander_name
                FROM decks d
                JOIN cards pc ON d.primary_commander_id = pc.card_id
                LEFT JOIN cards sc ON d.secondary_commander_id = sc.card_id
                WHERE d.deck_id = :deck_id
            """)
            result = conn.execute(query, {"deck_id": deck_id})
            row = result.fetchone()
            if row:
                return dict(zip(result.keys(), row))
            return None


    def get_decks_by_commander(self, commander_id):
        with self.get_connection() as conn:
            query = text("""
                SELECT d.*, 
                       pc.card_name as primary_commander_name,
                       sc.card_name as secondary_commander_name
                FROM decks d
                JOIN cards pc ON d.primary_commander_id = pc.card_id
                LEFT JOIN cards sc ON d.secondary_commander_id = sc.card_id
                WHERE d.primary_commander_id = :commander_id 
                OR d.secondary_commander_id = :commander_id
            """)
            result = conn.execute(query, {"commander_id": commander_id})
            rows = result.fetchall()
            return [dict(zip(result.keys(), row)) for row in rows]
    
    
    def get_card_types(self, card_id):
        with self.get_connection() as conn:
            query = text("""
                SELECT ct.*, t.type_name
                FROM card_types ct
                JOIN types t ON ct.type_id = t.type_id
                WHERE ct.card_id = :card_id
            """)
            result = conn.execute(query, {"card_id": card_id})
            rows = result.fetchall()
            return [dict(zip(result.keys(), row)) for row in rows]