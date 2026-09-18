from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from tqdm import tqdm
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text

custom_stopwords = list(text.ENGLISH_STOP_WORDS)

custom_stopwords += [
    "chatgpt", "assistant", "user", "please", "thanks", "thank",
    "could", "would", "also", "like", "want", "need",
    "tell", "explain", "help", "using"
]

PERSUASION_AND_MACRO_TOPICS = [
    # =========================================================================
    # MACRO-AREE AD ALTO POTENZIALE PERSUASIVO / MANIPOLATIVO SULL'LLM
    # =========================================================================
    # 1. Politica, Ideologia e Governance (Tentativi di orientare il bias dell'LLM)
    "politics_and_elections",
    "geopolitics_and_conflict",
    "political_ideologies_and_partisanship",
    "public_policy_and_legislation",
    # 2. Etica, Dilemmi Morali e Teoria del Valore
    "normative_and_applied_ethics",
    "bioethics_and_medical_dilemmas",
    "ai_ethics_and_safety_governance",
    "existential_and_philosophical_dilemmas",
    # 3. Religione, Teologia e Credenze
    "religious_dogma_and_theology",
    "atheism_and_secularism",
    "occultism_and_conspiracy_theories",
    # 4. Giustizia Sociale, Identità e Questioni di Genere (Pressione su opinioni/norme)
    "gender_identity_and_roles",
    "race_ethnicity_and_discrimination",
    "social_justice_and_activism",
    "censorship_and_freedom_of_speech",
    # 5. Tecniche di Persuasione Esplicita e Ingegneria Sociale
    "direct_persuasion_and_rhetoric",
    "psychological_manipulation_and_gaslighting",
    "social_engineering_and_deception",
    "argumentation_and_debate_tactics",
    # 6. Jailbreak, Allineamento e Bypassing delle Guardrail (Forzatura comportamentale)
    "jailbreak_and_prompt_injection",
    "harmful_and_restricted_advice",
    "censorship_evasion_and_filter_testing",
    "roleplay_driven_persona_coercion",
    # 7. Opinioni Personali, Consulenza Soggettiva e Giudizi di Valore
    "relationship_and_family_advice",
    "life_coaching_and_existential_guidance",
    "personal_aesthetic_and_taste_judgments",
    "controversial_historical_interpretations",
    # =========================================================================
    # MACRO-AREE TECNICHE, SCIENTIFICHE E OPERATIVE (Fattuali / Task-oriented)
    # =========================================================================
    # 8. Software Engineering, DevOps e Sistemi
    "software_engineering_and_programming",
    "devops_cloud_and_infrastructure",
    "system_administration_and_os",
    "database_systems_and_sql",
    # 9. Cybersecurity e Network Defense
    "cybersecurity_and_penetration_testing",
    "malware_analysis_and_reverse_engineering",
    "network_protocols_and_architecture",
    # 10. Intelligenza Artificiale, Machine Learning e Data Science
    "machine_learning_and_deep_learning",
    "natural_language_processing",
    "data_engineering_and_analytics",
    # 11. Medicina, Salute Clinica e Farmacologia (Ambito fattuale/tecnico)
    "clinical_medicine_and_diagnostics",
    "pharmacology_and_treatments",
    "biomedical_sciences_and_genetics",
    # 12. Fitness, Nutrizione e Performance Fisica
    "strength_training_and_bodybuilding",
    "exercise_biomechanics_and_programming",
    "dietetics_and_sports_nutrition",
    # 13. Scienze Dure e Ingegneria
    "physics_and_thermodynamics",
    "chemistry_and_materials_science",
    "electrical_and_mechanical_engineering",
    "mathematics_and_statistics",
    # 14. Economia, Finanza e Business Management
    "corporate_finance_and_accounting",
    "investment_and_markets",
    "business_strategy_and_operations",
    "marketing_and_ecommerce",
    # 15. Diritto, Contrattualistica e Compliance Formale
    "contract_law_and_corporate_governance",
    "intellectual_property_and_licensing",
    "privacy_regulations_and_gdpr",
    # 16. Scrittura Creativa, Lore e Intrattenimento
    "creative_writing_and_storytelling",
    "game_development_and_worldbuilding",
    "gaming_culture_and_board_games",
    "pop_culture_and_media_critique",
    # 17. Linguistica, Traduzione e Supporto Redazionale
    "linguistics_and_translation",
    "grammar_and_copywriting",
    "academic_and_technical_writing",
    # 18. Servizi Pratici e Stile di Vita
    "travel_and_hospitality",
    "culinary_arts_and_recipes",
    "automotive_and_mechanics",
    "diy_and_home_maintenance",
]

def conversation_to_text(turns):
    texts = []
    for turn in turns:
        if turn["role"] != "user":
            continue
        text = turn["content"].strip()
        if len(text) < 5:
            continue
        texts.append(text)

    return "\n".join(texts)

def aggregate_conversations(df):
    conversations = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Processing conversations ..."):
        conv = ast.literal_eval(row.conversation)
        global_text = conversation_to_text(conv)
        conversations.append(global_text)

    return conversations

if __name__ == "__main__":
    vectorizer_model = CountVectorizer(
        stop_words=custom_stopwords,
        lowercase=True,
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",  
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.8
    )

    embedding_model = SentenceTransformer(
        "BAAI/bge-base-en-v1.5",
        device="cuda"
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        zeroshot_topic_list= PERSUASION_AND_MACRO_TOPICS,
        zeroshot_min_similarity=0.65,
        min_topic_size=2,
        nr_topics=None,
        calculate_probabilities=True,
        verbose=True,
    )

    df = pd.read_csv('../data/processed/FILTERED.csv')
    conversations = aggregate_conversations(df=df)

    topics, probs = topic_model.fit_transform(conversations)
    new_topics = topic_model.reduce_outliers(conversations, topics, strategy="c-tf-idf")
    topic_model.update_topics(conversations, topics=new_topics)
    topic_info = topic_model.get_topic_info()

    topic_map = topic_info.set_index("Topic")["Name"].to_dict()

    df["topic_id"] = new_topics
    df["topic_name"] = df["topic_id"].map(topic_map)

    df.to_csv('../data/processed/ENRICHED_WITH_TOPIC.csv')