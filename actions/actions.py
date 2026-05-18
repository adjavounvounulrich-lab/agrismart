# =============================================================================
# ACTIONS.PY — Actions custom RASA pour AgriSmart
# =============================================================================
# Ce fichier contient toutes les actions Python que RASA exécute
# en réponse aux intentions de l'utilisateur.
#
# Chaque action est une classe Python qui hérite de Action.
# Elle a deux méthodes obligatoires :
#   - name()  : retourne le nom exact déclaré dans domain.yml
#   - run()   : contient toute la logique de l'action
#
# ARCHITECTURE IMPORTANTE :
# Toute la logique ML (prédiction XGBoost, calcul SHAP, waterfall) est
# entièrement gérée dans Streamlit (app.py). RASA n'a PAS besoin de
# recharger les fichiers .pkl ni de recalculer quoi que ce soit.
#
# Streamlit transmet les résultats à RASA via l'API tracker/events avant
# chaque message (culture_predite, top3_shap, confiance, paramètres).
# Les actions ci-dessous lisent ces slots et formatent des réponses texte.
#
# RÉPONSES PRÉCISES ET CONTEXTUELLES :
# Les actions action_pratiques_agricoles et action_infos_culture détectent
# des mots-clés dans le message de l'utilisateur pour retourner uniquement
# la section pertinente (semis, engrais, récolte...) au lieu de la fiche
# complète. Cela rend le chatbot plus conversationnel et moins verbeux.
# =============================================================================

import random
from typing import Any, Text, Dict, List

# Seules importations nécessaires — fournies par rasa-sdk déjà installé
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet


# =============================================================================
# CONSTANTES — Cultures béninoises et descriptions contextualisées
# Ces listes travaillent avec les noms béninois (Sorgho/Fonio/Mil)
# car Streamlit a déjà appliqué la correspondance avant d'envoyer
# les slots à RASA.
# =============================================================================

CULTURES = [
    "Riz", "Maïs", "Sorgho", "Coton", "Canne à sucre",
    "Fonio", "Mil", "Pomme de terre", "Légumineuses", "Tomate"
]

DESCRIPTIONS = {
    "Riz":           "Céréale semi-aquatique, idéale pour les zones humides du Bénin",
    "Maïs":          "Première céréale du Bénin, adaptée aux zones tropicales",
    "Sorgho":        "Céréale résistante à la sécheresse, cultivée dans le Nord-Bénin",
    "Coton":         "Culture de rente stratégique, pilier de l'économie béninoise",
    "Canne à sucre": "Culture sucrière tropicale, exige beaucoup d'eau",
    "Fonio":         "Céréale ancestrale africaine, cultivée dans l'Atacora (Nord-Bénin)",
    "Mil":           "Céréale semi-aride des zones Nord-Bénin (Borgou, Atacora, Alibori)",
    "Pomme de terre":"Tubercule productif, préfère les sols frais et bien drainés",
    "Légumineuses":  "Soja, Niébé ou Arachide selon votre zone — cultures majeures au Bénin",
    "Tomate":        "Culture maraîchère à haute valeur ajoutée, cultivée partout au Bénin",
}


# =============================================================================
# DÉTECTION D'ASPECTS — Mots-clés pour les réponses ciblées
# Quand l'utilisateur pose une question sur un aspect spécifique (période,
# engrais, récolte...), on retourne uniquement cette section de la fiche.
# =============================================================================

# Dictionnaire des aspects : chaque clé est un tag interne,
# la valeur est la liste de mots-clés qui déclenchent cet aspect.
ASPECTS_MOTS_CLES = {
    "labour": [
        "labour", "labourer", "préparer le sol", "préparation",
        "bêcher", "creuser", "ameublir", "terrain", "avant semer"
    ],
    "semis": [
        "semis", "semer", "planter", "quand planter", "période",
        "calendrier", "écartement", "poquet", "graine", "densité",
        "à quelle période", "quelle saison pour planter", "profondeur"
    ],
    "entretien": [
        "entretien", "entretenir", "maladie", "ravageur", "insecte",
        "sarclage", "désherbage", "parasite", "champignon", "traitement",
        "phytosanitaire", "pest", "attaque", "protection"
    ],
    "engrais": [
        "engrais", "fertiliser", "fertilisation", "npk", "urée",
        "phosphore", "azote", "potassium", "fumure", "amendement",
        "fertilité", "nutriment", "dose d'engrais"
    ],
    "recolte": [
        "récolte", "récolter", "quand récolter", "maturité", "harvester",
        "cueillette", "couper", "ramasser", "rendement", "production finale"
    ],
    "eau": [
        "eau", "arrosage", "arroser", "irrigation", "irriguer",
        "humidité", "pluie", "pluviométrie", "sécheresse", "besoin en eau"
    ],
}


def detecter_aspect(message):
    """
    Analyse le message de l'utilisateur et retourne l'aspect demandé.

    Parcourt les mots-clés de chaque aspect et retourne le premier match.
    Retourne None si aucun aspect spécifique n'est détecté — dans ce cas,
    l'action retourne la fiche complète.

    Paramètres :
        message (str) : texte du message en minuscules

    Retourne :
        str | None : clé d'aspect ("semis", "engrais"...) ou None
    """
    for aspect, mots in ASPECTS_MOTS_CLES.items():
        for mot in mots:
            if mot in message:
                return aspect
    return None


# =============================================================================
# BASE DE CONNAISSANCES AGRICOLES — Pratiques pour chaque culture béninoise
# =============================================================================

PRATIQUES_AGRICOLES = {

    "Maïs": {
        "labour": (
            "🚜 Labour du Maïs\n"
            "• 2 à 3 semaines avant le semis, en début de saison des pluies\n"
            "• Profondeur : 20 à 30 cm (labour profond)\n"
            "• Labour croisé + hersage pour affiner la surface\n"
            "• Objectif : éliminer les adventices et aérer le sol"
        ),
        "semis": (
            "🌱 Semis du Maïs\n"
            "• Dès les premières pluies régulières\n"
            "• Écartement : 80 cm × 40 cm\n"
            "• 2 à 3 grains par poquet, profondeur 3 à 5 cm\n"
            "• Démariage : garder 1 à 2 plants à 15 jours"
        ),
        "entretien": (
            "✂️ Entretien du Maïs\n"
            "• 1er sarclage : 15 à 20 jours après levée\n"
            "• 2ème sarclage + buttage : 35 à 40 jours\n"
            "• Surveiller les foreurs de tiges (chenilles)"
        ),
        "engrais": (
            "🧪 Engrais pour le Maïs\n"
            "• NPK (14-23-14) à 150 kg/ha au semis\n"
            "• Urée 46% à 50 kg/ha à 30 jours\n"
            "• Urée 46% à 50 kg/ha à 45-50 jours\n"
            "• Placer en bande latérale à 5 cm de la plante"
        ),
        "recolte": (
            "🌽 Récolte du Maïs\n"
            "• 90 à 120 jours après semis\n"
            "• Signes : feuilles jaunies, spathes sèches, grains durs\n"
            "• Sécher 5 à 7 jours avant l'égrenage"
        ),
        "eau": (
            "💧 Besoins en eau du Maïs\n"
            "• Pluviométrie optimale : 600 à 1200 mm/cycle\n"
            "• Stade critique : floraison et remplissage des grains\n"
            "• Irrigation si pluviométrie insuffisante en saison sèche\n"
            "• Éviter l'engorgement du sol"
        )
    },

    "Riz": {
        "labour": (
            "🚜 Labour du Riz\n"
            "• Riz pluvial : labour à 15-20 cm, hersage fin\n"
            "• Riz irrigué : labour à 20-25 cm puis mise en boue (puddling)\n"
            "• Niveler le sol pour une répartition uniforme de l'eau"
        ),
        "semis": (
            "🌱 Semis du Riz\n"
            "• En poquet : 20 cm × 20 cm, 3 à 5 grains\n"
            "• Transplantation : plants de 21 jours, 2-3 plants/poquet\n"
            "• Profondeur de semis : 2 à 3 cm"
        ),
        "entretien": (
            "✂️ Entretien du Riz\n"
            "• 2 à 3 sarclages selon pression des adventices\n"
            "• Maintenir 5-10 cm d'eau en période de tallage\n"
            "• Surveiller la pyriculariose (taches sur feuilles)\n"
            "• Drainer 2 semaines avant récolte"
        ),
        "engrais": (
            "🧪 Engrais pour le Riz\n"
            "• NPK (15-15-15) à 150 kg/ha au semis\n"
            "• Urée 50 kg/ha à 21 jours (tallage)\n"
            "• Urée 50 kg/ha à 45 jours (montaison)"
        ),
        "recolte": (
            "🌾 Récolte du Riz\n"
            "• 100 à 130 jours selon variété\n"
            "• Maturité : 85% des grains dorés/jaunis\n"
            "• Sécher jusqu'à 14% d'humidité pour le stockage"
        ),
        "eau": (
            "💧 Besoins en eau du Riz\n"
            "• Riz irrigué : maintenir 5-10 cm d'eau en tallage\n"
            "• Riz pluvial : 1000-2000 mm/cycle minimum\n"
            "• Drainer complètement 2 semaines avant la récolte\n"
            "• Stade critique : épiaison"
        )
    },

    "Sorgho": {
        "labour": (
            "🚜 Labour du Sorgho\n"
            "• Labour à 20-25 cm de profondeur\n"
            "• Sol bien ameubli, adapté aux sols argileux et sablo-argileux\n"
            "• Zones principales : Nord-Bénin (Borgou, Atacora)"
        ),
        "semis": (
            "🌱 Semis du Sorgho\n"
            "• Dès les premières pluies régulières\n"
            "• Écartement : 80 cm × 40 cm\n"
            "• 3 à 5 grains par poquet\n"
            "• Démariage : garder 2 à 3 plants"
        ),
        "entretien": (
            "✂️ Entretien du Sorgho\n"
            "• 2 sarclages : à 20 et 40 jours\n"
            "• Surveiller le striga dans les zones infestées\n"
            "• Très résistant à la sécheresse"
        ),
        "engrais": (
            "🧪 Engrais pour le Sorgho\n"
            "• NPK à 100 kg/ha au semis\n"
            "• Urée à 50 kg/ha à 30 jours\n"
            "• Ne pas sur-fertiliser"
        ),
        "recolte": (
            "🌾 Récolte du Sorgho\n"
            "• 90 à 130 jours selon variété\n"
            "• Couper les panicules à maturité complète\n"
            "• Sécher avant battage et stockage"
        ),
        "eau": (
            "💧 Besoins en eau du Sorgho\n"
            "• Pluviométrie minimale : 400 mm/cycle\n"
            "• Très résistant à la sécheresse — adapté aux zones semi-arides\n"
            "• Stade critique : floraison"
        )
    },

    "Coton": {
        "labour": (
            "🚜 Labour du Coton\n"
            "• En saison sèche (novembre-décembre) — labour précoce\n"
            "• Profondeur : 25 à 30 cm (labour profond obligatoire)\n"
            "• Labour à la charrue + passage de cover-crop"
        ),
        "semis": (
            "🌱 Semis du Coton\n"
            "• Dès les premières pluies régulières (mai-juin)\n"
            "• Écartement : 80 cm × 40 cm\n"
            "• 3 à 4 grains délintés par poquet\n"
            "• Démariage : garder 1 plant vigoureux à 15 jours"
        ),
        "entretien": (
            "✂️ Entretien du Coton\n"
            "• 1er sarclage : 15-20 jours — 2ème : 35-40 jours\n"
            "• 8 traitements phytosanitaires selon calendrier officiel\n"
            "• Surveiller Helicoverpa (chenille de la capsule)\n"
            "• Respecter le calendrier de traitement SONAPRA/AIC"
        ),
        "engrais": (
            "🧪 Engrais pour le Coton\n"
            "• NPKSB (14-18-18-6-1) à 150-200 kg/ha au semis\n"
            "• Urée 50 kg/ha à 30 jours\n"
            "• Urée 50 kg/ha à 50-60 jours"
        ),
        "recolte": (
            "🌿 Récolte du Coton\n"
            "• 150 à 180 jours après semis\n"
            "• Capsules ouvertes, coton blanc bien formé\n"
            "• 2 à 3 passages espacés de 15 jours"
        ),
        "eau": (
            "💧 Besoins en eau du Coton\n"
            "• Pluviométrie optimale : 700-1200 mm/cycle\n"
            "• Stade critique : floraison et fructification\n"
            "• Éviter le stress hydrique pendant la capsulation"
        )
    },

    "Tomate": {
        "labour": (
            "🚜 Labour pour la Tomate\n"
            "• Profondeur 20-25 cm, sol bien ameubli et nivelé\n"
            "• Incorporer 10 t/ha de compost ou fumier\n"
            "• Former des billons ou planches surélevées"
        ),
        "semis": (
            "🌱 Semis et transplantation de la Tomate\n"
            "• Pépinière puis repiquer à 25-30 jours\n"
            "• Écartement : 70 cm × 50 cm\n"
            "• Repiquer le soir, arroser abondamment après"
        ),
        "entretien": (
            "✂️ Entretien de la Tomate\n"
            "• Tuteurage à 30-40 cm de hauteur\n"
            "• Supprimer les gourmands régulièrement\n"
            "• Surveiller mildiou (Phytophthora) et alternariose\n"
            "• Traitement fongicide préventif toutes les 2 semaines"
        ),
        "engrais": (
            "🧪 Engrais pour la Tomate\n"
            "• NPK (15-15-15) à 200 kg/ha\n"
            "• Urée 50 kg/ha à 15 jours après transplantation\n"
            "• NPK riche en potassium à la floraison\n"
            "• Éviter l'excès d'azote"
        ),
        "recolte": (
            "🍅 Récolte de la Tomate\n"
            "• 60 à 90 jours après transplantation\n"
            "• Récolter au stade tournant (début de rougissement)\n"
            "• Récolte tous les 2-3 jours pendant 4 à 6 semaines"
        ),
        "eau": (
            "💧 Besoins en eau de la Tomate\n"
            "• Irrigation régulière et uniforme — éviter les à-coups\n"
            "• 400-600 mm/cycle selon variété\n"
            "• Arroser en goutte-à-goutte de préférence\n"
            "• Stade critique : floraison et grossissement des fruits"
        )
    },

    "Fonio": {
        "labour": (
            "🚜 Labour du Fonio\n"
            "• Grattage léger du sol 10-15 cm suffit\n"
            "• Sols pauvres, sableux ou caillouteux acceptés\n"
            "• Zone principale : Atacora (Boukoumbé)"
        ),
        "semis": (
            "🌱 Semis du Fonio\n"
            "• Début saison des pluies (mai-juin Nord-Bénin)\n"
            "• À la volée ou lignes espacées 20-25 cm\n"
            "• 8 à 10 kg de semences/ha\n"
            "• Cycle très court : 6 à 8 semaines"
        ),
        "entretien": (
            "✂️ Entretien du Fonio\n"
            "• 1 sarclage léger à 2-3 semaines après levée\n"
            "• Très résistant à la sécheresse\n"
            "• Surveiller les oiseaux à maturité"
        ),
        "engrais": (
            "🧪 Engrais pour le Fonio\n"
            "• Pousse sur sols pauvres sans engrais : son principal atout\n"
            "• Compost léger 2-3 t/ha améliore le rendement\n"
            "• Éviter l'azote en excès : favorise la paille"
        ),
        "recolte": (
            "🌿 Récolte du Fonio\n"
            "• 6 à 12 semaines selon la variété\n"
            "• Récolter tôt le matin pour limiter l'égrenage\n"
            "• Sécher 2-3 jours avant battage\n"
            "• Décortiqueur recommandé si disponible"
        ),
        "eau": (
            "💧 Besoins en eau du Fonio\n"
            "• Pluviométrie minimale : 300-500 mm/cycle\n"
            "• Très tolérant à la sécheresse\n"
            "• Irrigation rarement nécessaire dans les zones de culture"
        )
    },

    "Mil": {
        "labour": (
            "🚜 Labour du Mil\n"
            "• Labour léger 15-20 cm de profondeur\n"
            "• Adapté aux sols pauvres et sableux du Nord-Bénin\n"
            "• Zones : Borgou, Alibori, Atacora"
        ),
        "semis": (
            "🌱 Semis du Mil\n"
            "• Dès premières pluies régulières (mai-juin Nord-Bénin)\n"
            "• Écartement : 100 cm × 80 cm\n"
            "• 5 à 8 grains par poquet\n"
            "• Démariage : garder 3 à 4 plants"
        ),
        "entretien": (
            "✂️ Entretien du Mil\n"
            "• 1 à 2 sarclages à 20 et 40 jours\n"
            "• Très résistant à la sécheresse\n"
            "• Surveiller les oiseaux (mange-mil) à maturité"
        ),
        "engrais": (
            "🧪 Engrais pour le Mil\n"
            "• NPK (15-15-15) à 100 kg/ha au semis\n"
            "• Urée à 50 kg/ha à 30 jours après levée\n"
            "• Fumier organique 5 t/ha recommandé"
        ),
        "recolte": (
            "🌾 Récolte du Mil\n"
            "• 75 à 120 jours selon la variété\n"
            "• Épis bien formés, grains durs\n"
            "• Sécher 3-5 jours au soleil avant battage"
        ),
        "eau": (
            "💧 Besoins en eau du Mil\n"
            "• Pluviométrie minimale : 200-400 mm/cycle\n"
            "• Culture semi-aride par excellence\n"
            "• Résiste aux longues périodes de sécheresse"
        )
    },

    "Légumineuses": {
        "labour": (
            "🚜 Labour pour les Légumineuses (Soja, Niébé, Arachide)\n"
            "• Labour léger : 15 à 20 cm suffit\n"
            "• Incorporer matière organique si sol pauvre\n"
            "• Éviter les sols très compactés"
        ),
        "semis": (
            "🌱 Semis des Légumineuses\n"
            "• Soja : 50 cm × 10 cm en ligne ou 40 cm × 20 cm en poquet\n"
            "• Niébé : 75 cm × 25 cm, 2 grains par poquet\n"
            "• Arachide : 40 cm × 20 cm, 1-2 graines par poquet\n"
            "• Inoculer avec du rhizobium si disponible"
        ),
        "entretien": (
            "✂️ Entretien des Légumineuses\n"
            "• 1 à 2 sarclages selon pression des adventices\n"
            "• Couvrent vite le sol — limitent les adventices\n"
            "• Éviter les excès d'eau"
        ),
        "engrais": (
            "🧪 Engrais pour les Légumineuses\n"
            "• NPK faible en azote (0-20-20) à 100 kg/ha\n"
            "• PAS d'urée : l'azote inhibe la nodulation\n"
            "• Le phosphore améliore significativement le rendement"
        ),
        "recolte": (
            "🫘 Récolte des Légumineuses\n"
            "• Soja : 90-110 jours, gousses jaunies et sèches\n"
            "• Niébé : 65-80 jours, gousses sèches\n"
            "• Arachide : 90-120 jours, arracher les plants"
        ),
        "eau": (
            "💧 Besoins en eau des Légumineuses\n"
            "• Niébé : très tolérant à la sécheresse, 300-500 mm/cycle\n"
            "• Soja : 500-800 mm/cycle, stade critique : floraison\n"
            "• Arachide : 500-600 mm/cycle\n"
            "• Éviter l'engorgement qui pourrit les racines"
        )
    },

    "Pomme de terre": {
        "labour": (
            "🚜 Labour pour la Pomme de terre\n"
            "• Labour profond : 30 à 40 cm obligatoire\n"
            "• Sol meuble et drainant, former des buttes de 30-40 cm\n"
            "• Incorporer 15-20 t/ha de compost"
        ),
        "semis": (
            "🌱 Plantation de la Pomme de terre\n"
            "• Tubercules-semences sains, pré-germés 2 semaines avant\n"
            "• Profondeur : 10-15 cm, écartement : 70 cm × 30 cm\n"
            "• Saison sèche fraîche pour de meilleurs rendements"
        ),
        "entretien": (
            "✂️ Entretien de la Pomme de terre\n"
            "• 2 butages à 3 et 6 semaines après levée\n"
            "• Irrigation régulière et uniforme\n"
            "• Traitement fongicide préventif contre le mildiou"
        ),
        "engrais": (
            "🧪 Engrais pour la Pomme de terre\n"
            "• NPK riche en potassium (12-12-17) à 300 kg/ha\n"
            "• Urée à 100 kg/ha en 2 apports (3 et 6 semaines)"
        ),
        "recolte": (
            "🥔 Récolte de la Pomme de terre\n"
            "• 90 à 120 jours après plantation\n"
            "• Feuilles jaunies et sèches naturellement\n"
            "• Stocker en local frais, sombre et ventilé"
        ),
        "eau": (
            "💧 Besoins en eau de la Pomme de terre\n"
            "• 500-700 mm/cycle, irrigation régulière indispensable\n"
            "• Éviter les à-coups hydriques qui causent la gale\n"
            "• Stade critique : tubérisation"
        )
    },

    "Canne à sucre": {
        "labour": (
            "🚜 Labour pour la Canne à sucre\n"
            "• Labour très profond : 40 à 50 cm\n"
            "• Former des sillons espacés de 120-150 cm\n"
            "• Incorporer beaucoup de matière organique"
        ),
        "semis": (
            "🌱 Plantation de la Canne à sucre\n"
            "• Boutures de 3 à 4 bourgeons\n"
            "• Horizontalement dans les sillons à 5-8 cm\n"
            "• Début de saison des pluies"
        ),
        "entretien": (
            "✂️ Entretien de la Canne à sucre\n"
            "• Désherbage intensif les 3 premiers mois\n"
            "• Irrigation régulière — très gourmande en eau\n"
            "• Buttage à 2 mois + effeuillage des feuilles mortes"
        ),
        "engrais": (
            "🧪 Engrais pour la Canne à sucre\n"
            "• NPK (15-15-15) à 300 kg/ha à la plantation\n"
            "• Urée à 100 kg/ha à 2 mois, puis 100 kg/ha à 4 mois\n"
            "• KCl à 100 kg/ha pour la teneur en sucre"
        ),
        "recolte": (
            "🎋 Récolte de la Canne à sucre\n"
            "• 10 à 18 mois après plantation\n"
            "• Couper à ras du sol à la machette\n"
            "• Transformer dans les 24-48h après coupe"
        ),
        "eau": (
            "💧 Besoins en eau de la Canne à sucre\n"
            "• Très gourmande : 1500-2500 mm/cycle\n"
            "• Irrigation intensive nécessaire en saison sèche\n"
            "• Stade critique : grand développement végétatif"
        )
    },
}

# Guide générique pour cultures sans fiche détaillée
PRATIQUES_GENERIQUES = {
    "labour":    "🚜 Labour à 20-25 cm de profondeur, sol bien ameubli.",
    "semis":     "🌱 Semences certifiées, profondeur adaptée à la taille de la graine.",
    "entretien": "✂️ 2 à 3 sarclages + surveillance des maladies et ravageurs.",
    "engrais":   "🧪 NPK équilibré au semis + urée en couverture à 30-45 jours.",
    "recolte":   "🌾 Récolter à maturité complète, sécher avant stockage.",
    "eau":       "💧 Adapter l'irrigation aux besoins de la culture selon la saison.",
}

# Titres lisibles pour chaque aspect — utilisés dans les réponses ciblées
TITRES_ASPECTS = {
    "labour":    "🚜 Préparation du sol et labour",
    "semis":     "🌱 Période et technique de semis",
    "entretien": "✂️ Entretien et protection contre les maladies",
    "engrais":   "🧪 Apport d'engrais et fertilisation",
    "recolte":   "🌾 Récolte et stockage",
    "eau":       "💧 Besoins en eau et irrigation",
}


# =============================================================================
# ACTION 1 — ActionRecommander
# =============================================================================

class ActionRecommander(Action):
    """
    Lit la culture et les résultats SHAP depuis les slots transmis par Streamlit.
    NE RECALCULE PAS la prédiction — Streamlit l'a déjà fait et transmis via
    l'API tracker/events. Construit une réponse claire nommant explicitement
    la culture, ses caractéristiques et les raisons de la sélection.
    """

    def name(self) -> Text:
        return "action_recommander"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:

        # Lecture des slots transmis par Streamlit
        culture   = tracker.get_slot("culture_predite")
        top3_shap = tracker.get_slot("top3_shap")
        confiance = tracker.get_slot("confiance")

        if not culture:
            dispatcher.utter_message(text=(
                "🌿 Pour obtenir une recommandation, effectuez d'abord une analyse\n"
                "dans la page **Analyse** de l'application AgriSmart,\n"
                "puis revenez dans le chat."
            ))
            return []

        description = DESCRIPTIONS.get(culture, "Culture adaptée à votre parcelle")
        conf_txt    = f"{confiance}%" if confiance else "N/A"

        # Réponse explicite : nomme la culture, la décrit, cite la confiance
        # et les 3 facteurs décisifs pour une compréhension immédiate
        if top3_shap:
            reponse = (
                f"🌿 **Recommandation : {culture}**\n\n"
                f"📋 {description}\n\n"
                f"🎯 Confiance du modèle : **{conf_txt}**\n\n"
                f"📊 Les 3 facteurs les plus décisifs pour cette recommandation :\n"
                f"{top3_shap}\n\n"
                f"💡 Demandez-moi :\n"
                f"• *'Pourquoi ce choix ?'* pour l'explication détaillée\n"
                f"• *'Les bonnes pratiques pour {culture}'* pour le guide complet\n"
                f"• *'Des alternatives'* pour d'autres options"
            )
        else:
            reponse = (
                f"🌿 **Recommandation : {culture}**\n\n"
                f"📋 {description}\n\n"
                f"🎯 Confiance du modèle : **{conf_txt}**\n\n"
                f"💡 Consultez le graphique waterfall dans la page **Analyse**\n"
                f"pour voir l'influence de chaque paramètre."
            )

        dispatcher.utter_message(text=reponse)
        return []


# =============================================================================
# ACTION 2 — ActionExpliquer
# =============================================================================

class ActionExpliquer(Action):
    """
    Explique la recommandation via les valeurs SHAP mémorisées dans les slots.
    Ces slots ont été remplis par Streamlit — aucun recalcul nécessaire.
    Nomme explicitement la culture et explique les signes + et - des valeurs SHAP.
    """

    def name(self) -> Text:
        return "action_expliquer"

    def run(self, dispatcher, tracker, domain):

        culture   = tracker.get_slot("culture_predite")
        top3_shap = tracker.get_slot("top3_shap")
        confiance = tracker.get_slot("confiance")

        if not culture:
            dispatcher.utter_message(text=(
                "🤔 Aucune recommandation en mémoire.\n\n"
                "Effectuez d'abord une analyse dans la page **Analyse**,\n"
                "puis revenez poser vos questions."
            ))
            return []

        conf_txt = f"{confiance}%" if confiance else "N/A"
        desc     = DESCRIPTIONS.get(culture, "Culture adaptée à votre parcelle")

        reponse = (
            f"🔍 **Pourquoi {culture} a été recommandé ?**\n\n"
            f"📋 {desc}\n\n"
            f"Le modèle XGBoost a analysé vos 14 paramètres de sol et de climat\n"
            f"avec une confiance de **{conf_txt}**.\n\n"
        )

        if top3_shap:
            reponse += (
                f"📊 Les 3 paramètres les plus influents dans cette décision :\n"
                f"{top3_shap}\n\n"
                f"**Comment lire ces valeurs :**\n"
                f"• Signe ➕ → ce paramètre pousse VERS {culture}\n"
                f"• Signe ➖ → ce paramètre joue CONTRE cette culture\n"
                f"• Plus la valeur absolue est grande, plus l'influence est forte\n\n"
            )

        reponse += (
            f"💡 Consultez le graphique waterfall dans la page **Analyse**\n"
            f"pour voir l'influence complète des 14 paramètres."
        )

        dispatcher.utter_message(text=reponse)
        return []


# =============================================================================
# ACTION 3 — ActionAlternatives
# =============================================================================

class ActionAlternatives(Action):
    """
    Propose des cultures alternatives à celle recommandée.
    Exclut la culture principale et présente 3 alternatives avec descriptions.
    """

    def name(self) -> Text:
        return "action_alternatives"

    def run(self, dispatcher, tracker, domain):

        culture_principale = tracker.get_slot("culture_predite")
        alternatives       = [c for c in CULTURES if c != culture_principale]
        selection          = random.sample(alternatives, min(3, len(alternatives)))

        lignes = [
            f"• **{alt}** : {DESCRIPTIONS.get(alt, 'Culture adaptée à certains profils')}"
            for alt in selection
        ]

        reponse = (
            f"🌱 **Cultures alternatives envisageables :**\n\n"
            + "\n".join(lignes)
            + "\n\n💡 Retournez dans la page **Analyse** et ajustez vos paramètres\n"
            "pour explorer ces alternatives avec le modèle XGBoost."
        )

        dispatcher.utter_message(text=reponse)
        return []


# =============================================================================
# ACTION 4 — ActionConseils
# =============================================================================

class ActionConseils(Action):
    """
    Donne des conseils agricoles généraux pour optimiser la parcelle.
    Adapte la réponse selon la culture mémorisée dans les slots.
    """

    def name(self) -> Text:
        return "action_conseils"

    def run(self, dispatcher, tracker, domain):

        culture = tracker.get_slot("culture_predite") or "votre culture"

        reponse = (
            f"🌾 **Conseils pour optimiser votre parcelle pour {culture} :**\n\n"
            f"🧪 Gestion du sol :\n"
            f"• Maintenez un pH entre 5.5 et 7.0\n"
            f"• Apportez de l'azote (N) si la valeur est inférieure à 40 kg/ha\n"
            f"• Un taux de potassium (K) équilibré améliore la résistance\n\n"
            f"🌧️ Gestion de l'eau :\n"
            f"• Adaptez l'irrigation selon les besoins de {culture}\n"
            f"• Évitez les excès d'eau qui favorisent les maladies fongiques\n\n"
            f"🔄 Rotation des cultures :\n"
            f"• Les légumineuses enrichissent naturellement le sol en azote\n"
            f"• Évitez de planter la même culture deux saisons consécutives\n\n"
            f"💡 Demandez-moi *'les bonnes pratiques pour {culture}'*\n"
            f"pour le guide complet : labour, semis, engrais et récolte."
        )

        dispatcher.utter_message(text=reponse)
        return []


# =============================================================================
# ACTION 5 — ActionPratiquesAgricoles (VERSION PRÉCISE ET CONTEXTUELLE)
# =============================================================================

class ActionPratiquesAgricoles(Action):
    """
    Donne les bonnes pratiques agricoles pour la culture recommandée.

    RÉPONSE PRÉCISE ET CONTEXTUELLE :
    Si l'utilisateur pose une question spécifique (période de semis, besoins
    en eau, type d'engrais...), on retourne UNIQUEMENT la section pertinente
    au lieu de la fiche complète. Cela évite la surcharge d'information.

    Si la question est générale ("bonnes pratiques", "guide complet"), on
    retourne la fiche complète avec toutes les sections.

    Détection de la culture :
    1. D'abord dans les slots (transmis par Streamlit)
    2. Ensuite dans le message de l'utilisateur (si pas de slot)
    """

    def name(self) -> Text:
        return "action_pratiques_agricoles"

    def run(self, dispatcher, tracker, domain):

        # ÉTAPE 1 : chercher la culture dans les slots
        culture = tracker.get_slot("culture_predite")

        # ÉTAPE 2 : si pas de slot, chercher dans le message de l'utilisateur
        if not culture:
            message = tracker.latest_message.get("text", "").lower()
            for c in CULTURES:
                if c.lower() in message:
                    culture = c
                    break

        # ÉTAPE 3 : si toujours rien, demander à l'utilisateur
        if not culture:
            dispatcher.utter_message(text=(
                "🌿 Quelle culture vous intéresse ?\n\n"
                "Précisez la culture dans votre question ou effectuez\n"
                "d'abord une analyse dans la page **Analyse**.\n\n"
                "Exemple : *'Comment cultiver le maïs ?'*"
            ))
            return []

        # ÉTAPE 4 : détecter si l'utilisateur veut un aspect spécifique
        message = tracker.latest_message.get("text", "").lower()
        aspect  = detecter_aspect(message)

        pratiques = PRATIQUES_AGRICOLES.get(culture, None)

        if aspect and pratiques and aspect in pratiques:
            # Réponse ciblée : uniquement l'aspect demandé
            titre   = TITRES_ASPECTS.get(aspect, f"🌱 {aspect.capitalize()}")
            reponse = (
                f"**{titre} — {culture}**\n\n"
                f"{pratiques[aspect]}\n\n"
                f"💡 Demandez-moi *'guide complet pour {culture}'* "
                f"pour toutes les étapes (labour, semis, engrais, récolte)."
            )

        elif aspect and not pratiques:
            # Culture sans fiche + aspect spécifique → réponse générique ciblée
            reponse = (
                f"**{TITRES_ASPECTS.get(aspect, aspect)} — {culture}**\n\n"
                f"{PRATIQUES_GENERIQUES.get(aspect, 'Information non disponible.')}\n\n"
                f"💡 Consultez votre centre agricole local pour des informations\n"
                f"spécifiques à {culture}."
            )

        elif pratiques:
            # Pas d'aspect spécifique → fiche complète
            reponse = (
                f"📚 **Guide de production — {culture}**\n\n"
                f"{pratiques['labour']}\n\n"
                f"{'─' * 35}\n\n"
                f"{pratiques['semis']}\n\n"
                f"{'─' * 35}\n\n"
                f"{pratiques['entretien']}\n\n"
                f"{'─' * 35}\n\n"
                f"{pratiques['engrais']}\n\n"
                f"{'─' * 35}\n\n"
                f"{pratiques['recolte']}\n\n"
                f"💡 Ces recommandations sont adaptées aux conditions\n"
                f"agro-climatiques du Bénin."
            )

        else:
            # Culture sans fiche + pas d'aspect → guide générique complet
            reponse = (
                f"📚 **Guide de production général — {culture}**\n\n"
                f"{PRATIQUES_GENERIQUES['labour']}\n\n"
                f"{PRATIQUES_GENERIQUES['semis']}\n\n"
                f"{PRATIQUES_GENERIQUES['entretien']}\n\n"
                f"{PRATIQUES_GENERIQUES['engrais']}\n\n"
                f"{PRATIQUES_GENERIQUES['recolte']}\n\n"
                f"💡 Consultez votre centre agricole local pour des\n"
                f"recommandations spécifiques à {culture}."
            )

        dispatcher.utter_message(text=reponse)
        return []


# =============================================================================
# ACTION 6 — ActionConseilSaison
# =============================================================================

class ActionConseilSaison(Action):
    """
    Donne des conseils selon la saison mémorisée dans les slots.
    Le slot 'saison' contient Kharif/Rabi/Zaid (traduit par Streamlit
    depuis les saisons béninoises). On renvoie les réponses fixes
    définies dans domain.yml.
    """

    def name(self) -> Text:
        return "action_conseil_saison"

    def run(self, dispatcher, tracker, domain):

        # saison contient Kharif/Rabi/Zaid — traduit silencieusement par Streamlit
        saison = tracker.get_slot("saison")

        if saison == "Kharif":
            dispatcher.utter_message(response="utter_conseil_saison_pluies")

        elif saison == "Zaid":
            dispatcher.utter_message(response="utter_conseil_petite_saison_seche")

        elif saison == "Rabi":
            dispatcher.utter_message(response="utter_conseil_saison_seche")

        else:
            # Saison non définie dans les slots
            dispatcher.utter_message(text=(
                "🌤️ **Conseils selon les saisons agricoles du Bénin :**\n\n"
                "🌧️ Saison des pluies (Nord) / Grande saison des pluies (Sud)\n"
                "Idéale pour : Maïs, Riz, Coton, Sorgho, Légumineuses\n\n"
                "☀️ Petite saison sèche (Sud-Bénin)\n"
                "Idéale pour : Tomate, Légumes feuilles à cycle court\n\n"
                "🌵 Saison sèche (Nord) / Grande saison sèche (Sud)\n"
                "Idéale pour : Tomate irriguée, Fonio, Mil, Maraîchage\n\n"
                "💡 Renseignez votre saison dans la page **Analyse**\n"
                "pour des conseils personnalisés."
            ))

        return []


# =============================================================================
# ACTION 7 — ActionInfosCulture (PRÉCISE ET CONTEXTUELLE)
# =============================================================================

class ActionInfosCulture(Action):
    """
    Répond aux questions générales ou spécifiques sur une culture donnée.

    RÉPONSE PRÉCISE ET CONTEXTUELLE :
    Détecte d'abord si l'utilisateur demande un aspect spécifique (période,
    eau, engrais...) et retourne uniquement cette information.
    Si la question est générale, retourne une fiche synthétique.

    Détection de la culture dans le message ou dans les slots.
    """

    def name(self) -> Text:
        return "action_infos_culture"

    def run(self, dispatcher, tracker, domain):

        message = tracker.latest_message.get("text", "").lower()

        # Recherche du nom de culture dans le message
        culture_trouvee = None
        for c in CULTURES:
            if c.lower() in message:
                culture_trouvee = c
                break

        # Si pas dans le message, essai dans les slots
        if not culture_trouvee:
            culture_trouvee = tracker.get_slot("culture_predite")

        if not culture_trouvee:
            dispatcher.utter_message(text=(
                "🌿 Quelle culture souhaitez-vous connaître ?\n\n"
                "Cultures disponibles dans notre système :\n"
                "Maïs, Riz, Sorgho, Coton, Tomate, Fonio, Mil,\n"
                "Légumineuses, Pomme de terre, Canne à sucre\n\n"
                "Exemple : *'Parle-moi du sorgho'*"
            ))
            return []

        description = DESCRIPTIONS.get(culture_trouvee, "Culture adaptée au Bénin")
        pratiques   = PRATIQUES_AGRICOLES.get(culture_trouvee, None)

        # Détecter si l'utilisateur veut un aspect spécifique
        aspect = detecter_aspect(message)

        if aspect and pratiques and aspect in pratiques:
            # Réponse ciblée sur l'aspect demandé
            titre   = TITRES_ASPECTS.get(aspect, aspect.capitalize())
            reponse = (
                f"🌿 **{culture_trouvee} — {titre}**\n\n"
                f"{pratiques[aspect]}\n\n"
                f"💡 Demandez-moi *'guide complet pour {culture_trouvee}'*\n"
                f"pour toutes les étapes de production."
            )

        elif pratiques:
            # Fiche synthétique : description + semis + récolte résumés
            reponse = (
                f"🌿 **{culture_trouvee}**\n\n"
                f"📋 {description}\n\n"
                f"{pratiques['semis']}\n\n"
                f"{'─' * 35}\n\n"
                f"{pratiques['recolte']}\n\n"
                f"💡 Posez-moi des questions précises comme :\n"
                f"• *'Quand semer {culture_trouvee} ?'*\n"
                f"• *'Quel engrais pour {culture_trouvee} ?'*\n"
                f"• *'Besoins en eau de {culture_trouvee}'*\n"
                f"• *'Guide complet de {culture_trouvee}'*"
            )

        else:
            reponse = (
                f"🌿 **{culture_trouvee}**\n\n"
                f"📋 {description}\n\n"
                f"💡 Consultez votre centre agricole local pour des\n"
                f"informations détaillées sur cette culture."
            )

        dispatcher.utter_message(text=reponse)
        return []
