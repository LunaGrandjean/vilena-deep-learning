# Comparaison des modèles

Même jeu d'images, même séparation stratifiée, même graine et même protocole CPU.

| Modèle | Accuracy test | F1 macro | CPU modèle (ms) | CPU total médian (ms) | CPU total p95 (ms) | Entraînement (s) | Taille (Mo) |
|---|---:|---:|---:|---:|---:|---:|---:|
| mlp | 38.2% | 0.363 | 0.293 | 0.926 | 1.045 | 55.82 (cpu) | 12.72 |
| cnn | 60.9% | 0.602 | 7.029 | 7.670 | 8.407 | 1006.86 (cpu) | 0.39 |
| resnet18 | 80.9% | 0.809 | 43.204 | 42.510 | 48.894 | 3488.64 (cpu) | 44.79 |
| classical | 41.8% | 0.398 | 0.264 | 1.880 | 2.286 | 1.43 (cpu) | 0.04 |

- Meilleure accuracy observée : resnet18 (80.9%).
- Inférence complète la plus rapide : mlp (0.926 ms en médiane).
- Compromis non dominés en accuracy/latence médiane : mlp, cnn, resnet18, classical.

Le temps total inclut ouverture/décodage JPEG, prétraitement et prédiction pour une seule image. Le temps modèle seul utilise une entrée déjà préparée. Le processus et les caches sont chauds, le chargement initial des poids est exclu. Les tailles mesurent les fichiers sérialisés, pas le pic de RAM.

Les temps d'entraînement couvrent toutes les époques (et leurs validations) pour les réseaux, et tous les candidats C pour la régression logistique. La préparation des données/caractéristiques est séparée dans le CSV. Les architectures utilisent des prétraitements adaptés (tailles et normalisation enregistrées dans leurs métadonnées) ; les écarts reflètent donc les pipelines complets, pas seulement les architectures.

Pour recommander un modèle au musée, fixer d'abord la qualité minimale et le budget de latence, puis confirmer ces mesures sur le CPU des bornes. Le meilleur score test seul ne suffit pas à trancher. Aucun choix de réglages ne doit être fait à partir des erreurs test. Ces résultats portent sur une graine et une séparation par image ; ils ne prouvent ni la stabilité sur plusieurs séparations ni la généralisation à de nouveaux artistes.
